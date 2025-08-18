import os
import numpy as np
import pandas as pd
import gc
import logging # Assuming handlers might log independently or get a logger passed
from vestim.services.model_training.src.base_data_handler import BaseDataHandler

class WholeSequenceFNNDataHandler(BaseDataHandler):
    """
    Data handler for concatenating all time-step data from multiple files
    into a single dataset. Suitable for FNNs or other models that process
    time steps independently or take the entire dataset as one sequence.
    The 'lookback' concept for windowing is not applied here.
    """

    def __init__(self, feature_cols, target_col):
        super().__init__(feature_cols, target_col)
        self.logger = logging.getLogger(__name__) # Or get it passed in


    def load_and_process_data(self, folder_path: str, return_timestamp: bool = False, **kwargs) -> tuple:
        """
        Loads data from CSV files, concatenates them row-wise.
        The `lookback` parameter from kwargs is ignored by this handler.
        
        :param folder_path: Path to the folder containing CSV files.
        :param return_timestamp: If True, returns timestamps along with X and y data.
        :param kwargs: Additional arguments (lookback is ignored).
        :return: A tuple (X_data_processed, y_data_processed, timestamps) as numpy arrays.
                 Shape of X is [total_timesteps, num_features].
                 Shape of y is [total_timesteps, num_output_features] (typically [N,1]).
                 Shape of timestamps is [total_timesteps,].
        """
        csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
        if not csv_files:
            if self.logger:
                self.logger.warning(f"No CSV files found in {folder_path}.")
            num_features = len(self.feature_cols) if self.feature_cols else 0
            # For FNN style, y is typically [N,1] or [N, num_targets]
            num_target_features = 1 # Assuming single target for now
            return np.empty((0, num_features)), np.empty((0, num_target_features))


        all_X_data_list = []
        all_Y_data_list = []
        all_timestamps_list = []

        for file_path in csv_files:
            df_selected = self._read_and_select_columns(file_path)
            if df_selected is None or df_selected.empty:
                continue
            
            # Convert to numpy arrays
            # Features are expected to be [timesteps, num_features]
            X_data_file = df_selected[self.feature_cols].values
            # Target is expected to be [timesteps, 1] or [timesteps, num_output_features]
            Y_data_file = df_selected[[self.target_col]].values # Ensure Y_data_file is 2D

            all_X_data_list.append(X_data_file)
            all_Y_data_list.append(Y_data_file)
            
            if return_timestamp:
                if 'Timestamp' in df_selected.columns:
                    timestamps = df_selected['Timestamp'].values
                    all_timestamps_list.append(timestamps)
                else:
                    # If timestamp is requested but not present, we should handle it gracefully.
                    # For now, we'll append None and handle it during concatenation.
                    # A better approach might be to log a warning.
                    all_timestamps_list.append(None)

            del df_selected, X_data_file, Y_data_file
            if return_timestamp and 'timestamps' in locals():
                del timestamps
            gc.collect()

        if not all_X_data_list: # No valid data read from any file
            num_features = len(self.feature_cols) if self.feature_cols else 0
            num_target_features = 1 # Assuming single target
            return np.empty((0, num_features)), np.empty((0, num_target_features))

        # Concatenate all data row-wise
        X_processed = np.concatenate(all_X_data_list, axis=0)
        y_processed = np.concatenate(all_Y_data_list, axis=0)
        
        if return_timestamp:
            timestamps_processed = np.concatenate(all_timestamps_list, axis=0)
        else:
            timestamps_processed = None

        del all_X_data_list, all_Y_data_list, all_timestamps_list
        gc.collect()

        if self.logger:
            self.logger.info(f"WholeSequenceFNNDataHandler: Processed X shape: {X_processed.shape}, y shape: {y_processed.shape}")
        return X_processed, y_processed, timestamps_processed