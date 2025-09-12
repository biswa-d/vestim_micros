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
        
        # Check for data leakage: target column in features
        if target_col in feature_cols:
            if hasattr(self, 'logger') and self.logger:
                self.logger.warning(f"DATA LEAKAGE WARNING: Target column '{target_col}' is also in feature columns. "
                                  f"This may cause data leakage where the model learns to predict using the actual target values. "
                                  f"Consider removing '{target_col}' from features or using a different target column.")
            else:
                print(f"DATA LEAKAGE WARNING: Target column '{target_col}' is also in feature columns. "
                     f"This may cause data leakage where the model learns to predict using the actual target values. "
                     f"Consider removing '{target_col}' from features or using a different target column.")
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
            
            # Debug: Check column availability before extraction
            missing_feature_cols = [col for col in self.feature_cols if col not in df_selected.columns]
            if missing_feature_cols:
                error_msg = f"Missing feature columns in data: {missing_feature_cols}. Available columns: {list(df_selected.columns)}. Feature columns requested: {self.feature_cols}"
                if hasattr(self, 'logger') and self.logger:
                    self.logger.error(error_msg)
                else:
                    print(f"ERROR: {error_msg}")
                raise ValueError(error_msg)
            
            X_data_file = df_selected[self.feature_cols].values
            # Target is expected to be [timesteps, 1] or [timesteps, num_output_features]
            Y_data_file = df_selected[[self.target_col]].values # Ensure Y_data_file is 2D

            # Debug: Check the actual shapes
            if hasattr(self, 'logger') and self.logger:
                self.logger.info(f"DEBUG: feature_cols = {self.feature_cols}")
                self.logger.info(f"DEBUG: target_col = {self.target_col}")
                self.logger.info(f"DEBUG: df_selected.columns = {list(df_selected.columns)}")
                self.logger.info(f"DEBUG: X_data_file.shape = {X_data_file.shape}")
                self.logger.info(f"DEBUG: Y_data_file.shape = {Y_data_file.shape}")

            all_X_data_list.append(X_data_file)
            all_Y_data_list.append(Y_data_file)
            
            if return_timestamp:
                if 'Timestamp' in df_selected.columns:
                    timestamps = df_selected['Timestamp'].values
                    all_timestamps_list.append(timestamps)
                else:
                    all_timestamps_list.append(np.array([])) # Append empty array if no timestamp

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