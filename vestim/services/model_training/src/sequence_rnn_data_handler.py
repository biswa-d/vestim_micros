import os
import numpy as np
import pandas as pd
import gc
import logging # Assuming handlers might log independently or get a logger passed
from vestim.services.model_training.src.base_data_handler import BaseDataHandler

class SequenceRNNDataHandler(BaseDataHandler):
    """
    Data handler for creating lookback-based sequences suitable for RNN models.
    """

    def __init__(self, feature_cols, target_col, lookback, concatenate_raw_data=False):
        """
        :param feature_cols: List of feature column names.
        :param target_col: Target column name.
        :param lookback: The lookback window size.
        :param concatenate_raw_data: If True, concatenates raw data from all files 
                                     before creating sequences. Otherwise, creates sequences
                                     per file and then concatenates the sequences.
        """
        super().__init__(feature_cols, target_col)
        if not isinstance(lookback, int) or lookback <= 0:
            raise ValueError("lookback must be a positive integer.")
        self.lookback = lookback
        self.concatenate_raw_data = concatenate_raw_data
        self.logger = logging.getLogger(__name__) # Or get it passed in

    def _create_sequences_from_array(self, X_data_arr: np.ndarray, Y_data_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Creates input-output sequences from given X and Y numpy arrays based on the lookback period.
        """
        X_sequences, y_sequences = [], []
        if len(X_data_arr) <= self.lookback:
            if self.logger:
                self.logger.warning(f"Data length ({len(X_data_arr)}) is less than or equal to lookback ({self.lookback}). No sequences created.")
            return np.array(X_sequences), np.array(y_sequences) # Return empty arrays matching expected type

        for i in range(self.lookback, len(X_data_arr)):
            X_sequences.append(X_data_arr[i - self.lookback:i, :]) # X_data_arr is already [timesteps, features]
            y_sequences.append(Y_data_arr[i])
        
        if not X_sequences: # If loop didn't run, return empty arrays of correct dimension
             return np.empty((0, self.lookback, X_data_arr.shape[1])), np.empty((0,))

        return np.array(X_sequences), np.array(y_sequences)

    def load_and_process_data(self, folder_path: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """
        Loads data from CSV files, creates lookback sequences.
        Supports two modes of operation based on `self.concatenate_raw_data`:
        1. False (default): Creates sequences from each file, then concatenates these sequence arrays.
        2. True: Concatenates raw data from all files, then creates sequences from the single large array.
        """
        csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
        if not csv_files:
            if self.logger:
                self.logger.warning(f"No CSV files found in {folder_path}.")
            # Return empty arrays with correct dimensions for sequence data
            # Assuming num_features can be inferred if feature_cols is not empty, else default to 0 or raise error
            num_features = len(self.feature_cols) if self.feature_cols else 0
            return np.empty((0, self.lookback, num_features)), np.empty((0,))


        all_X_data_raw_list = []
        all_Y_data_raw_list = []
        
        all_X_sequences_list = []
        all_y_sequences_list = []

        for file_path in csv_files:
            df_selected = self._read_and_select_columns(file_path)
            if df_selected is None or df_selected.empty:
                continue

            X_data_file = df_selected[self.feature_cols].values
            Y_data_file = df_selected[self.target_col].values.reshape(-1, 1) # Ensure Y is 2D [N,1] for consistency

            if self.concatenate_raw_data:
                all_X_data_raw_list.append(X_data_file)
                all_Y_data_raw_list.append(Y_data_file)
            else: # Create sequences per file
                if X_data_file.shape[0] > self.lookback:
                    X_file_seq, y_file_seq = self._create_sequences_from_array(X_data_file, Y_data_file.flatten()) # flatten Y for this method
                    if X_file_seq.size > 0: # Check if any sequences were actually created
                        all_X_sequences_list.append(X_file_seq)
                        all_y_sequences_list.append(y_file_seq)
                else:
                    if self.logger:
                        self.logger.warning(f"File {file_path} has insufficient data (length {X_data_file.shape[0]}) for lookback {self.lookback}. Skipping sequence creation for this file.")
            
            del df_selected, X_data_file, Y_data_file
            gc.collect()

        if self.concatenate_raw_data:
            if not all_X_data_raw_list: # No valid data read from any file
                num_features = len(self.feature_cols) if self.feature_cols else 0
                return np.empty((0, self.lookback, num_features)), np.empty((0,))
            
            X_super_sequence = np.concatenate(all_X_data_raw_list, axis=0)
            Y_super_sequence = np.concatenate(all_Y_data_raw_list, axis=0).flatten() # flatten Y for _create_sequences_from_array
            
            # Clear raw lists
            del all_X_data_raw_list, all_Y_data_raw_list
            gc.collect()

            X_processed, y_processed = self._create_sequences_from_array(X_super_sequence, Y_super_sequence)
            del X_super_sequence, Y_super_sequence
        else: # Concatenate lists of sequence arrays
            if not all_X_sequences_list: # No sequences created from any file
                num_features = len(self.feature_cols) if self.feature_cols else 0
                return np.empty((0, self.lookback, num_features)), np.empty((0,))

            X_processed = np.concatenate(all_X_sequences_list, axis=0)
            y_processed = np.concatenate(all_y_sequences_list, axis=0)
            # Clear sequence lists
            del all_X_sequences_list, all_y_sequences_list
        
        gc.collect()
        if self.logger:
            self.logger.info(f"SequenceRNNDataHandler: Processed X shape: {X_processed.shape}, y shape: {y_processed.shape}")
        return X_processed, y_processed