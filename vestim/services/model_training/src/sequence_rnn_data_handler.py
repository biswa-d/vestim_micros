import os
import numpy as np
import pandas as pd
import gc
import logging
from vestim.services.model_training.src.base_data_handler import BaseDataHandler

class SequenceRNNDataHandler(BaseDataHandler):
    """
    Memory-efficient data handler for creating lookback-based sequences suitable for RNN models.
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
        self.logger = logging.getLogger(__name__)

    def _create_sequences_from_array(self, X_data_arr: np.ndarray, Y_data_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Creates input-output sequences from given X and Y numpy arrays based on the lookback period.
        Memory-efficient implementation using pre-allocated arrays instead of Python lists.
        
        Memory optimizations:
        1. Pre-allocated numpy arrays (eliminates list-to-array conversion spike)
        2. Direct memory copying (no temporary objects)
        3. Optimal data types (float32 instead of float64)
        4. Memory usage logging
        """
        num_sequences = len(X_data_arr) - self.lookback
        if num_sequences <= 0:
            if self.logger:
                self.logger.warning(f"Data length ({len(X_data_arr)}) is less than or equal to lookback ({self.lookback}). No sequences created.")
            return np.empty((0, self.lookback, X_data_arr.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.float32)

        # Ensure input arrays are float32 for memory efficiency
        if X_data_arr.dtype != np.float32:
            X_data_arr = X_data_arr.astype(np.float32)
        if Y_data_arr.dtype != np.float32:
            Y_data_arr = Y_data_arr.astype(np.float32)

        # Pre-allocate arrays with exact size needed - NO MEMORY SPIKE!
        X_sequences = np.empty((num_sequences, self.lookback, X_data_arr.shape[1]), dtype=np.float32)
        y_sequences = np.empty((num_sequences,), dtype=np.float32)
        
        # Fill pre-allocated arrays directly (no memory growth during loop!)
        # Use vectorized operations where possible for better performance
        for i in range(num_sequences):
            X_sequences[i] = X_data_arr[i:i + self.lookback, :]  # Direct slice copy
            y_sequences[i] = Y_data_arr[i + self.lookback]
        
        return X_sequences, y_sequences

    def load_and_process_data(self, folder_path: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """
        Loads data from CSV files and creates lookback sequences in a memory-efficient manner.
        This approach pre-calculates the total size needed to avoid large intermediate lists.
        """
        csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
        if not csv_files:
            self.logger.warning(f"No CSV files found in {folder_path}.")
            return np.empty((0, self.lookback, len(self.feature_cols))), np.empty((0,))

        # --- First pass: Calculate total number of sequences to pre-allocate memory ---
        self.logger.info("Calculating total number of sequences to pre-allocate memory...")
        total_sequences = 0
        file_row_counts = {}
        for file_path in csv_files:
            try:
                # Fast way to get row count without loading the whole file into a DataFrame
                with open(file_path, 'r') as f:
                    # Subtract 1 for the header
                    row_count = sum(1 for row in f) - 1
                
                if row_count > self.lookback:
                    total_sequences += row_count - self.lookback
                    file_row_counts[file_path] = row_count
            except Exception as e:
                self.logger.error(f"Could not read or count rows in {file_path}: {e}")
                continue
        
        if total_sequences == 0:
            self.logger.warning("No sequences could be created from the provided files.")
            return np.empty((0, self.lookback, len(self.feature_cols))), np.empty((0,))

        # --- Pre-allocate final arrays ---
        self.logger.info(f"Pre-allocating memory for a total of {total_sequences} sequences.")
        X_processed = np.empty((total_sequences, self.lookback, len(self.feature_cols)), dtype=np.float32)
        y_processed = np.empty((total_sequences,), dtype=np.float32)
        
        # --- Second pass: Load data, create sequences, and fill arrays ---
        current_sequence_idx = 0
        for file_idx, file_path in enumerate(csv_files):
            if file_path not in file_row_counts:
                continue # Skip files that were invalid in the first pass

            df_selected = self._read_and_select_columns(file_path)
            if df_selected is None or df_selected.empty or len(df_selected) <= self.lookback:
                continue

            X_data_file = df_selected[self.feature_cols].values.astype(np.float32)
            Y_data_file = df_selected[self.target_col].values.astype(np.float32)

            num_file_sequences = len(X_data_file) - self.lookback
            
            # Create sequences for this file
            X_file_seq, y_file_seq = self._create_sequences_from_array(X_data_file, Y_data_file)
            
            if X_file_seq.size > 0:
                # Place the generated sequences directly into the pre-allocated array
                end_idx = current_sequence_idx + num_file_sequences
                X_processed[current_sequence_idx:end_idx] = X_file_seq
                y_processed[current_sequence_idx:end_idx] = y_file_seq
                current_sequence_idx = end_idx

            # Aggressive cleanup
            del df_selected, X_data_file, Y_data_file, X_file_seq, y_file_seq
            gc.collect()

        self.logger.info(f"Finished processing all files. Final shapes: X={X_processed.shape}, y={y_processed.shape}")
        return X_processed, y_processed