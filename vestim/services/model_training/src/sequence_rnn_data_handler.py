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

    def __init__(self, feature_cols, target_col, lookback, concatenate_raw_data=False, pad_beginning: bool = False):
        """
        :param feature_cols: List of feature column names.
        :param target_col: Target column name.
        :param lookback: The lookback window size.
        :param concatenate_raw_data: If True, concatenates raw data from all files 
                                     before creating sequences. Otherwise, creates sequences
                                     per file and then concatenates the sequences.
        :param pad_beginning: If True, pad the beginning with the first row repeated
                              `lookback` times so that the very first target sample is
                              included as a sequence (mirrors test-time padding behavior).
        """
        super().__init__(feature_cols, target_col)
        if not isinstance(lookback, int) or lookback <= 0:
            raise ValueError("lookback must be a positive integer.")
        self.lookback = lookback
        self.concatenate_raw_data = concatenate_raw_data
        self.pad_beginning = pad_beginning
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
        # Optionally pad the beginning with the first row repeated `lookback` times
        if self.pad_beginning and len(X_data_arr) > 0:
            if self.logger:
                self.logger.info(f"Padding beginning: Adding {self.lookback} repeated rows to preserve earliest {self.lookback} targets")
            pad_X = np.repeat(X_data_arr[0:1, :], repeats=self.lookback, axis=0)
            # For physical consistency, zero out dynamic columns (current/power) in the padded rows
            zero_cols = [i for i, name in enumerate(self.feature_cols) if isinstance(name, str) and (('current' in name.lower()) or ('power' in name.lower()) or name.lower() in ('i','p'))]
            if zero_cols:
                pad_X[:, zero_cols] = 0.0
                if self.logger:
                    zero_names = [self.feature_cols[i] for i in zero_cols]
                    self.logger.info(f"Zeroed padded columns for consistency: {zero_names}")

            pad_Y = np.repeat(Y_data_arr[0:1, ...], repeats=self.lookback, axis=0)
            X_src = np.concatenate([pad_X.astype(X_data_arr.dtype, copy=False), X_data_arr], axis=0)
            Y_src = np.concatenate([pad_Y.astype(Y_data_arr.dtype, copy=False), Y_data_arr], axis=0)
            if self.logger:
                self.logger.info(f"After padding: X length {len(X_data_arr)} -> {len(X_src)}, will create {len(X_src) - self.lookback} sequences")
        else:
            X_src = X_data_arr
            Y_src = Y_data_arr

        num_sequences = len(X_src) - self.lookback
        if num_sequences <= 0:
            if self.logger:
                self.logger.warning(f"Data length ({len(X_src)}) is less than or equal to lookback ({self.lookback}). No sequences created.")
            return np.empty((0, self.lookback, X_data_arr.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.float32)

        # Ensure input arrays are float32 for memory efficiency
        if X_src.dtype != np.float32:
            X_src = X_src.astype(np.float32)
        if Y_src.dtype != np.float32:
            Y_src = Y_src.astype(np.float32)

        # Pre-allocate arrays with exact size needed - NO MEMORY SPIKE!
        X_sequences = np.empty((num_sequences, self.lookback, X_src.shape[1]), dtype=np.float32)
        y_sequences = np.empty((num_sequences,), dtype=np.float32)
        
        # Fill pre-allocated arrays directly (no memory growth during loop!)
        # Use vectorized operations where possible for better performance
        for i in range(num_sequences):
            X_sequences[i] = X_src[i:i + self.lookback, :]  # Direct slice copy
            y_sequences[i] = Y_src[i + self.lookback]
        
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
                
                if row_count > 0:
                    # If padding beginning, each row becomes a target; else we lose `lookback` initial targets
                    if self.pad_beginning:
                        total_sequences += row_count
                    elif row_count > self.lookback:
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
            if df_selected is None or df_selected.empty:
                continue
            # When padding at the beginning, allow short files (<= lookback) as they can still form sequences
            if (not self.pad_beginning) and (len(df_selected) <= self.lookback):
                continue

            X_data_file = df_selected[self.feature_cols].values.astype(np.float32)
            Y_data_file = df_selected[self.target_col].values.astype(np.float32)

            num_file_sequences = len(X_data_file) if self.pad_beginning else max(0, len(X_data_file) - self.lookback)
            
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