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
        
        if self.logger:
            memory_mb = (X_sequences.nbytes + y_sequences.nbytes) / 1024 / 1024
            self.logger.info(f"Pre-allocated sequence arrays: X_sequences shape: {X_sequences.shape}, y_sequences shape: {y_sequences.shape}, Memory: {memory_mb:.1f} MB")
        
        # Fill pre-allocated arrays directly (no memory growth during loop!)
        # Use vectorized operations where possible for better performance
        for i in range(num_sequences):
            X_sequences[i] = X_data_arr[i:i + self.lookback, :]  # Direct slice copy
            y_sequences[i] = Y_data_arr[i + self.lookback]
        
        return X_sequences, y_sequences

    def load_and_process_data(self, folder_path: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """
        Loads data from CSV files, creates lookback sequences.
        Memory-optimized approach with pre-allocation and aggressive cleanup.
        """
        csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
        if not csv_files:
            if self.logger:
                self.logger.warning(f"No CSV files found in {folder_path}.")
            num_features = len(self.feature_cols) if self.feature_cols else 0
            return np.empty((0, self.lookback, num_features)), np.empty((0,))

        return self._load_and_process_data_legacy(csv_files)


    def _load_and_process_data_legacy(self, csv_files):
        """
        Memory-optimized legacy approach - preserves temporal continuity.
        Supports two modes of operation based on `self.concatenate_raw_data`:
        1. False (default): Creates sequences from each file, then concatenates these sequence arrays.
        2. True: Concatenates raw data from all files, then creates sequences from the single large array.
        
        Memory optimizations applied:
        - Pre-allocated numpy arrays (no list-to-array conversion spike)
        - Immediate cleanup of intermediate variables
        - Aggressive garbage collection
        - Dtype optimization
        - Memory usage logging
        """

        all_X_data_raw_list = []
        all_Y_data_raw_list = []
        
        all_X_sequences_list = []
        all_y_sequences_list = []
        
        # Track memory usage if logger available
        initial_memory = None
        if self.logger:
            try:
                import psutil
                process = psutil.Process()
                initial_memory = process.memory_info().rss / 1024 / 1024  # MB
                self.logger.info(f"Starting memory usage: {initial_memory:.1f} MB")
            except ImportError:
                pass

        for file_idx, file_path in enumerate(csv_files):
            if self.logger:
                self.logger.info(f"Processing file {file_idx+1}/{len(csv_files)}: {file_path}")
            
            df_selected = self._read_and_select_columns(file_path)
            if df_selected is None or df_selected.empty:
                continue

            # Memory optimization: Use optimal dtypes
            X_data_file = df_selected[self.feature_cols].values.astype(np.float32)  # Use float32 instead of float64
            Y_data_file = df_selected[self.target_col].values.astype(np.float32).reshape(-1, 1)

            if self.concatenate_raw_data:
                all_X_data_raw_list.append(X_data_file)
                all_Y_data_raw_list.append(Y_data_file)
            else: # Create sequences per file
                if X_data_file.shape[0] > self.lookback:
                    X_file_seq, y_file_seq = self._create_sequences_from_array(X_data_file, Y_data_file.flatten())
                    if X_file_seq.size > 0:
                        all_X_sequences_list.append(X_file_seq)
                        all_y_sequences_list.append(y_file_seq)
                else:
                    if self.logger:
                        self.logger.warning(f"File {file_path} has insufficient data (length {X_data_file.shape[0]}) for lookback {self.lookback}. Skipping sequence creation for this file.")
            
            # Aggressive cleanup after each file
            del df_selected, X_data_file, Y_data_file
            gc.collect()
            
            # Log memory usage periodically
            if self.logger and initial_memory and file_idx % 5 == 0:
                try:
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    self.logger.info(f"Memory after file {file_idx+1}: {current_memory:.1f} MB (+{current_memory-initial_memory:.1f} MB)")
                except:
                    pass

        if self.concatenate_raw_data:
            if not all_X_data_raw_list: # No valid data read from any file
                num_features = len(self.feature_cols) if self.feature_cols else 0
                return np.empty((0, self.lookback, num_features)), np.empty((0,))
            
            if self.logger:
                total_rows = sum(arr.shape[0] for arr in all_X_data_raw_list)
                self.logger.info(f"Concatenating {len(all_X_data_raw_list)} arrays with {total_rows} total rows")
            
            # Memory optimization: Pre-calculate total size and allocate once
            total_rows = sum(arr.shape[0] for arr in all_X_data_raw_list)
            n_features = all_X_data_raw_list[0].shape[1]
            
            # Pre-allocate concatenated arrays
            X_super_sequence = np.empty((total_rows, n_features), dtype=np.float32)
            Y_super_sequence = np.empty((total_rows, 1), dtype=np.float32)
            
            # Fill pre-allocated arrays
            current_row = 0
            for X_arr, Y_arr in zip(all_X_data_raw_list, all_Y_data_raw_list):
                end_row = current_row + X_arr.shape[0]
                X_super_sequence[current_row:end_row] = X_arr
                Y_super_sequence[current_row:end_row] = Y_arr
                current_row = end_row
            
            # Clear raw lists immediately to free memory
            del all_X_data_raw_list, all_Y_data_raw_list
            gc.collect()
            
            if self.logger:
                self.logger.info(f"Created super sequences: X shape: {X_super_sequence.shape}, Y shape: {Y_super_sequence.shape}")

            X_processed, y_processed = self._create_sequences_from_array(X_super_sequence, Y_super_sequence.flatten())
            
            # Clean up super sequences immediately after use
            del X_super_sequence, Y_super_sequence
            gc.collect()
        else: # Concatenate lists of sequence arrays
            if not all_X_sequences_list: # No sequences created from any file
                num_features = len(self.feature_cols) if self.feature_cols else 0
                return np.empty((0, self.lookback, num_features)), np.empty((0,))

            if self.logger:
                total_sequences = sum(arr.shape[0] for arr in all_X_sequences_list)
                self.logger.info(f"Concatenating {len(all_X_sequences_list)} sequence arrays with {total_sequences} total sequences")

            # Memory optimization: Pre-calculate total size for concatenation
            total_sequences = sum(arr.shape[0] for arr in all_X_sequences_list)
            lookback_size = all_X_sequences_list[0].shape[1]
            n_features = all_X_sequences_list[0].shape[2]
            
            # Pre-allocate final arrays
            X_processed = np.empty((total_sequences, lookback_size, n_features), dtype=np.float32)
            y_processed = np.empty((total_sequences,), dtype=np.float32)
            
            # Fill pre-allocated arrays
            current_seq = 0
            for X_seq_arr, y_seq_arr in zip(all_X_sequences_list, all_y_sequences_list):
                end_seq = current_seq + X_seq_arr.shape[0]
                X_processed[current_seq:end_seq] = X_seq_arr
                y_processed[current_seq:end_seq] = y_seq_arr
                current_seq = end_seq
            
            # Clear sequence lists immediately
            del all_X_sequences_list, all_y_sequences_list
        
        gc.collect()
        
        # Final memory usage log
        if self.logger and initial_memory:
            try:
                final_memory = psutil.Process().memory_info().rss / 1024 / 1024
                self.logger.info(f"Final memory usage: {final_memory:.1f} MB (+{final_memory-initial_memory:.1f} MB)")
                self.logger.info(f"SequenceRNNDataHandler: Processed X shape: {X_processed.shape}, y shape: {y_processed.shape}")
            except:
                if self.logger:
                    self.logger.info(f"SequenceRNNDataHandler: Processed X shape: {X_processed.shape}, y shape: {y_processed.shape}")
        elif self.logger:
            self.logger.info(f"SequenceRNNDataHandler: Processed X shape: {X_processed.shape}, y shape: {y_processed.shape}")
        
        return X_processed, y_processed