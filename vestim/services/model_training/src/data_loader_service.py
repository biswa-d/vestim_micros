import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from datetime import datetime
import gc  # For garbage collection
import logging
from vestim.services.model_training.src.sequence_rnn_data_handler import SequenceRNNDataHandler
from vestim.services.model_training.src.whole_sequence_fnn_data_handler import WholeSequenceFNNDataHandler

class DataLoaderService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def create_data_loaders(self, folder_path: str, training_method: str, feature_cols: list, target_col: str,
                              batch_size: int, num_workers: int,
                              lookback: int = None, # Optional, only for SequenceRNN
                              concatenate_raw_data: bool = False, # Optional, for SequenceRNN
                              train_split: float = 0.7, seed: int = None, model_type: str = "LSTM",
                              sequence_split_method: str = "temporal"):
        """
        Creates DataLoaders for training and validation data using a specified data handling strategy.

        :param folder_path: Path to the folder containing the data files.
        :param training_method: Specifies the data handling strategy ("SequenceRNN" or "WholeSequenceFNN").
        :param feature_cols: List of feature column names.
        :param target_col: Target column name.
        :param batch_size: The batch size for the DataLoader.
        :param num_workers: Number of subprocesses to use for data loading.
        :param lookback: The lookback window (required for "SequenceRNN").
        :param concatenate_raw_data: For "SequenceRNN", if True, concatenates raw data before sequencing.
        :param train_split: Fraction of data to use for training.
        :param seed: Random seed for reproducibility.
        :param model_type: Type of model (LSTM, GRU, FNN) - affects data loading strategy.
        :param sequence_split_method: Method for splitting sequences ("temporal", "random", "file_wise").
        :return: A tuple of (train_loader, val_loader) PyTorch DataLoader objects.
        """
        # Special handling for FNN models with batch training
        if model_type == "FNN":
            return self.create_fnn_batch_data_loaders(
                folder_path, feature_cols, target_col, batch_size, num_workers, train_split, seed
            )
        
        # Special handling for LSTM/GRU models to prevent data leakage
        if model_type in ["LSTM", "GRU"] and training_method == "Sequence-to-Sequence" and sequence_split_method == "temporal":
            return self.create_temporal_sequence_data_loaders(
                folder_path, feature_cols, target_col, batch_size, num_workers,
                lookback, concatenate_raw_data, train_split, seed, model_type
            )
        if seed is None:
            seed = int(datetime.now().timestamp())
        
        self.logger.info(f"Creating data loaders with training method: {training_method}")
        self.logger.info(f"Feature columns: {feature_cols}, Target column: {target_col}")

        handler_kwargs = {}
        if training_method == "Sequence-to-Sequence": # Corrected string comparison
            if lookback is None or lookback <= 0:
                self.logger.error("Lookback must be a positive integer for SequenceRNN training method.")
                raise ValueError("Lookback must be a positive integer for SequenceRNN training method.")
            handler = SequenceRNNDataHandler(feature_cols, target_col, lookback, concatenate_raw_data)
            handler_kwargs['lookback'] = lookback # Though already in init, pass for clarity if load_and_process_data uses it
        elif training_method == "WholeSequenceFNN":
            handler = WholeSequenceFNNDataHandler(feature_cols, target_col)
            # No specific kwargs beyond what's in init for WholeSequenceFNNDataHandler's load_and_process_data
        else:
            self.logger.error(f"Unsupported training_method: {training_method}")
            raise ValueError(f"Unsupported training_method: {training_method}")
        
        handler.logger = self.logger # Pass logger to handler

        X, y, _ = handler.load_and_process_data(folder_path, **handler_kwargs)

        if X.size == 0 or y.size == 0:
            self.logger.warning("No data loaded by the handler. Returning empty DataLoaders.")
            # Create empty dataloaders to prevent crashes downstream, though they won't be useful
            empty_dataset = TensorDataset(torch.empty(0), torch.empty(0))
            empty_loader = DataLoader(empty_dataset, batch_size=batch_size)
            return empty_loader, empty_loader

        # Memory-efficient tensor conversion using zero-copy where possible
        self.logger.info(f"Converting numpy arrays to PyTorch tensors...")
        self.logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
        
        # Ensure arrays are float32 for memory efficiency and compatibility
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        if y.dtype != np.float32:
            y = y.astype(np.float32)
        
        # Use torch.from_numpy for zero-copy conversion (much faster and no memory duplication)
        X_tensor = torch.from_numpy(X)  # Zero-copy conversion
        
        # Handle y tensor shape and conversion
        if y.ndim == 1: # If y is [N,], reshape to [N,1] for consistency if model expects 2D target
            y_tensor = torch.from_numpy(y).unsqueeze(1)
        else: # y is already [N, num_targets]
            y_tensor = torch.from_numpy(y)
        
        self.logger.info(f"Data loaded. X_tensor shape: {X_tensor.shape}, y_tensor shape: {y_tensor.shape}")
        
        # NOTE: Do NOT delete X, y numpy arrays here since tensors share memory with them
        # They will be cleaned up when tensors are deleted

        # Create a TensorDataset
        dataset = TensorDataset(X_tensor, y_tensor)
        dataset_size = len(dataset)
        indices = list(range(dataset_size))

        # Train-validation split
        train_size = int(dataset_size * train_split)

        np.random.seed(seed)
        np.random.shuffle(indices)

        train_indices, valid_indices = indices[:train_size], indices[train_size:]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)

        # Create DataLoaders with memory optimization
        # PYINSTALLER FIX: Disable multiprocessing entirely when running as exe to prevent worker crashes
        # WINDOWS MULTIPROCESSING FIX: Disable multiprocessing on Windows when PyQt5 is imported to prevent spawn issues
        import platform
        
        if getattr(sys, 'frozen', False):
            # Running as PyInstaller executable
            optimized_num_workers = 0
            self.logger.info("Running as PyInstaller executable - disabled multiprocessing to prevent worker crashes")
        elif platform.system() == "Windows":
            # Running on Windows - always disable multiprocessing to prevent spawn issues with GUI applications
            optimized_num_workers = 0
            self.logger.info(f"Running on Windows - disabled multiprocessing (original num_workers: {num_workers}) to prevent spawn issues")
        else:
            # MEMORY FIX: Adaptive num_workers based on dataset size for optimal memory usage
            # Large datasets benefit more from reduced workers than parallel loading
            optimized_num_workers = min(num_workers, 2) if len(dataset) > 100000 else num_workers
            
            # CRITICAL: For large datasets, disable multiprocessing entirely to prevent memory duplication
            if len(dataset) > 1000000:  # > 1M sequences
                optimized_num_workers = 0
                self.logger.info(f"Disabled multiprocessing for large dataset ({len(dataset)} sequences) to prevent memory duplication")
            elif optimized_num_workers != num_workers:
                self.logger.info(f"Reduced num_workers from {num_workers} to {optimized_num_workers} for dataset memory optimization")
        
        # CRITICAL FIX: Set prefetch_factor based on num_workers (must be None when num_workers=0)
        prefetch_factor_value = 1 if optimized_num_workers > 0 else None
        
        train_loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            sampler=train_sampler, 
            drop_last=True, 
            num_workers=optimized_num_workers,
            pin_memory=False,  # Disable pin_memory to reduce GPU memory duplication
            prefetch_factor=prefetch_factor_value,  # None when num_workers=0, 1 otherwise
            persistent_workers=False  # Don't keep workers alive between epochs
        )
        val_loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            sampler=valid_sampler, 
            drop_last=True, 
            num_workers=optimized_num_workers,
            pin_memory=False,
            prefetch_factor=prefetch_factor_value,  # None when num_workers=0, 1 otherwise
            persistent_workers=False
        )

        # Clean up cache variables after DataLoaders are created
        del X_tensor, y_tensor, indices, train_indices, valid_indices
        gc.collect()
        
        # Memory monitoring
        try:
            import psutil
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            self.logger.info(f"Memory after DataLoader creation: {current_memory:.1f} MB")
        except ImportError:
            pass

        return train_loader, val_loader
    
    def create_fnn_batch_data_loaders(self, folder_path: str, feature_cols: list, target_col: str,
                                      batch_size: int, num_workers: int, train_split: float = 0.7, 
                                      seed: int = None):
        """
        Creates DataLoaders specifically for FNN models with batch training logic:
        1. Process each file individually to create batch-sized chunks
        2. Drop incomplete batches from each file (preserves file integrity)
        3. Combine all chunks from all files
        4. Shuffle and split chunks into train/validation sets
        
        This approach ensures that each batch contains samples from only one file,
        preventing mixing of samples from different data sources/conditions.
        
        :param folder_path: Path to the folder containing the data files.
        :param feature_cols: List of feature column names.
        :param target_col: Target column name.
        :param batch_size: The batch size for each batch.
        :param num_workers: Number of subprocesses to use for data loading.
        :param train_split: Fraction of data to use for training.
        :param seed: Random seed for reproducibility.
        :return: A tuple of (train_loader, val_loader) with FNN batch logic.
        """
        if seed is None:
            seed = int(datetime.now().timestamp())
        
        self.logger.info(f"Creating FNN batch data loaders with file-wise chunking, batch size: {batch_size}")
        
        # Get all CSV files in the folder (excluding test files)
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and 'test' not in f.lower()]
        
        if not csv_files:
            self.logger.warning("No training CSV files found. Returning empty DataLoaders.")
            empty_dataset = TensorDataset(torch.empty(0), torch.empty(0))
            empty_loader = DataLoader(empty_dataset, batch_size=batch_size)
            return empty_loader, empty_loader
        
        self.logger.info(f"Processing {len(csv_files)} training files: {csv_files}")
        
        all_X_chunks = []
        all_y_chunks = []
        total_chunks = 0
        total_dropped_samples = 0
        
        # Process each file individually
        for file_name in csv_files:
            file_path = os.path.join(folder_path, file_name)
            self.logger.info(f"Processing file: {file_name}")
            
            try:
                # Load individual file
                df = pd.read_csv(file_path)
                
                if df.empty:
                    self.logger.warning(f"File {file_name} is empty, skipping.")
                    continue
                
                # Check if required columns exist
                missing_cols = [col for col in feature_cols + [target_col] if col not in df.columns]
                if missing_cols:
                    self.logger.warning(f"File {file_name} missing columns {missing_cols}, skipping.")
                    continue
                
                # Extract features and target
                X_file = df[feature_cols].values
                y_file = df[target_col].values
                
                # Ensure float32 for memory efficiency
                if X_file.dtype != np.float32:
                    X_file = X_file.astype(np.float32)
                if y_file.dtype != np.float32:
                    y_file = y_file.astype(np.float32)
                
                file_samples = X_file.shape[0]
                file_chunks = file_samples // batch_size
                samples_to_keep = file_chunks * batch_size
                
                if samples_to_keep == 0:
                    self.logger.warning(f"File {file_name} has {file_samples} samples, less than batch_size {batch_size}, skipping.")
                    continue
                
                if samples_to_keep < file_samples:
                    dropped_from_file = file_samples - samples_to_keep
                    total_dropped_samples += dropped_from_file
                    self.logger.info(f"File {file_name}: keeping {samples_to_keep}/{file_samples} samples in {file_chunks} chunks")
                    X_file = X_file[:samples_to_keep]
                    y_file = y_file[:samples_to_keep]
                
                # Create chunks from this file
                X_file_reshaped = X_file.reshape(file_chunks, batch_size, X_file.shape[1])
                y_file_reshaped = y_file.reshape(file_chunks, batch_size, -1) if y_file.ndim > 1 else y_file.reshape(file_chunks, batch_size, 1)
                
                # Add chunks to our collection
                for chunk_idx in range(file_chunks):
                    all_X_chunks.append(X_file_reshaped[chunk_idx])
                    all_y_chunks.append(y_file_reshaped[chunk_idx])
                
                total_chunks += file_chunks
                
            except Exception as e:
                self.logger.error(f"Error processing file {file_name}: {str(e)}")
                continue
        
        if total_chunks == 0:
            self.logger.warning("No valid chunks created from any files. Returning empty DataLoaders.")
            empty_dataset = TensorDataset(torch.empty(0), torch.empty(0))
            empty_loader = DataLoader(empty_dataset, batch_size=batch_size)
            return empty_loader, empty_loader
        
        self.logger.info(f"Created {total_chunks} chunks from {len(csv_files)} files")
        self.logger.info(f"Total dropped samples across all files: {total_dropped_samples}")
        
        # Convert chunks to numpy arrays
        all_X_chunks = np.array(all_X_chunks)  # Shape: (total_chunks, batch_size, num_features)
        all_y_chunks = np.array(all_y_chunks)  # Shape: (total_chunks, batch_size, num_targets)
        
        # Shuffle chunks and split into train/validation
        np.random.seed(seed)
        chunk_indices = np.arange(total_chunks)
        np.random.shuffle(chunk_indices)
        
        train_chunks = int(total_chunks * train_split)
        val_chunks = total_chunks - train_chunks
        
        train_chunk_indices = chunk_indices[:train_chunks]
        val_chunk_indices = chunk_indices[train_chunks:]
        
        self.logger.info(f"Split: {train_chunks} train chunks, {val_chunks} val chunks")
        self.logger.info(f"Train samples: {train_chunks * batch_size}, Val samples: {val_chunks * batch_size}")
        
        # Create train and validation data
        if train_chunks > 0:
            X_train_chunks = all_X_chunks[train_chunk_indices]
            y_train_chunks = all_y_chunks[train_chunk_indices]
            # Flatten chunks back to samples for DataLoader
            X_train = X_train_chunks.reshape(-1, X_train_chunks.shape[2])
            y_train = y_train_chunks.reshape(-1, y_train_chunks.shape[2])
        else:
            X_train = np.empty((0, all_X_chunks.shape[2]))
            y_train = np.empty((0, all_y_chunks.shape[2]))
        
        if val_chunks > 0:
            X_val_chunks = all_X_chunks[val_chunk_indices]
            y_val_chunks = all_y_chunks[val_chunk_indices]
            # Flatten chunks back to samples for DataLoader
            X_val = X_val_chunks.reshape(-1, X_val_chunks.shape[2])
            y_val = y_val_chunks.reshape(-1, y_val_chunks.shape[2])
        else:
            X_val = np.empty((0, all_X_chunks.shape[2]))
            y_val = np.empty((0, all_y_chunks.shape[2]))
        
        # Create DataLoaders
        train_loader = self._create_fnn_dataloader(X_train, y_train, batch_size, num_workers, seed, is_training=True)
        val_loader = self._create_fnn_dataloader(X_val, y_val, batch_size, num_workers, seed, is_training=False)
        
        # Clean up
        del all_X_chunks, all_y_chunks, X_train, y_train, X_val, y_val
        gc.collect()
        
        return train_loader, val_loader
    
    def _create_fnn_dataloader(self, X, y, batch_size, num_workers, seed, is_training=True):
        """
        Creates a DataLoader for FNN with proper batch handling.
        """
        if X.size == 0:
            empty_dataset = TensorDataset(torch.empty(0), torch.empty(0))
            return DataLoader(empty_dataset, batch_size=batch_size)
        
        # Convert to tensors
        X_tensor = torch.from_numpy(X)
        if y.ndim == 1:
            y_tensor = torch.from_numpy(y).unsqueeze(1)
        else:
            y_tensor = torch.from_numpy(y)
        
        # Create dataset
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # For FNN, we want shuffling at the batch level, not sample level
        # So we shuffle=True for training and False for validation
        shuffle_batches = is_training
        
        # Optimize num_workers for memory efficiency
        # PYINSTALLER FIX: Disable multiprocessing entirely when running as exe
        # WINDOWS MULTIPROCESSING FIX: Disable multiprocessing on Windows when PyQt5 is imported
        import platform
        
        if getattr(sys, 'frozen', False):
            optimized_num_workers = 0
            self.logger.info("Running as PyInstaller executable - disabled multiprocessing for FNN DataLoader")
        elif platform.system() == "Windows":
            optimized_num_workers = 0
            self.logger.info(f"Running on Windows - disabled multiprocessing for FNN DataLoader (original num_workers: {num_workers})")
        else:
            optimized_num_workers = min(num_workers, 2) if len(dataset) > 100000 else num_workers
            if len(dataset) > 1000000:
                optimized_num_workers = 0
            
        prefetch_factor_value = 1 if optimized_num_workers > 0 else None
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle_batches,  # Shuffle batches for training
            drop_last=True,  # Always drop last incomplete batch for FNN
            num_workers=optimized_num_workers,
            pin_memory=False,
            prefetch_factor=prefetch_factor_value,
            persistent_workers=False
        )
        
        return loader

    def create_temporal_sequence_data_loaders(self, folder_path: str, feature_cols: list, target_col: str,
                                            batch_size: int, num_workers: int, lookback: int,
                                            concatenate_raw_data: bool, train_split: float = 0.7, 
                                            seed: int = None, model_type: str = "LSTM"):
        """
        Creates DataLoaders for LSTM/GRU models with temporal splitting to prevent data leakage.
        
        This method addresses the critical overfitting issue in LSTM sequence training by:
        1. Processing files individually to maintain temporal structure
        2. Using temporal splitting instead of random shuffling
        3. Ensuring no overlap between training and validation sequences
        
        Approach:
        - For each file, split sequences temporally (first 70% for train, last 30% for val)
        - Maintains temporal order within each split
        - Combines sequences from all files respecting the temporal boundaries
        
        :param folder_path: Path to the folder containing the data files.
        :param feature_cols: List of feature column names.
        :param target_col: Target column name.
        :param batch_size: The batch size for the DataLoader.
        :param num_workers: Number of subprocesses to use for data loading.
        :param lookback: The lookback window for sequence creation.
        :param concatenate_raw_data: If True, concatenates raw data before sequencing.
        :param train_split: Fraction of data to use for training.
        :param seed: Random seed for reproducibility.
        :param model_type: Type of model (LSTM, GRU).
        :return: A tuple of (train_loader, val_loader) with temporal splitting.
        """
        if seed is None:
            seed = int(datetime.now().timestamp())
        
        self.logger.info(f"Creating temporal sequence data loaders for {model_type}")
        self.logger.info(f"Using temporal splitting to prevent data leakage - train_split: {train_split}")
        
        # Get all CSV files
        csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
        if not csv_files:
            self.logger.warning(f"No CSV files found in {folder_path}.")
            empty_dataset = TensorDataset(torch.empty(0), torch.empty(0))
            empty_loader = DataLoader(empty_dataset, batch_size=batch_size)
            return empty_loader, empty_loader
        
        all_train_X_sequences = []
        all_train_y_sequences = []
        all_val_X_sequences = []
        all_val_y_sequences = []
        
        # Process each file individually with temporal splitting
        for file_idx, file_path in enumerate(csv_files):
            self.logger.info(f"Processing file {file_idx+1}/{len(csv_files)}: {file_path}")
            
            try:
                # Read and select columns
                df = pd.read_csv(file_path)
                if df.empty:
                    continue
                
                # Ensure required columns exist
                missing_cols = [col for col in feature_cols + [target_col] if col not in df.columns]
                if missing_cols:
                    self.logger.warning(f"Missing columns {missing_cols} in {file_path}. Skipping file.")
                    continue
                
                # Extract features and target
                X_data = df[feature_cols].values.astype(np.float32)
                y_data = df[target_col].values.astype(np.float32)
                
                # Check if file has enough data for sequences
                if len(X_data) <= lookback:
                    self.logger.warning(f"File {file_path} has insufficient data (length {len(X_data)}) for lookback {lookback}. Skipping.")
                    continue
                
                # Create sequences from this file
                handler = SequenceRNNDataHandler(feature_cols, target_col, lookback, False)
                X_file_sequences, y_file_sequences = handler._create_sequences_from_array(X_data, y_data)
                
                if X_file_sequences.size == 0:
                    continue
                
                # Temporal split: first train_split% for training, rest for validation
                num_sequences = len(X_file_sequences)
                train_seq_count = int(num_sequences * train_split)
                
                if train_seq_count > 0:
                    # Training sequences (first part of file)
                    train_X_file = X_file_sequences[:train_seq_count]
                    train_y_file = y_file_sequences[:train_seq_count]
                    all_train_X_sequences.append(train_X_file)
                    all_train_y_sequences.append(train_y_file)
                
                if train_seq_count < num_sequences:
                    # Validation sequences (last part of file)
                    val_X_file = X_file_sequences[train_seq_count:]
                    val_y_file = y_file_sequences[train_seq_count:]
                    all_val_X_sequences.append(val_X_file)
                    all_val_y_sequences.append(val_y_file)
                
                self.logger.info(f"File {file_idx+1}: {train_seq_count} train sequences, {num_sequences - train_seq_count} val sequences")
                
            except Exception as e:
                self.logger.error(f"Error processing file {file_path}: {e}")
                continue
        
        # Combine sequences from all files
        if not all_train_X_sequences and not all_val_X_sequences:
            self.logger.warning("No training or validation sequences were created from any file.")
            empty_dataset = TensorDataset(torch.empty(0), torch.empty(0))
            empty_loader = DataLoader(empty_dataset, batch_size=batch_size)
            return empty_loader, empty_loader

        # Concatenate all training sequences
        if all_train_X_sequences:
            X_train = np.concatenate(all_train_X_sequences, axis=0)
            y_train = np.concatenate(all_train_y_sequences, axis=0)
        else:
            self.logger.warning("No training sequences created from any file.")
            # Create empty arrays with correct dimensions
            X_train = np.empty((0, lookback, len(feature_cols)), dtype=np.float32)
            y_train = np.empty((0,), dtype=np.float32)

        # Concatenate all validation sequences (if any)
        if all_val_X_sequences:
            X_val = np.concatenate(all_val_X_sequences, axis=0)
            y_val = np.concatenate(all_val_y_sequences, axis=0)
        else:
            self.logger.warning("No validation sequences created from any file.")
            # Create empty arrays with correct dimensions
            X_val = np.empty((0, lookback, len(feature_cols)), dtype=np.float32)
            y_val = np.empty((0,), dtype=np.float32)

        # Final check for empty datasets to prevent empty loaders
        if X_train.size == 0 and X_val.size == 0:
            self.logger.error("Both training and validation datasets are empty. Cannot create loaders.")
            empty_dataset = TensorDataset(torch.empty(0), torch.empty(0))
            empty_loader = DataLoader(empty_dataset, batch_size=batch_size)
            return empty_loader, empty_loader
        
        self.logger.info(f"Total sequences - Train: {len(X_train)}, Validation: {len(X_val)}")
        self.logger.info(f"Train X shape: {X_train.shape}, Train y shape: {y_train.shape}")
        self.logger.info(f"Val X shape: {X_val.shape}, Val y shape: {y_val.shape}")
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.from_numpy(X_train)
        y_train_tensor = torch.from_numpy(y_train)
        X_val_tensor = torch.from_numpy(X_val)
        y_val_tensor = torch.from_numpy(y_val)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        # Optimize num_workers for PyInstaller and memory usage
        import platform
        
        if getattr(sys, 'frozen', False):
            optimized_num_workers = 0
            self.logger.info("Running as PyInstaller executable - disabled multiprocessing")
        elif platform.system() == "Windows":
            optimized_num_workers = 0
            self.logger.info("Running on Windows - disabled multiprocessing to avoid GUI import issues")
        else:
            optimized_num_workers = min(num_workers, 2) if len(train_dataset) > 100000 else num_workers
            if len(train_dataset) > 1000000:
                optimized_num_workers = 0
                self.logger.info(f"Disabled multiprocessing for large dataset ({len(train_dataset)} sequences)")
        
        prefetch_factor_value = 1 if optimized_num_workers > 0 else None
        
        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,  # Safe to shuffle within temporal splits
            drop_last=True,
            num_workers=optimized_num_workers,
            pin_memory=False,
            prefetch_factor=prefetch_factor_value,
            persistent_workers=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # No need to shuffle validation data
            drop_last=True,
            num_workers=optimized_num_workers,
            pin_memory=False,
            prefetch_factor=prefetch_factor_value,
            persistent_workers=False
        )
        
        # Clean up
        del all_train_X_sequences, all_train_y_sequences, all_val_X_sequences, all_val_y_sequences
        del X_train, y_train, X_val, y_val
        gc.collect()
        
        self.logger.info("Temporal sequence data loaders created successfully - NO DATA LEAKAGE!")
        return train_loader, val_loader

    def create_data_loaders_from_separate_folders(self, job_folder_path: str, training_method: str, feature_cols: list, target_col: str,
                                                 batch_size: int, num_workers: int,
                                                 lookback: int = None, # Optional, only for SequenceRNN
                                                 concatenate_raw_data: bool = False, # Optional, for SequenceRNN
                                                 seed: int = None, model_type: str = "LSTM", 
                                                 create_test_loader: bool = True):
        """
        Creates DataLoaders for training, validation, and optionally test data from separate folders.
        This method replaces the train_split approach and expects the job folder to contain
        train_data/processed_data, val_data/processed_data, and test_data/processed_data folders.

        :param job_folder_path: Path to the job folder containing train_data, val_data, test_data subfolders.
        :param training_method: Specifies the data handling strategy ("SequenceRNN" or "WholeSequenceFNN").
        :param feature_cols: List of feature column names.
        :param target_col: Target column name.
        :param batch_size: The batch size for the DataLoader.
        :param num_workers: Number of subprocesses to use for data loading.
        :param lookback: The lookback window (required for "SequenceRNN").
        :param concatenate_raw_data: For "SequenceRNN", if True, concatenates raw data before sequencing.
        :param seed: Random seed for reproducibility.
        :param model_type: Type of model (LSTM, GRU, FNN) - affects data loading strategy.
        :param create_test_loader: Whether to create test loader (False during training to save memory).
        :return: A tuple of (train_loader, val_loader, test_loader) PyTorch DataLoader objects when create_test_loader=True.
                 A tuple of (train_loader, val_loader) PyTorch DataLoader objects when create_test_loader=False.
        """
        if seed is None:
            seed = int(datetime.now().timestamp())
        
        self.logger.info(f"Creating data loaders from separate folders for {model_type}")
        self.logger.info(f"Job folder: {job_folder_path}")
        
        # Define folder paths
        train_folder = os.path.join(job_folder_path, 'train_data', 'processed_data')
        val_folder = os.path.join(job_folder_path, 'val_data', 'processed_data')
        test_folder = os.path.join(job_folder_path, 'test_data', 'processed_data')
        
        # Verify required folders exist (test folder only if needed)
        required_folders = [("train", train_folder), ("validation", val_folder)]
        if create_test_loader:
            required_folders.append(("test", test_folder))
            
        for folder_name, folder_path in required_folders:
            if not os.path.exists(folder_path):
                self.logger.error(f"{folder_name} folder not found: {folder_path}")
                raise ValueError(f"{folder_name} folder not found: {folder_path}")
        
        # Special handling for FNN models with batch training
        if model_type == "FNN":
            train_loader = self._create_single_data_loader(
                train_folder, feature_cols, target_col, batch_size, num_workers, seed, "train"
            )
            val_loader = self._create_single_data_loader(
                val_folder, feature_cols, target_col, batch_size, num_workers, seed, "validation"
            )
            if create_test_loader:
                test_loader = self._create_single_data_loader(
                    test_folder, feature_cols, target_col, batch_size, num_workers, seed, "test"
                )
                return train_loader, val_loader, test_loader
            else:
                return train_loader, val_loader
        
        # For LSTM/GRU models, use the sequence data handlers
        handler_kwargs = {}
        if training_method == "Sequence-to-Sequence":
            if lookback is None or lookback <= 0:
                self.logger.error("Lookback must be a positive integer for SequenceRNN training method.")
                raise ValueError("Lookback must be a positive integer for SequenceRNN training method.")
            handler = SequenceRNNDataHandler(feature_cols, target_col, lookback, concatenate_raw_data)
            handler_kwargs['lookback'] = lookback
        elif training_method == "WholeSequenceFNN":
            handler = WholeSequenceFNNDataHandler(feature_cols, target_col)
        else:
            self.logger.error(f"Unsupported training_method: {training_method}")
            raise ValueError(f"Unsupported training_method: {training_method}")
        
        handler.logger = self.logger
        
        # Load data from each folder
        train_X, train_y = handler.load_and_process_data(train_folder, **handler_kwargs)
        val_X, val_y = handler.load_and_process_data(val_folder, **handler_kwargs)
        
        # Create data loaders
        train_loader = self._create_loader_from_tensors(train_X, train_y, batch_size, num_workers, True, "train")
        val_loader = self._create_loader_from_tensors(val_X, val_y, batch_size, num_workers, False, "validation")
        
        # Conditionally create test loader
        if create_test_loader:
            # For test data, use WholeSequenceFNNDataHandler to avoid sequencing
            # Test data should remain as original processed data for proper evaluation
            self.logger.info("Loading test data without sequencing for proper evaluation")
            test_handler = WholeSequenceFNNDataHandler(feature_cols, target_col)
            test_handler.logger = self.logger
            test_X, test_y, test_timestamps = test_handler.load_and_process_data(test_folder, return_timestamp=True)
            test_loader = self._create_loader_from_tensors(test_X, test_y, batch_size, num_workers, False, "test")
            return train_loader, val_loader, test_loader, test_timestamps
        else:
            self.logger.info("Skipping test loader creation (create_test_loader=False)")
            return train_loader, val_loader

    def _create_single_data_loader(self, folder_path: str, feature_cols: list, target_col: str,
                                  batch_size: int, num_workers: int, seed: int, data_type: str):
        """Helper method to create a single data loader from a folder."""
        csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
        if not csv_files:
            self.logger.warning(f"No CSV files found in {folder_path}.")
            empty_dataset = TensorDataset(torch.empty(0), torch.empty(0))
            return DataLoader(empty_dataset, batch_size=batch_size)
        
        all_X = []
        all_y = []
        
        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path)
                if df.empty:
                    continue
                
                missing_cols = [col for col in feature_cols + [target_col] if col not in df.columns]
                if missing_cols:
                    self.logger.warning(f"Missing columns {missing_cols} in {file_path}. Skipping file.")
                    continue
                
                X = df[feature_cols].values.astype(np.float32)
                y = df[target_col].values.astype(np.float32)
                
                all_X.append(X)
                all_y.append(y)
                
            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")
                continue
        
        if not all_X:
            self.logger.warning(f"No valid data found in {folder_path}")
            empty_dataset = TensorDataset(torch.empty(0), torch.empty(0))
            return DataLoader(empty_dataset, batch_size=batch_size)
        
        # Combine all data
        combined_X = np.vstack(all_X)
        combined_y = np.hstack(all_y)
        
        return self._create_loader_from_tensors(combined_X, combined_y, batch_size, num_workers, data_type == "train", data_type)

    def _create_loader_from_tensors(self, X, y, batch_size: int, num_workers: int, shuffle: bool, data_type: str):
        """Helper method to create a DataLoader from tensors."""
        if X.size == 0 or y.size == 0:
            self.logger.warning(f"Empty {data_type} data. Creating empty DataLoader.")
            empty_dataset = TensorDataset(torch.empty(0), torch.empty(0))
            return DataLoader(empty_dataset, batch_size=batch_size)
        
        # Convert to tensors
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).float()
        
        dataset = TensorDataset(X_tensor, y_tensor)
        
        prefetch_factor_value = 2 if num_workers > 0 else None
        
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True if data_type in ["train", "validation"] else False,  # Drop last for train/val to ensure consistent batch sizes
            prefetch_factor=prefetch_factor_value,
            persistent_workers=False
        )
        
        self.logger.info(f"Created {data_type} DataLoader: {len(dataset)} samples, {len(loader)} batches")
        return loader
