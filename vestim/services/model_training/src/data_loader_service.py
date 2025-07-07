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
                              train_split: float = 0.7, seed: int = None, model_type: str = "LSTM"):
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
        :return: A tuple of (train_loader, val_loader) PyTorch DataLoader objects.
        """
        # Special handling for FNN models with batch training
        if model_type == "FNN":
            return self.create_fnn_batch_data_loaders(
                folder_path, feature_cols, target_col, batch_size, num_workers, train_split, seed
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

        X, y = handler.load_and_process_data(folder_path, **handler_kwargs)

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
        if getattr(sys, 'frozen', False):
            # Running as PyInstaller executable
            optimized_num_workers = 0
            self.logger.info("Running as PyInstaller executable - disabled multiprocessing to prevent worker crashes")
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
        if getattr(sys, 'frozen', False):
            optimized_num_workers = 0
            self.logger.info("Running as PyInstaller executable - disabled multiprocessing for FNN DataLoader")
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
