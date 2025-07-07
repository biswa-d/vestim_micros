import os
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
        1. Concatenate all training files
        2. Split into non-overlapping batches of fixed size
        3. Drop incomplete batches
        4. Support epoch-wise batch shuffling
        
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
        
        self.logger.info(f"Creating FNN batch data loaders with batch size: {batch_size}")
        
        # Use WholeSequenceFNNDataHandler to concatenate all files
        handler = WholeSequenceFNNDataHandler(feature_cols, target_col)
        handler.logger = self.logger
        
        X, y = handler.load_and_process_data(folder_path)
        
        if X.size == 0 or y.size == 0:
            self.logger.warning("No data loaded by the handler. Returning empty DataLoaders.")
            empty_dataset = TensorDataset(torch.empty(0), torch.empty(0))
            empty_loader = DataLoader(empty_dataset, batch_size=batch_size)
            return empty_loader, empty_loader
        
        # Ensure arrays are float32 for memory efficiency
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        if y.dtype != np.float32:
            y = y.astype(np.float32)
        
        total_samples = X.shape[0]
        self.logger.info(f"Total samples before batching: {total_samples}")
        
        # Calculate number of complete batches and drop incomplete samples
        num_complete_batches = total_samples // batch_size
        samples_to_keep = num_complete_batches * batch_size
        
        if samples_to_keep < total_samples:
            dropped_samples = total_samples - samples_to_keep
            self.logger.info(f"Dropping {dropped_samples} incomplete samples. Keeping {samples_to_keep} samples in {num_complete_batches} complete batches.")
            X = X[:samples_to_keep]
            y = y[:samples_to_keep]
        
        # Split into train and validation before creating batches
        train_size = int(samples_to_keep * train_split)
        # Ensure train_size is divisible by batch_size
        train_batches = train_size // batch_size
        train_size = train_batches * batch_size
        
        val_size = samples_to_keep - train_size
        val_batches = val_size // batch_size
        val_size = val_batches * batch_size
        
        # Adjust if validation set doesn't have complete batches
        if val_size == 0 and samples_to_keep > train_size:
            # If we can't make a complete validation batch, take one batch from training
            if train_batches > 1:
                train_batches -= 1
                train_size = train_batches * batch_size
                val_size = batch_size
                val_batches = 1
            else:
                # Not enough data for proper train/val split with complete batches
                self.logger.warning("Not enough data for proper train/validation split with complete batches.")
                val_size = 0
                val_batches = 0
        
        self.logger.info(f"Train samples: {train_size} ({train_batches} batches), Val samples: {val_size} ({val_batches} batches)")
        
        # Split the data
        X_train = X[:train_size] if train_size > 0 else np.empty((0, X.shape[1]))
        y_train = y[:train_size] if train_size > 0 else np.empty((0, y.shape[1] if y.ndim > 1 else 1))
        X_val = X[train_size:train_size + val_size] if val_size > 0 else np.empty((0, X.shape[1]))
        y_val = y[train_size:train_size + val_size] if val_size > 0 else np.empty((0, y.shape[1] if y.ndim > 1 else 1))
        
        # Create DataLoaders with FNN batch logic
        train_loader = self._create_fnn_dataloader(X_train, y_train, batch_size, num_workers, seed, is_training=True)
        val_loader = self._create_fnn_dataloader(X_val, y_val, batch_size, num_workers, seed, is_training=False)
        
        # Clean up
        del X, y, X_train, y_train, X_val, y_val
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
