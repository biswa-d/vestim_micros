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
                              train_split: float = 0.7, seed: int = None):
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
        :return: A tuple of (train_loader, val_loader) PyTorch DataLoader objects.
        """
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
        
        # Clean up numpy arrays after conversion to tensors
        del X, y
        gc.collect()

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
        # MEMORY FIX: Set num_workers=0 to prevent memory duplication for large datasets
        # Large in-memory datasets are faster with single-threaded loading (no IPC overhead)
        optimized_num_workers = 0  # Always use 0 for memory efficiency
        
        if num_workers > 0:
            self.logger.info(f"Setting num_workers=0 (was {num_workers}) for memory optimization with large datasets")
        
        train_loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            sampler=train_sampler, 
            drop_last=True, 
            num_workers=optimized_num_workers,
            pin_memory=False,  # Disable pin_memory to reduce GPU memory duplication
            prefetch_factor=1,   # Reduce prefetching to minimize memory usage
            persistent_workers=False  # Don't keep workers alive between epochs
        )
        val_loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            sampler=valid_sampler, 
            drop_last=True, 
            num_workers=optimized_num_workers,
            pin_memory=False,
            prefetch_factor=1,
            persistent_workers=False
        )

        # Clean up cache variables after DataLoaders are created
        del X_tensor, y_tensor, indices, train_indices, valid_indices
        gc.collect()

        return train_loader, val_loader
