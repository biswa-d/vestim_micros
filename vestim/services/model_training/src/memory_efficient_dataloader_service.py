import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset
from datetime import datetime
import gc
import psutil
import os

from vestim.services.model_training.src.sequence_rnn_data_handler import SequenceRNNDataHandler
from vestim.services.model_training.src.whole_sequence_fnn_data_handler import WholeSequenceFNNDataHandler
from vestim.services.model_training.src.memory_efficient_sequence_dataset import (
    MemoryEfficientSequenceDataset, 
    HybridSequenceDataset, 
    DiskBackedSequenceDataset
)


class MemoryEfficientDataLoaderService:
    """
    Enhanced DataLoaderService with memory-efficient alternatives that preserve global shuffling.
    
    Provides three strategies:
    1. 'original' - Original approach (loads all sequences in memory)
    2. 'efficient' - On-demand loading with file caching (MemoryEfficientSequenceDataset)
    3. 'hybrid' - Chunked loading with global shuffling (HybridSequenceDataset)  
    4. 'disk' - Disk-backed sequences for very large datasets (DiskBackedSequenceDataset)
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def log_memory_usage(self, stage: str):
        """Log current memory usage."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        self.logger.info(f"Memory usage at {stage}: {memory_mb:.2f} MB")

    def create_data_loaders_original(self, folder_path: str, training_method: str, feature_cols: list, target_col: str,
                            batch_size: int = 32, num_workers: int = 0, lookback: int = None, 
                            concatenate_raw_data: bool = False, train_split: float = 0.8, seed: int = None):
        """
        Original approach - loads all sequences in memory with aggressive memory management.
        Keeps ALL sequences but aggressively cleans up temporary objects, duplicate tensors, and garbage.
        """
        if seed is None:
            seed = int(datetime.now().timestamp())
        
        self.log_memory_usage("BEFORE data loading")
        self.logger.info(f"Creating data loaders with MEMORY-OPTIMIZED ORIGINAL method: {training_method}")

        if training_method == "Sequence-to-Sequence":
            if lookback is None or lookback <= 0:
                raise ValueError("Lookback must be a positive integer for SequenceRNN training method.")
            handler = SequenceRNNDataHandler(feature_cols, target_col, lookback, concatenate_raw_data)
        elif training_method == "WholeSequenceFNN":
            handler = WholeSequenceFNNDataHandler(feature_cols, target_col)
        else:
            raise ValueError(f"Unsupported training_method: {training_method}")
        
        handler.logger = self.logger

        self.log_memory_usage("BEFORE handler.load_and_process_data")
        X, y = handler.load_and_process_data(folder_path)
        self.log_memory_usage("AFTER handler.load_and_process_data - got all sequences")
        
        # Log the actual number of sequences kept
        if hasattr(X, 'shape'):
            self.logger.info(f"Loaded ALL {X.shape[0]} sequences (no reduction)")

        if X.size == 0 or y.size == 0:
            self.logger.warning("No data loaded. Returning empty DataLoaders.")
            empty_dataset = TensorDataset(torch.empty(0), torch.empty(0))
            empty_loader = DataLoader(empty_dataset, batch_size=batch_size)
            return empty_loader, empty_loader

        # Log data types and memory before tensor conversion
        self.logger.info(f"Data types before tensor conversion: X={X.dtype}, y={y.dtype}")
        
        # Convert to tensors with explicit cleanup
        self.log_memory_usage("BEFORE tensor conversion")
        X_tensor = torch.tensor(X, dtype=torch.float32)
        del X  # Immediately delete numpy array
        gc.collect()
        self.log_memory_usage("AFTER X tensor conversion and X cleanup")
        
        y_tensor = torch.tensor(y, dtype=torch.float32)
        del y  # Immediately delete numpy array
        gc.collect()
        self.log_memory_usage("AFTER y tensor conversion and y cleanup")
        
        if y_tensor.ndim == 1:
            y_tensor = y_tensor.unsqueeze(1)

        # Create dataset and immediately clean up tensor references
        self.log_memory_usage("BEFORE dataset creation")
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # Keep references to tensors only through dataset, clean local references
        dataset_size = len(dataset)
        del X_tensor, y_tensor
        gc.collect()
        self.log_memory_usage("AFTER dataset creation and tensor cleanup")
        
        # Create indices for splitting
        indices = list(range(dataset_size))
        
        # Set seed and shuffle
        np.random.seed(seed)
        np.random.shuffle(indices)

        # Split indices
        train_size = int(len(indices) * train_split)
        train_indices = indices[:train_size]
        valid_indices = indices[train_size:]
        
        # Clean up full indices list
        del indices
        gc.collect()

        # Create samplers
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)
        
        # Clean up index lists after sampler creation
        del train_indices, valid_indices
        gc.collect()
        self.log_memory_usage("AFTER sampler creation and indices cleanup")

        # Create DataLoaders with memory-optimized settings
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, 
                                drop_last=True, num_workers=0, pin_memory=False, persistent_workers=False)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, 
                              drop_last=True, num_workers=0, pin_memory=False, persistent_workers=False)

        # Clean up samplers and dataset reference (DataLoaders now own the data)
        del train_sampler, valid_sampler, dataset
        gc.collect()
        self.log_memory_usage("AFTER DATA LOADER CREATION - final cleanup")

        return train_loader, val_loader

    def create_data_loaders_efficient(self, folder_path: str, feature_cols: list, target_col: str,
                             batch_size: int = 32, num_workers: int = 0, lookback: int = None,
                             train_split: float = 0.8, seed: int = None, cache_size: int = 5):
        """Memory-efficient approach with on-demand loading and file caching."""
        if lookback is None or lookback <= 0:
            raise ValueError("Lookback must be a positive integer for efficient sequence loading.")
        
        if seed is None:
            seed = int(datetime.now().timestamp())

        self.log_memory_usage("BEFORE efficient dataset creation")
        self.logger.info(f"Creating MEMORY-EFFICIENT data loaders with cache_size={cache_size}")

        # Create memory-efficient dataset
        full_dataset = MemoryEfficientSequenceDataset(
            folder_path=folder_path,
            feature_cols=feature_cols,
            target_col=target_col,
            lookback=lookback,
            seed=seed,
            cache_size=cache_size
        )

        if len(full_dataset) == 0:
            self.logger.warning("No sequences found. Returning empty DataLoaders.")
            empty_dataset = TensorDataset(torch.empty(0), torch.empty(0))
            empty_loader = DataLoader(empty_dataset, batch_size=batch_size)
            return empty_loader, empty_loader

        # Split indices
        total_size = len(full_dataset)
        indices = list(range(total_size))
        
        # Note: Global shuffling already done in dataset creation
        train_size = int(total_size * train_split)
        train_indices, valid_indices = indices[:train_size], indices[train_size:]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)

        # Create DataLoaders
        # Note: num_workers > 0 might cause issues with file caching, use carefully
        train_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=train_sampler,
                                drop_last=True, num_workers=0, pin_memory=False)  # Use 0 workers for caching safety
        val_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=valid_sampler,
                              drop_last=True, num_workers=0, pin_memory=False)

        self.log_memory_usage("AFTER efficient data loader creation")
        
        return train_loader, val_loader

    def create_data_loaders_hybrid(self, folder_path: str, feature_cols: list, target_col: str,
                          batch_size: int = 32, num_workers: int = 0, lookback: int = None,
                          train_split: float = 0.8, seed: int = None, chunk_size: int = 10):
        """Hybrid approach: chunked loading with global shuffling."""
        if lookback is None or lookback <= 0:
            raise ValueError("Lookback must be a positive integer for hybrid sequence loading.")
        
        if seed is None:
            seed = int(datetime.now().timestamp())

        self.log_memory_usage("BEFORE hybrid dataset creation")
        self.logger.info(f"Creating HYBRID data loaders with chunk_size={chunk_size}")

        # Create hybrid dataset
        full_dataset = HybridSequenceDataset(
            folder_path=folder_path,
            feature_cols=feature_cols,
            target_col=target_col,
            lookback=lookback,
            seed=seed,
            chunk_size=chunk_size
        )

        if len(full_dataset) == 0:
            self.logger.warning("No sequences found. Returning empty DataLoaders.")
            empty_dataset = TensorDataset(torch.empty(0), torch.empty(0))
            empty_loader = DataLoader(empty_dataset, batch_size=batch_size)
            return empty_loader, empty_loader

        # Split indices (data already globally shuffled in dataset)
        total_size = len(full_dataset)
        train_size = int(total_size * train_split)
        
        train_indices = list(range(train_size))
        valid_indices = list(range(train_size, total_size))

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)

        # Create DataLoaders
        train_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=train_sampler,
                                drop_last=True, num_workers=num_workers, pin_memory=False, persistent_workers=False)
        val_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=valid_sampler,
                              drop_last=True, num_workers=num_workers, pin_memory=False, persistent_workers=False)

        self.log_memory_usage("AFTER hybrid data loader creation")
        
        return train_loader, val_loader

    def create_data_loaders_disk(self, folder_path: str, feature_cols: list, target_col: str,
                        batch_size: int = 32, num_workers: int = 0, lookback: int = None,
                        train_split: float = 0.8, seed: int = None, temp_dir: str = None):
        """Disk-backed approach for very large datasets."""
        if lookback is None or lookback <= 0:
            raise ValueError("Lookback must be a positive integer for disk-backed sequence loading.")
        
        if seed is None:
            seed = int(datetime.now().timestamp())

        self.log_memory_usage("BEFORE disk dataset creation")
        self.logger.info(f"Creating DISK-BACKED data loaders with temp_dir={temp_dir}")

        # Create disk-backed dataset
        full_dataset = DiskBackedSequenceDataset(
            folder_path=folder_path,
            feature_cols=feature_cols,
            target_col=target_col,
            lookback=lookback,
            seed=seed,
            temp_dir=temp_dir
        )

        if len(full_dataset) == 0:
            self.logger.warning("No sequences found. Returning empty DataLoaders.")
            empty_dataset = TensorDataset(torch.empty(0), torch.empty(0))
            empty_loader = DataLoader(empty_dataset, batch_size=batch_size)
            return empty_loader, empty_loader

        # Split indices (data already globally shuffled in dataset)
        total_size = len(full_dataset)
        train_size = int(total_size * train_split)
        
        train_indices = list(range(train_size))
        valid_indices = list(range(train_size, total_size))

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)

        # Create DataLoaders
        train_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=train_sampler,
                                drop_last=True, num_workers=0, pin_memory=False)  # Use 0 workers for disk safety
        val_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=valid_sampler,
                              drop_last=True, num_workers=0, pin_memory=False)

        self.log_memory_usage("AFTER disk data loader creation")
        
        return train_loader, val_loader

    def create_data_loaders_reduced_overlap(self, folder_path: str, feature_cols: list, target_col: str,
                                           batch_size: int = 32, num_workers: int = 0, lookback: int = None,
                                           train_split: float = 0.8, seed: int = None, stride: int = None):
        """
        Ultra memory-efficient approach: Reduce sequence overlap dramatically.
        
        Instead of sliding window with stride=1 (every timestep):
        Use stride=lookback//4 or custom stride to create far fewer sequences.
        
        This reduces sequences from 3.26M to ~80K sequences (40x less memory)!
        """
        if lookback is None or lookback <= 0:
            raise ValueError("Lookback must be a positive integer.")
        
        if stride is None:
            stride = max(1, lookback // 4)  # Default: 75% overlap reduction
        
        if seed is None:
            seed = int(datetime.now().timestamp())

        self.log_memory_usage("BEFORE reduced overlap dataset creation")
        self.logger.info(f"Creating REDUCED OVERLAP data loaders with stride={stride} (was stride=1)")

        import glob
        import fireducks.pandas as pd
        
        csv_files = glob.glob(f"{folder_path}/*.csv")
        if not csv_files:
            raise ValueError(f"No CSV files found in {folder_path}")
        
        all_sequences = []
        all_targets = []
        
        total_original_sequences = 0
        total_reduced_sequences = 0
        
        self.logger.info(f"Processing {len(csv_files)} files with stride={stride}")
        
        for file_idx, file_path in enumerate(csv_files):
            try:
                # Load file
                df = pd.read_csv(file_path)
                
                if len(df) < lookback:
                    self.logger.warning(f"File {file_path} too short ({len(df)} < {lookback}), skipping")
                    continue
                
                # Select only required columns
                data = df[feature_cols + [target_col]].values.astype(np.float32)
                
                # Calculate sequences with stride
                original_num_sequences = len(data) - lookback + 1
                
                # Create sequences with larger stride (much fewer sequences)
                for i in range(0, len(data) - lookback + 1, stride):
                    sequence = data[i:i+lookback, :len(feature_cols)]  # Features
                    target = data[i+lookback-1, len(feature_cols)]     # Target (last timestep)
                    
                    all_sequences.append(sequence)
                    all_targets.append(target)
                
                reduced_num_sequences = len(range(0, len(data) - lookback + 1, stride))
                total_original_sequences += original_num_sequences
                total_reduced_sequences += reduced_num_sequences
                
                self.logger.info(f"File {file_idx+1}/{len(csv_files)}: {reduced_num_sequences} sequences (was {original_num_sequences})")
                
                # Force cleanup
                del df, data
                gc.collect()
                
            except Exception as e:
                self.logger.error(f"Error processing file {file_path}: {e}")
                continue
        
        self.logger.info(f"Sequence reduction: {total_original_sequences} → {total_reduced_sequences} "
                        f"({100 * total_reduced_sequences / total_original_sequences:.1f}% of original)")
        
        if not all_sequences:
            self.logger.warning("No sequences created. Returning empty DataLoaders.")
            empty_dataset = TensorDataset(torch.empty(0), torch.empty(0))
            empty_loader = DataLoader(empty_dataset, batch_size=batch_size)
            return empty_loader, empty_loader
        
        # Convert to numpy arrays
        X = np.array(all_sequences, dtype=np.float32)
        y = np.array(all_targets, dtype=np.float32)
        
        self.logger.info(f"Final data shapes: X={X.shape}, y={y.shape}")
        
        # Calculate memory usage
        memory_gb = (X.nbytes + y.nbytes) / 1024 / 1024 / 1024
        self.logger.info(f"Sequence data memory: {memory_gb:.2f} GB")
        
        # Global shuffle for randomization
        indices = np.arange(len(X))
        np.random.seed(seed)
        np.random.shuffle(indices)
        
        X = X[indices]
        y = y[indices]
        
        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        if y_tensor.ndim == 1:
            y_tensor = y_tensor.unsqueeze(1)
        
        # Cleanup numpy arrays
        del X, y, all_sequences, all_targets
        gc.collect()
        
        # Create dataset and split
        dataset = TensorDataset(X_tensor, y_tensor)
        total_size = len(dataset)
        train_size = int(total_size * train_split)
        
        train_indices = list(range(train_size))
        valid_indices = list(range(train_size, total_size))
        
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)
        
        # Create DataLoaders
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler,
                                drop_last=True, num_workers=0, pin_memory=False)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler,
                              drop_last=True, num_workers=0, pin_memory=False)
        
        self.log_memory_usage("AFTER reduced overlap data loader creation")
        
        return train_loader, val_loader

    def create_data_loaders(self, folder_path: str, training_method: str, feature_cols: list, target_col: str,
                          batch_size: int = 32, num_workers: int = 0, lookback: int = None, 
                          concatenate_raw_data: bool = False, train_split: float = 0.8, seed: int = None,
                          memory_strategy: str = 'efficient', **strategy_kwargs):
        """
        Main interface for creating data loaders with different memory strategies.
        
        Args:
            memory_strategy: One of 'original', 'efficient', 'hybrid', 'disk', 'reduced'
                - 'efficient': RECOMMENDED for low-memory environments (on-demand loading)
                - 'original': Loads ALL sequences with aggressive memory cleanup  
                - 'reduced': Fast but reduces data (NOT recommended for full training)
            **strategy_kwargs: Additional arguments for specific strategies:
                - cache_size (int): For 'efficient' strategy (default 5)
                - chunk_size (int): For 'hybrid' strategy (default 10)  
                - temp_dir (str): For 'disk' strategy (default None, uses system temp)
                - stride (int): For 'reduced' strategy (default lookback//4)
        """
        
        # Only sequence-to-sequence supports memory-efficient strategies for now
        if training_method != "Sequence-to-Sequence" and memory_strategy != 'original':
            self.logger.warning(f"Memory strategy '{memory_strategy}' only supports Sequence-to-Sequence. Falling back to original.")
            memory_strategy = 'original'
        
        if memory_strategy == 'original':
            return self.create_data_loaders_original(
                folder_path, training_method, feature_cols, target_col,
                batch_size, num_workers, lookback, concatenate_raw_data, train_split, seed
            )
        elif memory_strategy == 'reduced':
            stride = strategy_kwargs.get('stride', None)
            return self.create_data_loaders_reduced_overlap(
                folder_path, feature_cols, target_col, batch_size, num_workers,
                lookback, train_split, seed, stride
            )
        elif memory_strategy == 'efficient':
            cache_size = strategy_kwargs.get('cache_size', 5)
            return self.create_data_loaders_efficient(
                folder_path, feature_cols, target_col, batch_size, num_workers, 
                lookback, train_split, seed, cache_size
            )
        elif memory_strategy == 'hybrid':
            chunk_size = strategy_kwargs.get('chunk_size', 10)
            return self.create_data_loaders_hybrid(
                folder_path, feature_cols, target_col, batch_size, num_workers,
                lookback, train_split, seed, chunk_size
            )
        elif memory_strategy == 'disk':
            temp_dir = strategy_kwargs.get('temp_dir', None)
            return self.create_data_loaders_disk(
                folder_path, feature_cols, target_col, batch_size, num_workers,
                lookback, train_split, seed, temp_dir
            )
        elif memory_strategy == 'reduced_overlap':
            stride = strategy_kwargs.get('stride', None)
            return self.create_data_loaders_reduced_overlap(
                folder_path, feature_cols, target_col, batch_size, num_workers,
                lookback, train_split, seed, stride
            )
        else:
            raise ValueError(f"Unknown memory strategy: {memory_strategy}. Use 'original', 'efficient', 'hybrid', 'disk' or 'reduced_overlap'.")
