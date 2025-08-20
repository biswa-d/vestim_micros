import os
import numpy as np
import fireducks.pandas as pd
import torch
from torch.utils.data import Dataset
import gc
import logging
from typing import List, Tuple, Dict
import pickle
import tempfile
import shutil


class MemoryEfficientSequenceDataset(Dataset):
    """
    Memory-efficient dataset that maintains global sequence shuffling while loading data on-demand.
    
    This approach:
    1. Scans all CSV files to build a global index of all possible sequences
    2. Shuffles this global index to maintain randomization across all files
    3. Loads and creates sequences on-demand during training (via __getitem__)
    4. Uses file caching to avoid re-reading the same file multiple times
    """
    
    def __init__(self, folder_path: str, feature_cols: List[str], target_col: str, 
                 lookback: int, seed: int = None, cache_size: int = 5):
        """
        Args:
            folder_path: Path containing CSV files
            feature_cols: List of feature column names
            target_col: Target column name
            lookback: Sequence length for RNN
            seed: Random seed for shuffling
            cache_size: Number of files to keep in memory cache (default 5)
        """
        self.folder_path = folder_path
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.lookback = lookback
        self.seed = seed
        self.cache_size = cache_size
        self.logger = logging.getLogger(__name__)
        
        # File cache: {file_path: (X_data, y_data)}
        self.file_cache = {}
        self.cache_order = []  # For LRU eviction
        
        # Build global sequence index
        self.sequence_index = self._build_global_sequence_index()
        self.total_sequences = len(self.sequence_index)
        
        # Shuffle the global index to maintain randomization
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(self.sequence_index)
        
        self.logger.info(f"MemoryEfficientSequenceDataset initialized with {self.total_sequences} sequences across {len(set(item[0] for item in self.sequence_index))} files")
    
    def _build_global_sequence_index(self) -> List[Tuple[str, int]]:
        """
        Build a global index of all possible sequences.
        Returns list of (file_path, sequence_start_index) tuples.
        """
        csv_files = [os.path.join(self.folder_path, f) 
                    for f in os.listdir(self.folder_path) if f.endswith('.csv')]
        
        sequence_index = []
        
        for file_path in csv_files:
            try:
                # Read only the shape to determine sequence count
                df = pd.read_csv(file_path)
                if len(df) <= self.lookback:
                    self.logger.warning(f"File {file_path} has insufficient data (length {len(df)}) for lookback {self.lookback}")
                    continue
                
                # Add all possible sequence start indices for this file
                num_sequences = len(df) - self.lookback
                for seq_start in range(num_sequences):
                    sequence_index.append((file_path, seq_start))
                
                del df
                gc.collect()
                
            except Exception as e:
                self.logger.error(f"Error processing file {file_path}: {e}")
                continue
        
        return sequence_index
    
    def _load_file_data(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and cache file data with LRU eviction."""
        if file_path in self.file_cache:
            # Move to end (most recently used)
            self.cache_order.remove(file_path)
            self.cache_order.append(file_path)
            return self.file_cache[file_path]
        
        # Load file data
        try:
            df = pd.read_csv(file_path)
            X_data = df[self.feature_cols].values
            y_data = df[self.target_col].values
            
            # Cache the data
            self.file_cache[file_path] = (X_data, y_data)
            self.cache_order.append(file_path)
            
            # Evict least recently used files if cache is full
            while len(self.file_cache) > self.cache_size:
                lru_file = self.cache_order.pop(0)
                del self.file_cache[lru_file]
            
            del df
            gc.collect()
            
            return X_data, y_data
            
        except Exception as e:
            self.logger.error(f"Error loading file {file_path}: {e}")
            raise
    
    def __len__(self):
        return self.total_sequences
    
    def __getitem__(self, idx):
        """Load sequence on-demand."""
        file_path, seq_start = self.sequence_index[idx]
        
        # Get file data (from cache or load)
        X_data, y_data = self._load_file_data(file_path)
        
        # Create the sequence
        X_sequence = X_data[seq_start:seq_start + self.lookback]
        y_target = y_data[seq_start + self.lookback]
        
        return torch.tensor(X_sequence, dtype=torch.float32), torch.tensor(y_target, dtype=torch.float32)
    
    def clear_cache(self):
        """Clear the file cache to free memory."""
        self.file_cache.clear()
        self.cache_order.clear()
        gc.collect()


class HybridSequenceDataset(Dataset):
    """
    Hybrid approach: Load files in chunks, create sequences per chunk, maintain global shuffling.
    This balances memory usage with I/O efficiency.
    """
    
    def __init__(self, folder_path: str, feature_cols: List[str], target_col: str, 
                 lookback: int, seed: int = None, chunk_size: int = 10):
        """
        Args:
            chunk_size: Number of files to process together in each chunk
        """
        self.folder_path = folder_path
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.lookback = lookback
        self.seed = seed
        self.chunk_size = chunk_size
        self.logger = logging.getLogger(__name__)
        
        # Build chunked data with global shuffling
        self.X_sequences, self.y_sequences = self._load_and_shuffle_chunked_data()
        self.total_sequences = len(self.X_sequences)
        
        self.logger.info(f"HybridSequenceDataset initialized with {self.total_sequences} sequences")
    
    def _load_and_shuffle_chunked_data(self):
        """Load data in chunks and maintain global shuffling."""
        csv_files = [os.path.join(self.folder_path, f) 
                    for f in os.listdir(self.folder_path) if f.endswith('.csv')]
        
        # Process files in chunks
        all_sequences_X = []
        all_sequences_y = []
        
        for i in range(0, len(csv_files), self.chunk_size):
            chunk_files = csv_files[i:i + self.chunk_size]
            chunk_X, chunk_y = self._process_file_chunk(chunk_files)
            
            if len(chunk_X) > 0:
                all_sequences_X.append(chunk_X)
                all_sequences_y.append(chunk_y)
            
            # Clean up after each chunk
            gc.collect()
        
        if not all_sequences_X:
            return np.array([]), np.array([])
        
        # Concatenate all chunks
        X_all = np.concatenate(all_sequences_X, axis=0)
        y_all = np.concatenate(all_sequences_y, axis=0)
        
        # Global shuffle
        if self.seed is not None:
            np.random.seed(self.seed)
        
        indices = np.arange(len(X_all))
        np.random.shuffle(indices)
        
        return X_all[indices], y_all[indices]
    
    def _process_file_chunk(self, file_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Process a chunk of files and return all sequences."""
        chunk_sequences_X = []
        chunk_sequences_y = []
        
        for file_path in file_paths:
            try:
                df = pd.read_csv(file_path)
                if len(df) <= self.lookback:
                    continue
                
                X_data = df[self.feature_cols].values
                y_data = df[self.target_col].values
                
                # Create sequences for this file
                for i in range(self.lookback, len(X_data)):
                    X_seq = X_data[i - self.lookback:i]
                    y_target = y_data[i]
                    chunk_sequences_X.append(X_seq)
                    chunk_sequences_y.append(y_target)
                
                del df, X_data, y_data
                gc.collect()
                
            except Exception as e:
                self.logger.error(f"Error processing file {file_path}: {e}")
                continue
        
        if not chunk_sequences_X:
            return np.array([]), np.array([])
        
        return np.array(chunk_sequences_X), np.array(chunk_sequences_y)
    
    def __len__(self):
        return self.total_sequences
    
    def __getitem__(self, idx):
        return torch.tensor(self.X_sequences[idx], dtype=torch.float32), torch.tensor(self.y_sequences[idx], dtype=torch.float32)


class DiskBackedSequenceDataset(Dataset):
    """
    Disk-backed approach: Store sequences temporarily on disk, load on-demand.
    Best for very large datasets that don't fit in memory at all.
    """
    
    def __init__(self, folder_path: str, feature_cols: List[str], target_col: str, 
                 lookback: int, seed: int = None, temp_dir: str = None):
        self.folder_path = folder_path
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.lookback = lookback
        self.seed = seed
        self.logger = logging.getLogger(__name__)
        
        # Create temporary directory for storing sequences
        if temp_dir:
            self.temp_dir = temp_dir
            os.makedirs(temp_dir, exist_ok=True)
        else:
            self.temp_dir = tempfile.mkdtemp(prefix="vestim_sequences_")
        
        # Build sequence files and global index
        self.sequence_files, self.total_sequences = self._build_sequence_files()
        
        # Create global shuffled index
        self.global_index = list(range(self.total_sequences))
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(self.global_index)
        
        self.logger.info(f"DiskBackedSequenceDataset initialized with {self.total_sequences} sequences in {self.temp_dir}")
    
    def _build_sequence_files(self) -> Tuple[List[str], int]:
        """Build sequence files on disk and return file list and total count."""
        csv_files = [os.path.join(self.folder_path, f) 
                    for f in os.listdir(self.folder_path) if f.endswith('.csv')]
        
        sequence_files = []
        total_count = 0
        
        for file_idx, file_path in enumerate(csv_files):
            try:
                df = pd.read_csv(file_path)
                if len(df) <= self.lookback:
                    continue
                
                X_data = df[self.feature_cols].values
                y_data = df[self.target_col].values
                
                # Create sequences and save to disk
                sequences_X = []
                sequences_y = []
                
                for i in range(self.lookback, len(X_data)):
                    X_seq = X_data[i - self.lookback:i]
                    y_target = y_data[i]
                    sequences_X.append(X_seq)
                    sequences_y.append(y_target)
                
                if sequences_X:
                    # Save sequences to disk
                    seq_file_path = os.path.join(self.temp_dir, f"sequences_{file_idx}.npz")
                    np.savez_compressed(seq_file_path, 
                                      X=np.array(sequences_X), 
                                      y=np.array(sequences_y))
                    sequence_files.append(seq_file_path)
                    total_count += len(sequences_X)
                
                del df, X_data, y_data, sequences_X, sequences_y
                gc.collect()
                
            except Exception as e:
                self.logger.error(f"Error processing file {file_path}: {e}")
                continue
        
        return sequence_files, total_count
    
    def _find_sequence_location(self, global_idx: int) -> Tuple[str, int]:
        """Find which file and local index contains the global sequence index."""
        current_count = 0
        
        for seq_file in self.sequence_files:
            # Load file to check size
            data = np.load(seq_file)
            file_size = len(data['X'])
            
            if global_idx < current_count + file_size:
                local_idx = global_idx - current_count
                return seq_file, local_idx
            
            current_count += file_size
        
        raise IndexError(f"Global index {global_idx} out of range")
    
    def __len__(self):
        return self.total_sequences
    
    def __getitem__(self, idx):
        # Get the actual global index after shuffling
        global_idx = self.global_index[idx]
        
        # Find the file and local index
        seq_file, local_idx = self._find_sequence_location(global_idx)
        
        # Load the specific sequence
        data = np.load(seq_file)
        X_seq = data['X'][local_idx]
        y_target = data['y'][local_idx]
        
        return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y_target, dtype=torch.float32)
    
    def cleanup(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.cleanup()
