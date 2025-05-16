import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from datetime import datetime
import  logging


class DataLoaderService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def load_and_process_data(self, folder_path, lookback, feature_cols, target_col):
        """
        Loads and processes CSV files into data sequences based on the lookback period.

        :param folder_path: Path to the folder containing the CSV files.
        :param lookback: The lookback window for creating sequences.
        :return: Arrays of input sequences and corresponding output values.
        """
        #print("Entered load_and_process_data")
        csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
        data_sequences = []
        target_sequences = []

        for file in csv_files:
            df = pd.read_csv(file)
            X_data = df[feature_cols].values
            Y_data = df[[target_col]].values
            #print(f"shape of X_data and Y_data before sequening: {X_data.shape}, {Y_data.shape}")
            X, y = self.create_data_sequence(X_data, Y_data, lookback)
            data_sequences.append(X)
            target_sequences.append(y)

        if len(data_sequences) > 1:
            X_combined = np.concatenate(data_sequences, axis=0)
            y_combined = np.concatenate(target_sequences, axis=0)
        else:
            print("Only one CSV file found in the folder.")
            X_combined = data_sequences[0]
            y_combined = target_sequences[0]

        # Clean up cache after processing
        del data_sequences, target_sequences

        return X_combined, y_combined


    def create_data_sequence(self, X_data, Y_data, lookback):
        """
        Creates input-output sequences from raw data arrays based on the lookback period.

        :param X_data: Array of input data (features).
        :param Y_data: Array of output data (targets).
        :param lookback: The lookback window for creating sequences.
        :return: Sequences of inputs and outputs.
        """
        #print("Entered create_data_sequence")
        X_sequences, y_sequences = [], []
        # **Padding the first `lookback` rows with the first row values**
        pad_X = np.tile(X_data[0], (lookback, 1))  # Repeat first row for lookback times
        pad_Y = np.tile(Y_data[0], (lookback, 1))  # Repeat first target row
        print(f"size of pad_X and pad_Y: {pad_X.shape}, {pad_Y.shape}")
        # Concatenate padding with the original data
        X_data_padded = np.vstack((pad_X, X_data))
        Y_data_padded = np.vstack((pad_Y, Y_data))

        #print(f"Padded dataset shape: {X_data_padded.shape}")

        # Create sequences
        print(f"Creating sequential dataset with lookback={lookback}...")
        for i in range(lookback, len(Y_data_padded)):
            X_sequences.append(X_data_padded[i - lookback:i])
            y_sequences.append(Y_data_padded[i])

        return np.array(X_sequences), np.array(y_sequences)

    def create_data_loaders(self, folder_path, lookback,feature_cols, target_col, batch_size, num_workers, use_full_train_batch: bool, train_split=0.7, seed=2000):
        """
        Creates DataLoaders for training and validation data.

        :param folder_path: Path to the folder containing the data files.
        :param lookback: The lookback window for creating sequences.
        :param batch_size: The batch size for the DataLoader (used if use_full_train_batch is False).
        :param num_workers: Number of subprocesses to use for data loading.
        :param use_full_train_batch: If True, train_loader batch_size is set to the entire training set size.
        :param train_split: Fraction of data to use for training (rest will be used for validation).
        :param seed: Random seed for reproducibility (default is current time).
        :return: A tuple of (train_loader, val_loader) PyTorch DataLoader objects.
        """
        self.logger.info(f"Entering create_data_loaders. use_full_train_batch: {use_full_train_batch}, user_batch_size: {batch_size}")
        # Use current time as seed if none is provided
        if seed is None:
            seed = int(datetime.now().timestamp())
        
        # Load and process data
        X, y = self.load_and_process_data(folder_path, lookback, feature_cols, target_col)

        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        # Create a TensorDataset
        dataset = TensorDataset(X_tensor, y_tensor)
        dataset_size = len(dataset)
        indices = list(range(dataset_size))

        # Train-validation split
        self.logger.info(f"Using train_split: {train_split}")
        train_size = int(dataset_size * train_split)
        if train_size == 0 and dataset_size > 0 : # Ensure train_size is at least 1 if dataset is not empty
            self.logger.warning(f"Calculated train_size is 0 with dataset_size {dataset_size} and train_split {train_split}. Adjusting to 1 if possible, or full dataset if split is 1.0.")
            if train_split == 1.0:
                train_size = dataset_size
            elif dataset_size > 0:
                 train_size = 1 # Avoid train_size being 0 if there's data.
        
        valid_size = dataset_size - train_size
        self.logger.info(f"Dataset size: {dataset_size}, Train size: {train_size}, Validation size: {valid_size}")


        np.random.seed(seed)
        np.random.shuffle(indices)

        train_indices, valid_indices = indices[:train_size], indices[train_size:]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices) # Re-add the missing valid_sampler definition
        
        effective_train_batch_size = batch_size # User-provided batch_size
        effective_val_batch_size = batch_size   # User-provided batch_size

        if use_full_train_batch:
            if train_size > 0:
                effective_train_batch_size = train_size
                self.logger.info(f"Full batch mode: Effective train batch size set to: {train_size}")
            else: # Should not happen if dataset_size > 0 due to earlier checks
                self.logger.warning("Full batch mode, but train_size is 0. Using user_batch_size or 1 for training.")
                effective_train_batch_size = batch_size if batch_size > 0 else 1
            
            if valid_size > 0:
                effective_val_batch_size = valid_size
                self.logger.info(f"Full batch mode: Effective val batch size set to: {valid_size}")
            else: # No validation data
                self.logger.info("Full batch mode, but no validation data. Val batch size effectively 1 (or user input if >0).")
                effective_val_batch_size = batch_size if batch_size > 0 else 1
        else: # Mini-batch training enabled
             self.logger.info(f"Mini-batch mode: Train batch size: {batch_size}")
             if batch_size == 0:
                 self.logger.warning("User-provided batch_size is 0 for mini-batch training. Setting train and val to 1.")
                 effective_train_batch_size = 1
                 effective_val_batch_size = 1
             else: # Ensure val_batch_size is at least 1 if there's validation data, or capped by valid_size
                 if valid_size > 0:
                     effective_val_batch_size = min(batch_size, valid_size) if batch_size > 0 else valid_size
                 elif valid_size == 0 : # No validation data
                     effective_val_batch_size = 1 # Avoid DataLoader error with batch_size 0
        
        self.logger.info(f"Final effective train batch size: {effective_train_batch_size}, Final effective val batch size: {effective_val_batch_size}")

        # Create DataLoaders with num_workers included
        # Ensure batch_size is at least 1
        train_loader = DataLoader(dataset, batch_size=max(1, effective_train_batch_size), sampler=train_sampler, drop_last=True, num_workers=num_workers)
        val_loader = DataLoader(dataset, batch_size=max(1, effective_val_batch_size), sampler=valid_sampler, drop_last=True, num_workers=num_workers)


        # Clean up cache variables after DataLoaders are created
        del X_tensor, y_tensor, indices, train_indices, valid_indices

        return train_loader, val_loader

    def _load_and_concatenate_raw_data(self, folder_path, feature_cols, target_col):
        """
        Loads all CSVs from a folder and concatenates their raw feature/target data.
        """
        self.logger.info(f"Concatenating raw data from folder: {folder_path}")
        csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
        
        all_X_data = []
        all_Y_data = []

        if not csv_files:
            self.logger.warning(f"No CSV files found in {folder_path} for raw concatenation.")
            return None, None

        for file_idx, file_path in enumerate(csv_files):
            try:
                df = pd.read_csv(file_path)
                if not all(col in df.columns for col in feature_cols) or target_col not in df.columns:
                    self.logger.warning(f"Skipping file {file_path} due to missing columns.")
                    continue
                
                X_data = df[feature_cols].values
                Y_data = df[[target_col]].values # Keep as 2D for consistent concatenation

                all_X_data.append(X_data)
                all_Y_data.append(Y_data)
                self.logger.debug(f"Loaded {X_data.shape[0]} rows from {file_path}")
            except Exception as e:
                self.logger.error(f"Error processing file {file_path} for raw concatenation: {e}")
                continue # Skip problematic files

        if not all_X_data:
            self.logger.warning(f"No data could be loaded and concatenated from {folder_path}.")
            return None, None

        X_all_files_concat = np.concatenate(all_X_data, axis=0)
        Y_all_files_concat = np.concatenate(all_Y_data, axis=0)
        self.logger.info(f"Concatenated raw data: X shape {X_all_files_concat.shape}, Y shape {Y_all_files_concat.shape}")
        
        return X_all_files_concat, Y_all_files_concat

    def create_concatenated_whole_sequence_loaders(self, folder_path, feature_cols, target_col, num_workers, train_split=0.7, seed=2000):
        """
        Creates DataLoaders where the entire concatenated dataset (split into train/val)
        is treated as a single sequence per set, with the target being the final value of that sequence.
        Ignores lookback and batch_size from hyperparams for this mode.
        """
        self.logger.info("Creating concatenated whole sequence loaders.")
        
        X_all_files_concat, Y_all_files_concat = self._load_and_concatenate_raw_data(folder_path, feature_cols, target_col)

        if X_all_files_concat is None or Y_all_files_concat is None or X_all_files_concat.shape[0] == 0:
            self.logger.error("Failed to load or concatenate any raw data. Cannot create whole sequence loaders.")
            # Return empty DataLoaders or raise an error
            empty_dataset = TensorDataset(torch.empty(0, 1), torch.empty(0, 1)) # Dummy dimensions
            return DataLoader(empty_dataset, batch_size=1), DataLoader(empty_dataset, batch_size=1)

        total_len = X_all_files_concat.shape[0]
        if total_len == 0:
            self.logger.warning("Concatenated data is empty. Returning empty DataLoaders.")
            empty_dataset = TensorDataset(torch.empty(0, X_all_files_concat.shape[1] if X_all_files_concat.ndim > 1 else 1), torch.empty(0,1))
            return DataLoader(empty_dataset, batch_size=1), DataLoader(empty_dataset, batch_size=1)

        # Split the single long sequence into train and validation parts
        # Note: Shuffling the entire sequence before split might be an option, but for time-series, a chronological split is common.
        # For simplicity, we'll do a direct chronological split here.
        split_idx = int(total_len * train_split)

        X_train_full = X_all_files_concat[:split_idx]
        Y_train_full_sequence = Y_all_files_concat[:split_idx]
        
        X_val_full = X_all_files_concat[split_idx:]
        Y_val_full_sequence = Y_all_files_concat[split_idx:]

        self.logger.info(f"Whole sequence: Train length {X_train_full.shape[0]}, Val length {X_val_full.shape[0]}")

        # Prepare training data
        if X_train_full.shape[0] > 0:
            X_train_tensor = torch.tensor(X_train_full, dtype=torch.float32).unsqueeze(0) # Shape: [1, train_len, num_features]
            # Target is the entire sequence of Y_train_full_sequence
            Y_train_target_tensor = torch.tensor(Y_train_full_sequence, dtype=torch.float32).unsqueeze(0) # Shape: [1, train_len, num_targets]
            train_dataset = TensorDataset(X_train_tensor, Y_train_target_tensor)
            train_loader = DataLoader(train_dataset, batch_size=1, num_workers=num_workers, shuffle=False) # Shuffle is False for single item
            self.logger.info(f"Train loader created for concatenated whole sequence. X_shape: {X_train_tensor.shape}, Y_shape: {Y_train_target_tensor.shape}")
        else:
            self.logger.warning("Training portion of concatenated sequence is empty.")
            empty_dataset_train = TensorDataset(torch.empty(0, X_all_files_concat.shape[1] if X_all_files_concat.ndim > 1 else 1), torch.empty(0,1))
            train_loader = DataLoader(empty_dataset_train, batch_size=1)

        # Prepare validation data
        if X_val_full.shape[0] > 0:
            X_val_tensor = torch.tensor(X_val_full, dtype=torch.float32).unsqueeze(0) # Shape: [1, val_len, num_features]
            # Target is the entire sequence of Y_val_full_sequence
            Y_val_target_tensor = torch.tensor(Y_val_full_sequence, dtype=torch.float32).unsqueeze(0) # Shape: [1, val_len, num_targets]
            val_dataset = TensorDataset(X_val_tensor, Y_val_target_tensor)
            val_loader = DataLoader(val_dataset, batch_size=1, num_workers=num_workers, shuffle=False)
            self.logger.info(f"Validation loader created for concatenated whole sequence. X_shape: {X_val_tensor.shape}, Y_shape: {Y_val_target_tensor.shape}")
        else:
            self.logger.warning("Validation portion of concatenated sequence is empty.")
            empty_dataset_val = TensorDataset(torch.empty(0, X_all_files_concat.shape[1] if X_all_files_concat.ndim > 1 else 1), torch.empty(0,1))
            val_loader = DataLoader(empty_dataset_val, batch_size=1)
            
        return train_loader, val_loader
