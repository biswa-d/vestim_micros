import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from datetime import datetime

class DataLoaderService:
    def __init__(self):
        pass

    def load_and_process_data(self, folder_path, lookback):
        """
        Loads and processes CSV files into data sequences based on the lookback period.

        :param folder_path: Path to the folder containing the CSV files.
        :param lookback: The lookback window for creating sequences.
        :return: Arrays of input sequences and corresponding output values.
        """
        csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
        data_sequences = []
        target_sequences = []

        for file in csv_files:
            df = pd.read_csv(file)
            X_data = df[['Voltage', 'Current', 'Temp']].values
            Y_data = df['SOC'].values
            X, y = self.create_data_sequence(X_data, Y_data, lookback)
            data_sequences.append(X)
            target_sequences.append(y)

        X_combined = np.concatenate(data_sequences, axis=0)
        y_combined = np.concatenate(target_sequences, axis=0)

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
        X_sequences, y_sequences = [], []
        for i in range(lookback, len(X_data)):
            X_sequences.append(X_data[i - lookback:i])
            y_sequences.append(Y_data[i])

        return np.array(X_sequences), np.array(y_sequences)

    def create_data_loaders(self, folder_path, lookback, batch_size, num_workers, train_split=0.7, seed=None):
        """
        Creates DataLoaders for training and validation data.

        :param folder_path: Path to the folder containing the data files.
        :param lookback: The lookback window for creating sequences.
        :param batch_size: The batch size for the DataLoader.
        :param num_workers: Number of subprocesses to use for data loading.
        :param train_split: Fraction of data to use for training (rest will be used for validation).
        :param seed: Random seed for reproducibility (default is current time).
        :return: A tuple of (train_loader, val_loader) PyTorch DataLoader objects.
        """
        # Use current time as seed if none is provided
        if seed is None:
            seed = int(datetime.now().timestamp())
        
        # Load and process data
        X, y = self.load_and_process_data(folder_path, lookback)

        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        # Create a TensorDataset
        dataset = TensorDataset(X_tensor, y_tensor)
        dataset_size = len(dataset)
        indices = list(range(dataset_size))

        # Train-validation split
        train_size = int(dataset_size * train_split)
        valid_size = dataset_size - train_size

        np.random.seed(seed)
        np.random.shuffle(indices)

        train_indices, valid_indices = indices[:train_size], indices[train_size:]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)

        # Create DataLoaders with num_workers included
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True, num_workers=num_workers)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, drop_last=True, num_workers=num_workers)

        # Clean up cache variables after DataLoaders are created
        del X_tensor, y_tensor, indices, train_indices, valid_indices

        return train_loader, val_loader
