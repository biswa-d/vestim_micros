import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

class DataLoaderService:
    def __init__(self):
        pass

    def create_data_sequences(self, data_folder, lookback):
        """
        Creates input-output sequences from processed data files in the specified folder.

        :param data_folder: Path to the folder containing processed data files.
        :param lookback: Number of previous time steps to consider for each timestep.
        :return: Arrays of input sequences and corresponding output values.
        """
        X_data = []
        Y_data = []

        # Load all files in the data folder
        for filename in os.listdir(data_folder):
            if filename.endswith('.npy'):
                file_path = os.path.join(data_folder, filename)
                data = np.load(file_path)

                # Assuming the data is in the form of (time_steps, features)
                for i in range(lookback, len(data)):
                    X_data.append(data[i-lookback:i, :-1])  # Take all columns except the last one as input
                    Y_data.append(data[i, -1])  # Take the last column as the output

        X_data = np.array(X_data)
        Y_data = np.array(Y_data)

        return X_data, Y_data

    def create_data_loaders(self, data_folder, lookback, batch_size, num_workers=4, valid_split=0.2):
        """
        Creates DataLoaders for training and validation data.

        :param data_folder: Path to the folder containing the data files.
        :param lookback: The lookback window for creating sequences.
        :param batch_size: The batch size for the DataLoader.
        :param num_workers: Number of subprocesses to use for data loading.
        :param valid_split: Fraction of the data to use for validation.
        :return: A tuple of (train_loader, val_loader) PyTorch DataLoader objects.
        """
        # Generate data sequences
        X_data, Y_data = self.create_data_sequences(data_folder, lookback)

        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X_data, dtype=torch.float32)
        Y_tensor = torch.tensor(Y_data, dtype=torch.float32)

        # Create a TensorDataset
        dataset = TensorDataset(X_tensor, Y_tensor)

        # Calculate split sizes
        val_size = int(len(dataset) * valid_split)
        train_size = len(dataset) - val_size

        # Split the dataset
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return train_loader, val_loader
