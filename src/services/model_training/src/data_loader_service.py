import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

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

    def create_data_loader(self, train_folder, lookback, batch_size):
        """
        Creates a DataLoader for the given training data.

        :param train_folder: Path to the folder containing the training data files.
        :param lookback: The lookback window for creating sequences.
        :param batch_size: The batch size for the DataLoader.
        :return: A PyTorch DataLoader object.
        """
        # Generate data sequences
        X_data, Y_data = self.create_data_sequences(train_folder, lookback)

        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X_data, dtype=torch.float32)
        Y_tensor = torch.tensor(Y_data, dtype=torch.float32)

        # Create a TensorDataset and DataLoader
        dataset = TensorDataset(X_tensor, Y_tensor)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return data_loader
