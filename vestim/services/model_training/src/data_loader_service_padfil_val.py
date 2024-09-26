import os
import numpy as np
import pandas as pd, h5py
import torch
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from datetime import datetime
from scipy.signal import savgol_filter  # Example filter (Savitzky-Golay for smoothing)
import logging


class DataLoaderService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def apply_filter(self, data):
        """
        Apply filtering to the data to smooth it and reduce noise.
        
        :param data: Input data (numpy array) to be filtered.
        :return: Filtered data (numpy array).
        """
        # Apply Savitzky-Golay filter to smooth the data (you can use other filters like a moving average, etc.)
        # Adjust the window length and polyorder to match your needs.
        filtered_data = savgol_filter(data, window_length=11, polyorder=2, axis=0)  # Change window_length and polyorder as per your requirement
        return filtered_data

    def load_and_process_data(self, folder_path):
        """
        Loads and processes HDF5 files into input features and output values without sequence creation.

        :param folder_path: Path to the folder containing the HDF5 files.
        :return: Arrays of input features and corresponding output values.
        """
        hdf5_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.h5')]
        if len(hdf5_files) == 0:
            raise ValueError(f"No HDF5 files found in folder: {folder_path}")
        
        data_sequences = []
        target_sequences = []

        for file in hdf5_files:
            try:
                with h5py.File(file, 'r') as hdf5_file:
                    # Load the datasets from the HDF5 file
                    SOC = hdf5_file['SOC'][:]
                    Current = hdf5_file['Current'][:]
                    Temp = hdf5_file['Temp'][:]
                    Voltage = hdf5_file['Voltage'][:]
                    
                    # Combine the input features
                    X_data = np.column_stack((SOC, Current, Temp))
                    Y_data = Voltage  # Output target is Voltage

                    data_sequences.append(X_data)
                    target_sequences.append(Y_data)
            
            except Exception as e:
                self.logger.error(f"Error processing file {file}: {str(e)}")
                continue

        if len(data_sequences) == 0 or len(target_sequences) == 0:
            raise ValueError("No valid data sequences were generated from the HDF5 files.")

        # Combine all sequences from different files
        X_combined = np.concatenate(data_sequences, axis=0)
        y_combined = np.concatenate(target_sequences, axis=0)

        return X_combined, y_combined


    def create_data_sequence(self, X_data, Y_data, lookback):
        """
        Creates input-output sequences from raw data arrays based on the lookback period with padding.

        :param X_data: Array of input data (features).
        :param Y_data: Array of output data (targets).
        :param lookback: The lookback window for creating sequences.
        :return: Sequences of inputs and outputs.
        """
        X_sequences, y_sequences = [], []

        # Create sequences with padding at the beginning where necessary
        for i in range(lookback, len(X_data)):
            # Pad the sequences at the start if necessary
            if i - lookback < 0:
                pad_length = lookback - i
                X_padded = np.vstack([np.zeros((pad_length, X_data.shape[1])), X_data[0:i]])  # Zero-pad the sequence
            else:
                X_padded = X_data[i - lookback:i]  # Take the lookback window as the sequence

            X_sequences.append(X_padded)
            y_sequences.append(Y_data[i])

        return np.array(X_sequences), np.array(y_sequences)

    def create_data_loaders(self, folder_path, lookback, batch_size, num_workers, train_split=0.7, seed=None):
        """
        Creates DataLoaders for training and validation data with different logic for validation.

        :param folder_path: Path to the folder containing the data files.
        :param lookback: The lookback window for creating sequences (only for training).
        :param batch_size: The batch size for the DataLoader.
        :param num_workers: Number of subprocesses to use for data loading.
        :param train_split: Fraction of data to use for training (rest will be used for validation).
        :param seed: Random seed for reproducibility (default is current time).
        :return: A tuple of (train_loader, val_loader) PyTorch DataLoader objects.
        """
        # Use current time as seed if none is provided
        if seed is None:
            seed = int(datetime.now().timestamp())

        # Load and process the raw data
        X, y = self.load_and_process_data(folder_path)
        print(f"Loaded data with shape: {X.shape}, Target shape: {y.shape}")

        # Split data into training and validation sets
        dataset_size = X.shape[0]
        train_size = int(dataset_size * train_split)
        valid_size = dataset_size - train_size

        indices = list(range(dataset_size))
        np.random.seed(seed)
        np.random.shuffle(indices)

        train_indices = indices[:train_size]
        valid_indices = indices[train_size:]
        print(f"Train indices: {len(train_indices)}, Validation indices: {len(valid_indices)}")

        # Split training and validation data
        X_train, y_train = X[train_indices], y[train_indices]
        X_valid, y_valid = X[valid_indices], y[valid_indices]

        # Create sequences only for the training data
        X_train_seq, y_train_seq = self.create_data_sequence(X_train, y_train, lookback)

        # Convert training data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32)

        # Create a TensorDataset for training data
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        # Create the validation loader without sequences
        val_loader, padding_size = self.create_validation_loader(X_valid, y_valid, valid_indices, batch_size, num_workers)

        print(f"Total number of batches in train_loader: {len(train_loader)}")
        print(f"Total number of batches in val_loader: {len(val_loader)}")

        return train_loader, val_loader, padding_size

    
    def create_validation_loader(self, X, y, valid_indices, batch_size, num_workers):
        """
        Creates a DataLoader for the validation set without sequences and pads data as necessary.

        :param X: Input features (numpy array).
        :param y: Target values (numpy array).
        :param valid_indices: Indices for validation data.
        :param batch_size: Batch size for the validation loader.
        :param num_workers: Number of subprocesses to use for data loading.
        :return: A DataLoader for the validation data.
        """
        # Extract validation data based on indices
        X_valid = X
        y_valid = y

        # Pad the validation data to match batch sizes
        X_padded, y_padded, padding_size = self.pad_data(X_valid, y_valid, batch_size)
        print(f"Validation data padded: X_padded shape: {X_padded.shape}, y_padded shape: {y_padded.shape}, padding_size: {padding_size}")

        # Convert to PyTorch tensors
        X_valid_tensor = torch.tensor(X_padded, dtype=torch.float32)
        y_valid_tensor = torch.tensor(y_padded, dtype=torch.float32)

        # Create a TensorDataset and DataLoader for validation
        valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
        val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        print(f"Total number of batches in val_loader: {len(val_loader)}")

        return val_loader, padding_size

    
    def pad_data(self, X, y, batch_size):
        print(f"Entered pad_data with X shape: {X.shape}, y shape: {y.shape}, batch_size: {batch_size}")
        """
        Pads the input data to ensure the last batch matches the batch size.

        :param X: Input features (numpy array).
        :param y: Target values (numpy array).
        :param batch_size: Desired batch size.
        :return: Padded X, y, and the number of padding rows.
        """
        num_samples = X.shape[0]
        remainder = num_samples % batch_size
        
        if remainder == 0:
            return X, y, 0  # No padding needed

        # Calculate the number of padding samples needed
        padding_size = batch_size - remainder

        # Create padding arrays (all zeros for simplicity)
        X_padding = np.zeros((padding_size, X.shape[1]))
        y_padding = np.zeros((padding_size,))

        # Append the padding to X and y
        X_padded = np.vstack([X, X_padding])
        y_padded = np.concatenate([y, y_padding])

        return X_padded, y_padded, padding_size


