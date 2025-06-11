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

    def load_and_process_data(self, folder_path, lookback):
        """
        Loads and processes HDF5 files into data sequences with padding and filtering.

        :param folder_path: Path to the folder containing the HDF5 files.
        :param lookback: The lookback window for creating sequences.
        :return: Arrays of input sequences and corresponding output values.
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
                    
                    # Combine the input features and apply filtering
                    X_data = np.column_stack((SOC, Current, Temp))
                    X_data = self.apply_filter(X_data)  # Apply filtering
                    
                    # Filter the output (Voltage) as well
                    Y_data = self.apply_filter(Voltage)
                    
                    # Create input-output sequences
                    X, y = self.create_data_sequence(X_data, Y_data, lookback)
                    if len(X) > 0 and len(y) > 0:
                        data_sequences.append(X)
                        target_sequences.append(y)
                    else:
                        self.logger.warning(f"No data sequences generated for file: {file}")
            
            except Exception as e:
                self.logger.error(f"Error processing file {file}: {str(e)}")
                continue

        if len(data_sequences) == 0 or len(target_sequences) == 0:
            raise ValueError("No valid data sequences were generated from the HDF5 files.")

        # Combine all sequences from different files
        X_combined = np.concatenate(data_sequences, axis=0)
        y_combined = np.concatenate(target_sequences, axis=0)

        # Clean up cache after processing
        del data_sequences, target_sequences

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
        Creates DataLoaders for training and validation data with filtering and padding.

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
        print(f"Loaded data with shape: {X.shape}, Target shape: {y.shape}")

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
        print(f"Train indices: {len(train_indices)}, Validation indices: {len(valid_indices)}")

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)
        print(f"Total training samples: {len(train_sampler)}")
        print(f"Total validation samples: {len(valid_sampler)}")

        # Create DataLoaders with num_workers included
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True, num_workers=num_workers)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, drop_last=True, num_workers=num_workers)
        print(f"Total number of batches in train_loader: {len(train_loader)}")
        print(f"Total number of batches in val_loader: {len(val_loader)}")

        # Clean up cache variables after DataLoaders are created
        del X_tensor, y_tensor, indices, train_indices, valid_indices

        return train_loader, val_loader
