# ---------------------------------------------------------------------------------
# Author: Biswanath Dehury
# Date: `{{date:YYYY-MM-DD}}`
# Version: 1.0.0
# Description: This returns the test loader for each file in the test folder where the intintial part of the testfile is not missed as we pad the data and 
# make sequences out of it and we pad the initial vales with the same values as the first row of the test file.
# ---------------------------------------------------------------------------------


import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

class VEstimTestDataService:
    def __init__(self):
        print("Initializing VEstimTestDataService...")

    def create_test_file_loader(self, test_file_path, lookback, batch_size):
        """
        Create a test DataLoader from a CSV file without shuffling, ensuring sequential order.
        Pads the initial data to preserve all values when creating sequences.

        Args:
            file_path: Path to the CSV file containing the dataset.
            lookback: Lookback window size for creating sequences (default: 400).
            batch_size: Batch size for DataLoader (default: 100).

        Returns:
            testLoader: PyTorch DataLoader for testing.
        """
        print(f"Loading test data from: {test_file_path}")

        # Load and preprocess data from CSV
        data = pd.read_csv(test_file_path)
        print(f"Dataset shape: {data.shape}")

        X_data = data[["SOC", "Current", "Temp"]].values.astype('float32')
        Y_data = data[["Voltage"]].values.astype('float32')

        # **Padding the first `lookback` rows with the first row values**
        pad_X = np.tile(X_data[0], (lookback, 1))  # Repeat first row for lookback times
        pad_Y = np.tile(Y_data[0], (lookback, 1))  # Repeat first target row
        
        # Concatenate padding with the original data
        X_data_padded = np.vstack((pad_X, X_data))
        Y_data_padded = np.vstack((pad_Y, Y_data))

        print(f"Padded dataset shape: {X_data_padded.shape}")

        # Create sequences for testing
        print(f"Creating sequential test dataset with lookback={lookback}...")
        X, y = [], []
        for i in range(lookback, len(Y_data_padded)):
            X.append(X_data_padded[i - lookback:i])
            y.append(Y_data_padded[i])
        
        # Convert to tensors
        X = torch.tensor(np.array(X))
        y = torch.tensor(np.array(y))
        print(f"Total test sequences created: {len(y)}")
        print(f"Final Test Dataset Size (after padding if needed): {X.shape[0]} samples")

        # Create DataLoader
        dataset = TensorDataset(X, y)
        testLoader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        print("Test DataLoader created successfully!")
        return testLoader
        
