import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

class VEstimTestDataService:
    def __init__(self):
        print("Initializing VEstimTestDataService...")

    def create_ordered_loader(self, folder_path, lookback, batch_size):
        """
        Creates a sequential DataLoader for testing/inference without shuffling.

        :param X_data: Feature data (NumPy array).
        :param Y_data: Target data (NumPy array).
        :param lookback: Lookback window for sequence generation.
        :param batch_size: Batch size for DataLoader.
        :return: PyTorch DataLoader.
        """
        # Generate sequences using the modified function
        X_sequences, Y_sequences = self.load_and_process_data(folder_path, lookback, batch_size)
        X_sequences = torch.tensor(X_sequences, dtype=torch.float32)
        Y_sequences = torch.tensor(Y_sequences, dtype=torch.float32)

        # Create dataset and DataLoader (No shuffling to maintain time order)
        dataset = TensorDataset(X_sequences, Y_sequences)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        print(f"Ordered DataLoader created: {len(X_sequences)} sequences, batch size = {batch_size}")
        return loader

    def load_and_process_data(self, folder_path, lookback, batch_size):
        """
        Loads and processes CSV files into data sequences based on the lookback period.

        :param folder_path: Path to the folder containing the CSV files.
        :param lookback: The lookback window for creating sequences.
        :return: Arrays of input sequences and corresponding output values.
        """
        print(f"Loading data from folder: {folder_path}")
        
        # Retrieve all CSV files in the specified folder
        csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
        print(f"Found {len(csv_files)} CSV files: {csv_files}")

        data_sequences = []
        target_sequences = []

        for file in csv_files:
            print(f"Processing file: {file}")
            # Load CSV file into a DataFrame
            df = pd.read_csv(file)
            print(f"Loaded {len(df)} rows and {len(df.columns)} columns from {file}.")
            
            # Extract relevant features (SOC, Current, Temp) and target (Voltage)
            X_data = df[['SOC', 'Current', 'Temp']].values
            Y_data = df['Voltage'].values
            
            print(f"Extracted features with shape: {X_data.shape} and target with shape: {Y_data.shape}.")
            
            # Create sequences using the lookback window
            X, y = self.create_data_sequence(X_data, Y_data, lookback, batch_size)
            print(f"Created input sequences with shape: {X.shape} and output sequences with shape: {y.shape}.")
            
            # Store the sequences
            data_sequences.append(X)
            target_sequences.append(y)

        # Concatenate all the sequences into a single array for inputs and outputs
        X_combined = np.concatenate(data_sequences, axis=0)
        y_combined = np.concatenate(target_sequences, axis=0)

        print(f"Combined input sequences shape: {X_combined.shape}, Combined output sequences shape: {y_combined.shape}.")

        # Clean up memory
        del data_sequences, target_sequences

        return X_combined, y_combined

    def create_data_sequence(self, X_data, Y_data, lookback, batch_size):
        """
        Creates input-output sequences from raw data arrays based on the lookback period.
        If necessary, pads the last sequence to ensure the full dataset is covered.

        :param X_data: Array of input data (features).
        :param Y_data: Array of output data (targets).
        :param lookback: The lookback window for creating sequences.
        :return: Padded sequences of inputs and outputs.
        """
        print(f"Creating data sequences with lookback: {lookback}...")
        X, y = [], []
        for i in range(lookback, len(Y_data)):
            X.append(X_data[i - lookback:i])
            y.append(Y_data[i])
            
        # Convert to tensors
        X = torch.tensor(np.array(X), dtype=torch.float32)
        y = torch.tensor(np.array(y), dtype=torch.float32)
        # **Ensure y is always 2D before padding**
        if y.ndim == 1:
            y = y.unsqueeze(1)  # Converts shape from (N,) → (N,1)
        print(f"Total test sequences created: {len(y)}")

        # Handle padding if the dataset size is not a multiple of batch_size
        pad_size = batch_size - (len(y) % batch_size) if len(y) % batch_size != 0 else 0
        if pad_size > 0:
            print(f"Padding test data with {pad_size} samples to match batch size.")
            X_pad = torch.zeros((pad_size, lookback, X.shape[2]), dtype=torch.float32)  # Zero padding for X
            y_pad = torch.zeros((pad_size, y.shape[1]), dtype=torch.float32)  # Zero padding for y
            X = torch.cat((X, X_pad), dim=0)
            y = torch.cat((y, y_pad), dim=0)
        print(f"Final test data shapes: X = {X.shape}, y = {y.shape}")
        print(f"DEBUG: batch_size = {batch_size}, type = {type(batch_size)}")
        return X, y
    
