#----------------------------------------------------------------------------------------
#Descrition: This file _1 is to implement the testing service without sequential data preparationfor testing the LSTM model
#
# Created On: Tue Sep 24 2024 16:51:00
# Author: Biswanath Dehury
# Company: Dr. Phil Kollmeyer's Battery Lab at McMaster University
# Copyright (c) 2024 Biswanath Dehury, Dr. Phil Kollmeyer's Battery Lab at McMaster University
#----------------------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import h5py  # Make sure to import h5py for reading HDF5 files

class VEstimTestDataService:
    def __init__(self):
        print("Initializing VEstimTestDataService...")

    def load_and_process_data(self, folder_path):
        """
        Loads and processes HDF5 files without creating sequences.

        :param folder_path: Path to the folder containing the HDF5 files.
        :return: Arrays of input data (features) and corresponding output values.
        """
        print(f"Loading data from folder: {folder_path}")

        # Retrieve all HDF5 files in the specified folder
        hdf5_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.h5')]
        print(f"Found {len(hdf5_files)} HDF5 files: {hdf5_files}")

        data_features = []
        target_values = []

        for file in hdf5_files:
            print(f"Processing file: {file}")

            # Load HDF5 file into a DataFrame
            with h5py.File(file, 'r') as hdf:
                SOC = hdf['SOC'][:]
                Current = hdf['Current'][:]
                Temp = hdf['Temp'][:]
                Voltage = hdf['Voltage'][:]

            # Create a DataFrame from the loaded data
            df = pd.DataFrame({'SOC': SOC, 'Current': Current, 'Temp': Temp, 'Voltage': Voltage})
            print(f"Loaded {len(df)} rows and {len(df.columns)} columns from {file}.")
            
            # Extract relevant features (SOC, Current, Temp) and target (Voltage)
            X_data = df[['SOC', 'Current', 'Temp']].values
            Y_data = df['Voltage'].values
            
            print(f"Extracted features with shape: {X_data.shape} and target with shape: {Y_data.shape}.")
            
            # Store the features and target
            data_features.append(X_data)
            target_values.append(Y_data)

        # Concatenate all the data into a single array for inputs and outputs
        X_combined = np.concatenate(data_features, axis=0)
        y_combined = np.concatenate(target_values, axis=0)

        print(f"Combined input data shape: {X_combined.shape}, Combined output values shape: {y_combined.shape}.")
        # Clean up memory
        del data_features, target_values

        return X_combined, y_combined


    def create_data_sequence(self, X_data, Y_data, lookback):
        """
        Creates input-output sequences from raw data arrays based on the lookback period.

        :param X_data: Array of input data (features).
        :param Y_data: Array of output data (targets).
        :param lookback: The lookback window for creating sequences.
        :return: Sequences of inputs and outputs.
        """
        print(f"Creating data sequences with lookback: {lookback}...")
        X_sequences, y_sequences = [], []
        
        # Create sequences by iterating over the data with the specified lookback window
        for i in range(lookback, len(X_data)):
            X_sequences.append(X_data[i - lookback:i])  # Collect the features over the lookback period
            y_sequences.append(Y_data[i])  # Collect the target value at the current time step

        print(f"Generated {len(X_sequences)} sequences.")
        return np.array(X_sequences), np.array(y_sequences)
