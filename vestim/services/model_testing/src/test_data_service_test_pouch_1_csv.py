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

    def load_and_process_data(folder_path):
        """
        Loads and processes CSV files, extracts features and target values, 
        and returns combined input and output arrays.

        :param folder_path: Path to the folder containing CSV files.
        :return: Arrays of input data (features) and corresponding output values.
        """
        print(f"Loading data from folder: {folder_path}")

        # Retrieve all CSV files in the specified folder
        csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
        print(f"Found {len(csv_files)} CSV files: {csv_files}")

        data_features = []
        target_values = []

        for file in csv_files:
            print(f"Processing file: {file}")

            # Load CSV file into a DataFrame
            df = pd.read_csv(file, usecols=['SOC', 'Current', 'Temp', 'Voltage'])
            print(f"Loaded {len(df)} rows and {len(df.columns)} columns from {file}.")

            # Extract relevant features (SOC, Current, Temp) and target (Voltage)
            X_data = df[['SOC', 'Current', 'Temp']].values
            Y_data = df['Voltage'].values

            print(f"Extracted features with shape: {X_data.shape} and target with shape: {Y_data.shape}.")

            # Store the features and target
            data_features.append(X_data)
            target_values.append(Y_data)

        # Concatenate all the data into a single array for inputs and outputs
        if data_features and target_values:
            X_combined = np.concatenate(data_features, axis=0)
            y_combined = np.concatenate(target_values, axis=0)
            print(f"Combined input data shape: {X_combined.shape}, Combined output values shape: {y_combined.shape}.")
            return X_combined, y_combined
        else:
            print("No CSV files found in the folder.")
            return None, None