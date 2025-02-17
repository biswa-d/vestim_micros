import os
import numpy as np
import pandas as pd
import h5py  # Make sure to import h5py for reading HDF5 files

class VEstimTestDataService:
    def __init__(self):
        print("Initializing VEstimTestDataService...")

    def load_test_data(self, folder_path):
        """
        Loads test data from HDF5 files **without** lookback handling.

        :param folder_path: Path to the folder containing the test HDF5 files.
        :return: Arrays of input features (X) and output values (y).
        """
        print(f"Loading test data from folder: {folder_path}")

        hdf5_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.h5')]
        print(f"Found {len(hdf5_files)} HDF5 files: {hdf5_files}")

        all_X, all_y = [], []

        for file in hdf5_files:
            print(f"Processing file: {file}")

            with h5py.File(file, 'r') as hdf:
                SOC = hdf['SOC'][:]
                Current = hdf['Current'][:]
                Temp = hdf['Temp'][:]
                Voltage = hdf['Voltage'][:]

            # Convert to feature & target arrays
            X_data = np.column_stack((SOC, Current, Temp))  # Shape: (N, 3)
            Y_data = Voltage  # Shape: (N,)

            print(f"Extracted features: {X_data.shape}, Target: {Y_data.shape}")

            all_X.append(X_data)
            all_y.append(Y_data)

        # Concatenate all files
        X_combined = np.concatenate(all_X, axis=0)
        y_combined = np.concatenate(all_y, axis=0)

        print(f"Final Test Data Shape: X={X_combined.shape}, y={y_combined.shape}")
        return X_combined, y_combined
