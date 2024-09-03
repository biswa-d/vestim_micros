import os
import numpy as np
import pandas as pd

class VEstimTestDataService:
    def __init__(self):
        pass

    def load_and_process_data(self, folder_path, lookback):
        """
        Loads and processes CSV files into data sequences based on the lookback period.

        :param folder_path: Path to the folder containing the CSV files.
        :param lookback: The lookback window for creating sequences.
        :return: Arrays of input sequences and corresponding output values.
        """
        # Retrieve all CSV files in the specified folder
        csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
        data_sequences = []
        target_sequences = []

        for file in csv_files:
            # Load CSV file into a DataFrame
            df = pd.read_csv(file)
            
            # Extract relevant features (SOC, Current, Temp) and target (Voltage)
            X_data = df[['SOC', 'Current', 'Temp']].values
            Y_data = df['Voltage'].values
            
            # Create sequences using the lookback window
            X, y = self.create_data_sequence(X_data, Y_data, lookback)
            
            # Store the sequences
            data_sequences.append(X)
            target_sequences.append(y)

        # Concatenate all the sequences into a single array for inputs and outputs
        X_combined = np.concatenate(data_sequences, axis=0)
        y_combined = np.concatenate(target_sequences, axis=0)

        # Clean up memory
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
        
        # Create sequences by iterating over the data with the specified lookback window
        for i in range(lookback, len(X_data)):
            X_sequences.append(X_data[i - lookback:i])  # Collect the features over the lookback period
            y_sequences.append(Y_data[i])  # Collect the target value at the current time step

        return np.array(X_sequences), np.array(y_sequences)
