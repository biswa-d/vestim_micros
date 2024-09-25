#----------------------------------------------------------------------------------------
#Descrition: This file _1 is to implement the testing service without sequential data preparationfor testing the LSTM model
#
# Created On: Tue Sep 24 2024 16:50:29
# Author: Biswanath Dehury
# Company: Dr. Phil Kollmeyer's Battery Lab at McMaster University
# Copyright (c) 2024 Biswanath Dehury, Dr. Phil Kollmeyer's Battery Lab at McMaster University
#----------------------------------------------------------------------------------------
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from vestim.services.model_training.src.LSTM_model_service_test import LSTMModel

class VEstimTestingService:
    def __init__(self, device='cpu'):
        print("Initializing VEstimTestingService...")
        """
        Initialize the TestingService with the specified device.

        :param device: Device to run the model on ('cpu' or 'cuda').
        """
        self.device = torch.device(device)

    def load_model(self, model_path):
        """
        Loads a model from the specified .pth file.

        :param model_path: Path to the model .pth file.
        :return: The loaded model.
        """
        model = torch.load(model_path)
        model.to(self.device)
        model.eval()  # Set the model to evaluation mode
        return model

    def test_model(self, model, test_loader, padding_size=0):
        model.eval()  # Ensure the model is in evaluation mode
        all_predictions = []
        all_true_values = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                batch_size = X_batch.size(0)

                # Reshape X_batch to [batch_size, seq_len, input_size]
                X_batch = X_batch.unsqueeze(1)  # Adds a sequence length of 1

                # Initialize hidden and cell states based on batch size
                h_s = torch.zeros(model.num_layers, batch_size, model.hidden_units).to(self.device)
                h_c = torch.zeros(model.num_layers, batch_size, model.hidden_units).to(self.device)
                print(f"X_batch shape after unsqueeze: {X_batch.shape}, batch_size: {batch_size}")
                print(f"h_s shape: {h_s.shape}, h_c shape: {h_c.shape}")

                # Forward pass
                y_pred_tensor, _ = model(X_batch.to(self.device), h_s, h_c)

                # Collect predictions and true values
                all_predictions.append(y_pred_tensor.cpu().numpy())
                all_true_values.append(y_batch.cpu().numpy())

        # Convert to flat arrays for evaluation
        y_pred = np.concatenate(all_predictions, axis=0)
        y_true = np.concatenate(all_true_values, axis=0)

        print(f"y_pred shape before removing padding: {y_pred.shape}")
        print(f"y_true shape before removing padding: {y_true.shape}")

        # Remove the padded data from the results if padding was applied
        if padding_size > 0:
            y_pred = y_pred[:-padding_size]
            y_true = y_true[:-padding_size]

        # Compute evaluation metrics and convert to millivolts (mV)
        rms_error = np.sqrt(mean_squared_error(y_true, y_pred)) * 1000  # Convert to mV
        mae = mean_absolute_error(y_true, y_pred) * 1000  # Convert to mV
        mape = self.calculate_mape(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"RMS Error: {rms_error}, MAE: {mae}, MAPE: {mape}, R²: {r2}")

        return {
            'predictions': y_pred,
            'true_values': y_true,
            'rms_error_mv': rms_error,  # Error in mV
            'mae_mv': mae,  # Error in mV
            'mape': mape,  # MAPE remains a percentage
            'r2': r2
        }


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

    def calculate_mape(self, y_true, y_pred):
        """
        Calculates the Mean Absolute Percentage Error (MAPE).

        :param y_true: Array of true values.
        :param y_pred: Array of predicted values.
        :return: MAPE as a percentage.
        """
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def save_test_results(self, results, model_name, save_dir):
        """
        Saves the test results to a model-specific subdirectory within the save directory.

        :param results: Dictionary containing predictions, true values, and metrics.
        :param model_name: Name of the model (or model file) to label the results.
        :param save_dir: Directory where the results will be saved.
        """
        # Create a model-specific subdirectory within save_dir
        model_dir = os.path.join(save_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)  # Create the directory if it doesn't exist

        # Flatten the arrays to ensure they are 1-dimensional
        true_values_flat = results['true_values'].flatten()  # Flatten the true values
        predictions_flat = results['predictions'].flatten()  # Flatten the predictions

        # Create a DataFrame to store the predictions and true values
        df = pd.DataFrame({
            'True Values (V)': true_values_flat,
            'Predictions (V)': predictions_flat,
            'Difference (mV)': (true_values_flat - predictions_flat) * 1000  # Difference in mV
        })

        # Save the DataFrame as a CSV file in the model-specific directory
        result_file = os.path.join(model_dir, f"{model_name}_test_results.csv")
        df.to_csv(result_file, index=False)

        # Save the metrics separately in the same model-specific directory
        metrics_file = os.path.join(model_dir, f"{model_name}_metrics.txt")
        with open(metrics_file, 'w') as f:
            f.write(f"RMS Error (mV): {results['rms_error_mv']:.2f}\n")
            f.write(f"MAE (mV): {results['mae_mv']:.2f}\n")
            f.write(f"MAPE (%): {results['mape']:.2f}\n")
            f.write(f"R²: {results['r2']:.4f}\n")

        print(f"Results and metrics for model '{model_name}' saved to {model_dir}")


    def run_testing(self, task, model_path, X_test, y_test, save_dir):
        print(f"Entered run_testing for model at {model_path}")
        print(f"Task hyperparameters: {task['model_metadata']}")
        print(f"Test data shapes: X_test={X_test.shape}, y_test={y_test.shape}")

        # Extract hyperparameters from the task
        model_metadata = task["model_metadata"]
        input_size = model_metadata["input_size"]
        hidden_units = model_metadata["hidden_units"]
        num_layers = model_metadata["num_layers"]

        print(f"Instantiating LSTM model with input_size={input_size}, hidden_units={hidden_units}, num_layers={num_layers}")

        # Instantiate the model
        model = LSTMModel(input_size=input_size,
                        hidden_units=hidden_units,
                        num_layers=num_layers,
                        device=self.device)

        # Load the model weights
        try:
            model.load_state_dict(torch.load(model_path))
            model.eval()  # Set the model to evaluation mode
            print("Model loaded and set to evaluation mode")
        except Exception as e:
            print(f"Error loading model from {model_path}: {str(e)}")
            return

        # Pad the test data to ensure the last batch matches the batch size
        X_test_padded, y_test_padded, padding_size = self.pad_data(X_test, y_test, 100)
        print(f" X_padded shape: {X_test_padded.shape}, y_padded shape: {y_test_padded.shape}, padding_size: {padding_size}")

        # Create a DataLoader for testing
        test_dataset = TensorDataset(torch.tensor(X_test_padded, dtype=torch.float32), torch.tensor(y_test_padded, dtype=torch.float32))
        test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

        # Run the testing process
        try:
            results = self.test_model(model, test_loader, padding_size)
            print("Model testing completed")
        except Exception as e:
            print(f"Error during model testing: {str(e)}")
            return

        # Get the model name for saving results
        model_name = os.path.splitext(os.path.basename(model_path))[0]

        # Save the test results
        try:
            self.save_test_results(results, model_name, save_dir)
            print(f"Test results saved for model: {model_name}")
        except Exception as e:
            print(f"Error saving test results: {str(e)}")

        return results