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
from vestim.services.model_training.src.LSTM_model_service import LSTMModel
import logging

class VEstimTestingService:
    def __init__(self, device='cpu'):
        self.logger = logging.getLogger(__name__)
        print("Initializing VEstimTestingService...")
        """
        Initialize the TestingService with the specified device.

        :param device: Device to run the model on ('cpu' or 'cuda').
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    def test_model(self, model, test_loader, h_s, h_c, device, padding_size):
        model.eval()  # Ensure the model is in evaluation mode
        total_rmse = 0
        total_mae = 0
        total_samples = 0
        all_predictions = []
        all_true_values = []

        with torch.no_grad():
            for batch_idx, (X_batch, y_batch) in enumerate(test_loader):
                batch_size = X_batch.size(0)
                print(f"Batch {batch_idx + 1}: X_batch shape: {X_batch.shape}, y_batch shape: {y_batch.shape}")

                # Forward pass  
                X_batch, h_s, h_c = X_batch.to(device), h_s.to(device), h_c.to(device)
                assert X_batch.device == h_s.device == h_c.device, \
                    f"Device mismatch: X_batch {X_batch.device}, h_s {h_s.device}, h_c {h_c.device}"
                y_pred_tensor, (h_s, h_c) = model(X_batch, h_s, h_c)
                print(f"Batch {batch_idx + 1}: y_pred_tensor shape: {y_pred_tensor.shape}")

                # Collect predictions and true values
                all_predictions.append(y_pred_tensor.cpu().numpy())
                all_true_values.append(y_batch.cpu().numpy())

                # Compute errors for each batch and accumulate
                y_pred = y_pred_tensor.cpu().numpy().flatten()
                y_true = y_batch.cpu().numpy()
   
                batch_rmse = np.sqrt(mean_squared_error(y_true, y_pred)) * 1000  # Convert to mV
                batch_mae = mean_absolute_error(y_true, y_pred) * 1000  # Convert to mV
                total_rmse += batch_rmse * batch_size
                total_mae += batch_mae * batch_size
                total_samples += batch_size

                print(f"Batch {batch_idx + 1}: RMSE: {batch_rmse} mV, MAE: {batch_mae} mV")

                # Free up GPU memory
                del X_batch, y_batch, y_pred_tensor
                torch.cuda.empty_cache()
                print(f"Batch {batch_idx + 1}: Freed memory.")

        # Final average metrics
        avg_rmse = total_rmse / total_samples
        avg_mae = total_mae / total_samples

        # Convert to flat arrays for saving predictions and true values
        y_pred_final = np.concatenate(all_predictions, axis=0).flatten()
        y_true_final = np.concatenate(all_true_values, axis=0)
        # Remove the padded data from the results if padding was applied
        if padding_size > 0:
            y_pred_final = y_pred_final[:-padding_size]
            y_true_final = y_true_final[:-padding_size]

        # MAPE and R2 calculation
        mape = self.calculate_mape(y_true_final, y_pred_final)
        r2 = r2_score(y_true_final, y_pred_final)

        print(f"Final Metrics - RMS Error: {avg_rmse} mV, MAE: {avg_mae} mV, MAPE: {mape}%, R²: {r2}")
        self.logger.info(f"Final Test Metrics for model {model} - RMS Error: {avg_rmse} mV, MAE: {avg_mae} mV, MAPE: {mape}%, R²: {r2}")

        return {
            'predictions': y_pred_final,
            'true_values': y_true_final,
            'rms_error_mv': avg_rmse,  # Error in mV
            'mae_mv': avg_mae,  # Error in mV
            'mape': mape,  # MAPE remains a percentage
            'r2': r2
        }

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
        model_dir = os.path.join(save_dir, "test_results")
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
        self.logger.info(f"Results and metrics for model '{model_name}' saved to {model_dir}")  


    def run_testing(self, task, model_path, X_test, y_test, save_dir, device):
        """
        Runs the testing process for a given task and model, and saves the results.

        :param task: Task containing model metadata and hyperparameters.
        :param model_path: Path to the model .pth file.
        :param X_test: Test input data (features).
        :param y_test: Test output data (targets).
        :param save_dir: Directory to save the test results.
        """
        print(f"Entered run_testing for model at {model_path}")
        print(f"Task hyperparameters: {task['model_metadata']}")
        print(f"Test data shapes: X_test={X_test.shape}, y_test={y_test.shape}")

        # Extract hyperparameters from the task
        model_metadata = task["model_metadata"]
        input_size = model_metadata["input_size"]
        hidden_units = model_metadata["hidden_units"]
        num_layers = model_metadata["num_layers"]
        batch_size = task['hyperparams']['BATCH_SIZE']

        print(f"Instantiating LSTM model with input_size={input_size}, hidden_units={hidden_units}, num_layers={num_layers}")

        # Instantiate the model
        model = LSTMModel(input_size=input_size,
                        hidden_units=hidden_units,
                        num_layers=num_layers,
                        device=self.device)

        # Load the model weights and remove pruning if needed
        try:
            # Load the saved state dict
            model_state_dict = torch.load(model_path, map_location=self.device)

            # Adjust the state_dict to handle pruning-related keys
            new_state_dict = {}
            for key, value in model_state_dict.items():
                if "_orig" in key:
                    # If the original weights exist, remove "_orig" and use the clean key
                    new_key = key.replace("_orig", "")
                    new_state_dict[new_key] = value
                elif "_mask" not in key:
                    # If it's not a mask, include it in the new state_dict
                    new_state_dict[key] = value

            # Load the adjusted state dict into the model
            model.load_state_dict(new_state_dict)

            # Set the model to evaluation mode
            model.eval()
            print("Model loaded, pruning keys handled, and set to evaluation mode")
        except Exception as e:
            print(f"Error loading model from {model_path}: {str(e)}")
            return
        # Pad the test data to ensure the last batch matches the batch size
        X_test_padded, y_test_padded, padding_size = self.pad_data(X_test, y_test, batch_size)
        print(f"X_padded shape: {X_test_padded.shape}, y_padded shape: {y_test_padded.shape}, padding_size: {padding_size}")
        self.logger.info(f"X_padded shape: {X_test_padded.shape}, y_padded shape: {y_test_padded.shape}, padding_size: {padding_size}")

        test_dataset = TensorDataset(torch.tensor(X_test_padded, dtype=torch.float32), torch.tensor(y_test_padded, dtype=torch.float32))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Initialize hidden states and move them to device
        h_s = torch.zeros(num_layers, batch_size, hidden_units).to(device)  # Shape: (num_layers, batch_size, hidden_units)
        h_c = torch.zeros(num_layers, batch_size, hidden_units).to(device)  # Shape: (num_layers, batch_size, hidden_units)

        # Run the testing process
        try:
            results = self.test_model(model, test_loader, h_s, h_c, device, padding_size)
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
        X_padding = np.zeros((padding_size, X.shape[1], X.shape[2]))
        y_padding = np.zeros((padding_size,))

        # Append the padding to X and y
        X_padded = np.vstack([X, X_padding])
        y_padded = np.concatenate([y, y_padding])

        return X_padded, y_padded, padding_size
