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


    def test_model(self, model, test_data, device, hidden_size, layers):
        """
        Test the model and generate predictions.

        Args:
            model: The LSTM model.
            test_data: Tuple (X_test_tensor, y_test_tensor) containing test dataset.
            device: Device (CPU or GPU) for testing.
            hidden_size: Number of hidden units in the LSTM.
            layers: Number of LSTM layers.

        Returns:
            y_pred: Predicted values.
            y_actual: Actual values from the test set.
        """
        print("Entered test_model")
        model.eval()  # Set model to evaluation mode

        # Extract test inputs & outputs
        X_test_tensor, y_test_tensor = test_data
        print(f"X_test_tensor shape: {X_test_tensor.shape}, y_test_tensor shape: {y_test_tensor.shape}")

        # Initialize hidden states with zeros (No batches, just one full sequence)
        h_s = torch.zeros(layers, X_test_tensor.size(0), hidden_size).to(device)
        h_c = torch.zeros(layers, X_test_tensor.size(0), hidden_size).to(device)

        with torch.no_grad():
            # Forward pass
            y_pred_tensor, _ = model(X_test_tensor, h_s, h_c)

            # Convert predictions & true values to numpy
            y_pred = y_pred_tensor.squeeze().cpu().numpy()
            y_actual = y_test_tensor.cpu().numpy()

        return y_pred, y_actual

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

        print(f"Results and metrics for model saved ")
        self.logger.info(f"Results and metrics for model '{model_name}' saved to {model_dir}")  


    def run_testing(self, task, model_path, X_test, y_test, save_dir, device):
        """
        Runs the testing process for a given model and returns results for UI display.
        """
        print(f"Entered run_testing for model")

        # Extract model metadata
        model_metadata = task["model_metadata"]
        input_size = model_metadata["input_size"]
        hidden_units = model_metadata["hidden_units"]
        num_layers = model_metadata["num_layers"]

        print(f"Instantiating LSTM model with input_size={input_size}, hidden_units={hidden_units}, num_layers={num_layers}")

        # Load model
        model = LSTMModel(input_size=input_size, hidden_units=hidden_units, num_layers=num_layers, device=self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        print(f"Loaded model , going to eval mode")
        model.eval()

        # Convert test data to tensors
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(0).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

        # Run the test_model function
        print(f"expecting values from test_model")
        y_pred, y_actual = self.test_model(model, (X_test_tensor, y_test_tensor), device, hidden_units, num_layers)

        # Compute evaluation metrics
        rms_error_mv = np.sqrt(mean_squared_error(y_actual, y_pred)) * 1000  # Convert to mV
        mae_mv = mean_absolute_error(y_actual, y_pred) * 1000  # Convert to mV
        mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100  # Percentage
        r2 = r2_score(y_actual, y_pred)

        # Save results
        task_id = task.get("task_id", "unknown_task")
        self.save_predictions(y_pred, task_id, save_dir)

        # Return formatted results for UI
        return {
            "predictions": y_pred,
            "true_values": y_actual,
            "rms_error_mv": round(rms_error_mv, 2),
            "mae_mv": round(mae_mv, 2),
            "mape": round(mape, 2),
            "r2": round(r2, 4),
        }

    
    def save_predictions(self, predictions, task_id, save_dir):
        """
        Saves the predictions to a CSV file with the task_id in the filename.

        :param predictions: Array of model predictions.
        :param task_id: Task ID for naming the file.
        :param save_dir: Directory to save the predictions.
        """
        # Create the save directory if it doesn't exist
        predictions_dir = os.path.join(save_dir, "test_results")
        os.makedirs(predictions_dir, exist_ok=True)
        # Save predictions to a CSV file
        predictions_file = os.path.join(save_dir, f"{task_id}_pred.csv")
        pd.DataFrame(predictions, columns=['Predictions (V)']).to_csv(predictions_file, index=False)

        print(f"Predictions saved to {predictions_file}")
        self.logger.info(f"Predictions saved to {predictions_file}")

