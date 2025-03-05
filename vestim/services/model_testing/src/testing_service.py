import torch
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from vestim.services.model_training.src.LSTM_model_service_test import LSTMModel, LSTMModelLN, LSTMModelBN

class VEstimTestingService:
    def __init__(self, device='cpu'):
        print("Initializing VEstimTestingService...")
        """
        Initialize the TestingService with the specified device.

        :param device: Device to run the model on ('cpu' or 'cuda').
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    def test_model(self, model, test_loader, hidden_units, num_layers):
        """
        Tests the model on the provided test data and calculates multiple evaluation metrics in millivolts (mV).

        :param model: The loaded model.
        :param X_test: Input sequences for testing.
        :param y_test: True output values.
        :return: A dictionary containing the predictions and evaluation metrics.
        """
        all_predictions, y_test = [], []
    
        with torch.no_grad():
            # Initialize hidden states for the first sample (removing batch dimension)
            h_s = torch.zeros(num_layers, 1, hidden_units).to(self.device)
            h_c = torch.zeros(num_layers, 1, hidden_units).to(self.device)

            # Loop over the test set one sequence at a time
            for X_batch, y_batch in test_loader:
                # Since test_loader is batched, process each sequence in the batch individually
                for i in range(X_batch.size(0)):
                    # Extract a single sequence (shape: [1, lookback, features])
                    x_seq = X_batch[i].unsqueeze(0).to(self.device)
                    y_true = y_batch[i].unsqueeze(0).to(self.device)

                    # Forward pass with current hidden states
                    y_out, (h_s, h_c) = model(x_seq, h_s, h_c)

                    # Detach hidden states to avoid accumulation of gradients
                    h_s, h_c = h_s.detach(), h_c.detach()

                    # Store the last timestep prediction and corresponding true value
                    all_predictions.append(y_out[:, -1].cpu().numpy())
                    y_test.append(y_true.cpu().numpy())

            # Convert all batch predictions to a single array
            y_pred = np.concatenate(all_predictions).flatten()  # Combine all stored predictions
            y_test = np.concatenate([y.flatten() for y in y_test])  # Combine all stored true values

            # Debug prints
            print(f"DEBUG: Trimmed y_pred shape: {y_pred.shape}, y_actual shape: {y_test.shape}")

            # Compute evaluation metrics
            rms_error = np.sqrt(mean_squared_error(y_test, y_pred)) * 1000  # Convert to mV
            mae = mean_absolute_error(y_test, y_pred) * 1000  # Convert to mV
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # MAPE in percentage
            r2 = r2_score(y_test, y_pred)

            print(f"RMS Error: {rms_error}, MAE: {mae}, MAPE: {mape}, R²: {r2}")

            return {
                'predictions': y_pred,
                'true_values': y_test,
                'rms_error_mv': rms_error,  # Error in mV
                'mae_mv': mae,  # Error in mV
                'mape': mape,  # MAPE remains a percentage
                'r2': r2
            }
 
    def run_testing(self, task, model_path, test_loader, test_file_path):
        """Runs testing for a given model and test file, returning results without saving."""
        print(f"Running testing for model: {model_path}")

        try:
            # Load the model weights
            model= torch.load(model_path).to(self.device)
            model.eval()  # Set the model to evaluation mode

            # Run the testing process (returns results but does NOT save them)
            results = self.test_model(
                model,
                test_loader,
                task['model_metadata']["hidden_units"],
                task['model_metadata']["num_layers"]
            )

            print(f"Model testing completed for file: {test_file_path}")
            return results  # Return results without saving

        except Exception as e:
            print(f"Error testing model {model_path}: {str(e)}")
            return None

