import torch
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class VEstimTestingService:
    def __init__(self, device='cpu'):
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

    def test_model(self, model, X_test, y_test):
        """
        Tests the model on the provided test data and calculates multiple evaluation metrics in millivolts (mV).

        :param model: The loaded model.
        :param X_test: Input sequences for testing.
        :param y_test: True output values.
        :return: A dictionary containing the predictions and evaluation metrics.
        """
        with torch.no_grad():
            # Convert test data to tensors and move to the device
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
            # y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(self.device)

            # Initialize hidden states (if your model requires them)
            h_s = torch.zeros(model.num_layers, X_test_tensor.size(0), model.hidden_units).to(self.device)
            h_c = torch.zeros(model.num_layers, X_test_tensor.size(0), model.hidden_units).to(self.device)

            # Generate predictions
            y_pred_tensor, _ = model(X_test_tensor, h_s, h_c)
            y_pred_tensor = y_pred_tensor.squeeze(-1)  # Ensure output shape matches

            # Convert predictions to numpy for easier evaluation
            y_pred = y_pred_tensor.cpu().numpy()

            # Compute evaluation metrics and convert to millivolts (mV)
            rms_error = np.sqrt(mean_squared_error(y_test, y_pred)) * 1000  # Convert to mV
            mae = mean_absolute_error(y_test, y_pred) * 1000  # Convert to mV
            mape = self.calculate_mape(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            return {
                'predictions': y_pred,
                'true_values': y_test,
                'rms_error_mv': rms_error,  # Error in mV
                'mae_mv': mae,  # Error in mV
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
        Saves the test results to a specified directory.

        :param results: Dictionary containing predictions, true values, and metrics.
        :param model_name: Name of the model (or model file) to label the results.
        :param save_dir: Directory where the results will be saved.
        """
        # Create a DataFrame to store the predictions and true values
        df = pd.DataFrame({
            'True Values (V)': results['true_values'],
            'Predictions (V)': results['predictions'],
            'Difference (mV)': (results['true_values'] - results['predictions']) * 1000  # Difference in mV
        })

        # Save the DataFrame as a CSV file
        result_file = os.path.join(save_dir, f"{model_name}_test_results.csv")
        df.to_csv(result_file, index=False)

        # Save the metrics separately
        metrics_file = os.path.join(save_dir, f"{model_name}_metrics.txt")
        with open(metrics_file, 'w') as f:
            f.write(f"RMS Error (mV): {results['rms_error_mv']:.2f}\n")
            f.write(f"MAE (mV): {results['mae_mv']:.2f}\n")
            f.write(f"MAPE (%): {results['mape']:.2f}\n")
            f.write(f"R²: {results['r2']:.4f}\n")

    def run_testing(self, model_path, X_test, y_test, save_dir):
        """
        Orchestrates the testing process by loading the model, running tests, and saving the results.

        :param model_path: Path to the model .pth file.
        :param X_test: Input sequences for testing.
        :param y_test: True output values.
        :param save_dir: Directory where the results will be saved.
        :return: A dictionary containing predictions and evaluation metrics.
        """
        model = self.load_model(model_path)
        results = self.test_model(model, X_test, y_test)
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        self.save_test_results(results, model_name, save_dir)
        return results
