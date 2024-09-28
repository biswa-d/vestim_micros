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
        Loads a pruned model from the specified .pth file and removes pruning masks.
        
        :param model_path: Path to the model .pth file.
        :return: The loaded model with pruning removed.
        """
        model_state_dict = torch.load(model_path)
        
        # Remove pruning-related parameters from the state dict
        new_state_dict = {}
        for key, value in model_state_dict.items():
            if "_orig" in key:
                new_key = key.replace("_orig", "")
                new_state_dict[new_key] = value
            elif "_mask" not in key:
                new_state_dict[key] = value
        
        # Instantiate the model based on task hyperparameters
        model_metadata = task["model_metadata"]
        input_size = model_metadata["input_size"]
        hidden_units = model_metadata["hidden_units"]
        num_layers = model_metadata["num_layers"]
        
        # Instantiate the model
        model = LSTMModel(input_size=input_size,
                        hidden_units=hidden_units,
                        num_layers=num_layers,
                        device=self.device)
        
        # Load the new state dict into the model
        model.load_state_dict(new_state_dict)
        model.to(self.device)
        model.eval()  # Set the model to evaluation mode
        return model


    def test_model(self, model, test_loader):
        """
        Tests the model on the provided DataLoader and calculates multiple evaluation metrics in millivolts (mV).
        """
        model.eval()  # Ensure model is in evaluation mode
        all_predictions = []
        all_true_values = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                batch_size = X_batch.size(0)

                # Initialize hidden and cell states based on the batch size
                h_s = torch.zeros(model.num_layers, batch_size, model.hidden_units).to(self.device)
                h_c = torch.zeros(model.num_layers, batch_size, model.hidden_units).to(self.device)

                # Forward pass
                y_pred_tensor, _ = model(X_batch.to(self.device), h_s, h_c)
                y_pred_tensor = y_pred_tensor.squeeze(-1)

                # Collect predictions and true values
                all_predictions.append(y_pred_tensor.cpu().numpy())
                all_true_values.append(y_batch.cpu().numpy())

        # Convert to flat arrays for evaluation
        y_pred = np.concatenate(all_predictions, axis=0)
        y_true = np.concatenate(all_true_values, axis=0)

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

        # Create a DataFrame to store the predictions and true values
        df = pd.DataFrame({
            'True Values (V)': results['true_values'],
            'Predictions (V)': results['predictions'],
            'Difference (mV)': (results['true_values'] - results['predictions']) * 1000  # Difference in mV
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

        # Load the model with pruning handled
        model = self.load_model(model_path)

        # Create a DataLoader for testing
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
        test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)  # Batch size can be tuned

        # Run the testing process
        try:
            results = self.test_model(model, test_loader)
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
