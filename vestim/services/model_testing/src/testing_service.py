import torch
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from vestim.services.model_training.src.LSTM_model_service_test import LSTMModel, LSTMModelLN, LSTMModelBN
from vestim.services import normalization_service as norm_svc # Added for normalization
import json # For potentially loading metadata

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

    def test_model(self, model, test_loader, hidden_units, num_layers, target_column_name: str,
                   task_info: dict = None): # Added task_info to access scaler details
        """
        Tests the model on the provided test data and calculates multiple evaluation metrics.
        If normalization was applied during training, attempts to inverse_transform predictions and true values.

        :param model: The loaded model.
        :param test_loader: DataLoader for the test set.
        :param hidden_units: Number of hidden units in the model.
        :param num_layers: Number of layers in the model.
        :param target_column_name: Name of the target column to determine units.
        :param task_info: Dictionary containing task details, potentially including normalization info.
        :return: A dictionary containing the predictions and evaluation metrics.
        """
        all_predictions, y_test_normalized_list = [], [] # y_test is initially normalized if data was normalized
    
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
                    y_test_normalized_list.append(y_true.cpu().numpy())

            # Convert all batch predictions to a single array
            y_pred_normalized = np.concatenate(all_predictions).flatten()
            y_test_normalized = np.concatenate([y.flatten() for y in y_test_normalized_list])

            print(f"DEBUG: Normalized y_pred shape: {y_pred_normalized.shape}, Normalized y_actual shape: {y_test_normalized.shape}")

            # --- Inverse Transform if normalization was applied ---
            y_pred_original_scale = y_pred_normalized.copy()
            y_test_original_scale = y_test_normalized.copy()
            scaler_loaded = None
            normalization_applied_during_training = False
            normalized_columns_during_training = [] # Columns the scaler was fit on

            if task_info:
                # Предполагается, что task_info['job_metadata'] содержит данные из job_metadata.json
                # или task_info напрямую содержит 'normalization_applied', 'scaler_path', 'normalized_columns'
                # Это должно быть установлено на этапе настройки обучения и сохранено с артефактами модели.
                job_metadata = task_info.get('job_metadata', {}) # Example: if job_metadata is nested
                normalization_applied_during_training = job_metadata.get('normalization_applied', False)
                scaler_path_relative = job_metadata.get('scaler_path')
                normalized_columns_during_training = job_metadata.get('normalized_columns', [])
                
                # Example: Construct absolute scaler_path if job_folder is available in task_info
                # This part needs to be robust based on how scaler_path is stored.
                job_folder_for_scaler = task_info.get('job_folder_augmented_from') # Needs to be added to task_info
                
                if normalization_applied_during_training and scaler_path_relative and job_folder_for_scaler and normalized_columns_during_training:
                    # Ensure scaler_path is absolute
                    if not os.path.isabs(scaler_path_relative):
                        scaler_path_absolute = os.path.join(job_folder_for_scaler, scaler_path_relative)
                    else:
                        scaler_path_absolute = scaler_path_relative
                    
                    print(f"Normalization was applied during training. Attempting to load scaler from: {scaler_path_absolute}")
                    scaler_loaded = norm_svc.load_scaler(scaler_path_absolute)
                    if scaler_loaded:
                        print(f"Scaler loaded. Applying inverse transform to target '{target_column_name}' using original feature set: {normalized_columns_during_training}")
                        
                        # Prepare DataFrame for inverse transform (this is crucial)
                        # Create a DataFrame with all columns the scaler was fit on
                        # Fill with dummy values (e.g., 0) then place the target column's data
                        
                        # Inverse transform predictions
                        temp_df_pred = pd.DataFrame(0, index=np.arange(len(y_pred_normalized)), columns=normalized_columns_during_training)
                        if target_column_name in temp_df_pred.columns:
                            temp_df_pred[target_column_name] = y_pred_normalized
                            df_pred_inv = norm_svc.inverse_transform_data(temp_df_pred, scaler_loaded, normalized_columns_during_training)
                            y_pred_original_scale = df_pred_inv[target_column_name].values
                        else:
                            print(f"Warning: Target column '{target_column_name}' not in scaler's known columns. Cannot inverse transform y_pred.")

                        # Inverse transform true values
                        temp_df_test = pd.DataFrame(0, index=np.arange(len(y_test_normalized)), columns=normalized_columns_during_training)
                        if target_column_name in temp_df_test.columns:
                            temp_df_test[target_column_name] = y_test_normalized
                            df_test_inv = norm_svc.inverse_transform_data(temp_df_test, scaler_loaded, normalized_columns_during_training)
                            y_test_original_scale = df_test_inv[target_column_name].values
                        else:
                            print(f"Warning: Target column '{target_column_name}' not in scaler's known columns. Cannot inverse transform y_test.")
                        
                        print("Inverse transform applied to y_pred and y_test.")
                    else:
                        print(f"Warning: Failed to load scaler from {scaler_path_absolute}. Metrics will be on normalized scale.")
                elif normalization_applied_during_training:
                    print("Warning: Normalization was applied during training, but scaler path or related info is missing. Metrics on normalized scale.")
            
            # Use y_test_original_scale and y_pred_original_scale for metrics
            # The existing multiplier logic might need adjustment if inverse transform is successful.
            # If inverse transform happened, data is in original units. 'soc' might be 0-1.
            # The multiplier for 'soc' (100.0) would still apply to convert 0-1 to percentage for display.
            
            # Determine multiplier and unit based on target_column_name
            unit_suffix = ""
            multiplier = 1.0
            unit_display = ""
            
            if "voltage" in target_column_name.lower():
                unit_suffix = "_mv"
                unit_display = "mV"
                multiplier = 1000.0  # Match training task part - V to mV conversion
            elif "soc" in target_column_name.lower():
                unit_suffix = "_percent"
                unit_display = "% SOC"  # Match training task GUI format
                multiplier = 100.0  # Match training task part - 0-1 to percentage conversion
            elif "temperature" in target_column_name.lower() or "temp" in target_column_name.lower():
                unit_suffix = "_degC"
                unit_display = "Deg C"  # Match training task GUI format
                multiplier = 1.0  # Temperature already in the correct scale
            
            # Calculate error metrics with appropriate multipliers
            # Metrics are calculated on the potentially original scale values
            y_for_metrics_actual = y_test_original_scale
            y_for_metrics_pred = y_pred_original_scale

            # If normalization was applied and inverse transform was successful,
            # the data is now in its original scale (e.g. SOC might be 0-1).
            # The multiplier logic for display (e.g. *100 for SOC %) should still apply.
            # If inverse transform failed, y_for_metrics will be the normalized values.

            rms_error_val = np.sqrt(mean_squared_error(y_for_metrics_actual, y_for_metrics_pred)) * multiplier
            mae_val = mean_absolute_error(y_for_metrics_actual, y_for_metrics_pred) * multiplier
            # MAPE calculation should use the original scale if available, otherwise normalized.
            # Ensure y_for_metrics_actual in MAPE denominator is not zero.
            mape_denominator = np.maximum(1e-10, np.abs(y_for_metrics_actual))
            mape = np.mean(np.abs((y_for_metrics_actual - y_for_metrics_pred) / mape_denominator)) * 100
            r2 = r2_score(y_for_metrics_actual, y_for_metrics_pred)

            print(f"Metrics calculated on (potentially) original scale values.")
            print(f"RMS Error: {rms_error_val} {unit_display}, MAE: {mae_val} {unit_display}, MAPE: {mape}%, R²: {r2}")

            results_dict = {
                'predictions': y_pred_original_scale, # Renamed for compatibility, points to original scale
                'true_values': y_test_original_scale, # Renamed for compatibility, points to original scale
                'predictions_normalized': y_pred_normalized if normalization_applied_during_training else None,
                'true_values_normalized': y_test_normalized if normalization_applied_during_training else None,
                f'rms_error{unit_suffix}': rms_error_val,
                f'mae{unit_suffix}': mae_val,
                'mape_percent': mape, # Explicitly state mape is percent
                'r2': r2,
                'unit_display': unit_display,  # Add the unit display string for consistent labeling
                'multiplier': multiplier  # Store the multiplier for potential reuse
            }
            return results_dict
 
    def run_testing(self, task, model_path, test_loader, test_file_path):
        """Runs testing for a given model and test file, returning results without saving."""
        print(f"Running testing for model: {model_path}")

        try:
            # Load the model weights
            model= torch.load(model_path).to(self.device)
            model.eval()  # Set the model to evaluation mode

            # Run the testing process (returns results but does NOT save them)
            target_column = task['data_loader_params'].get('target_column', 'unknown_target')
            # Pass the whole 'task' dictionary as 'task_info'
            results = self.test_model(
                model=model,
                test_loader=test_loader,
                hidden_units=task['model_metadata']["hidden_units"],
                num_layers=task['model_metadata']["num_layers"],
                target_column_name=target_column,
                task_info=task # Pass the entire task dictionary
            )

            print(f"Model testing completed for file: {test_file_path}")
            return results  # Return results without saving

        except Exception as e:
            print(f"Error testing model {model_path}: {str(e)}")
            return None

