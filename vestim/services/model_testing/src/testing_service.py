import torch
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from vestim.services.model_training.src.LSTM_model_service_test import LSTMModel, LSTMModelLN, LSTMModelBN # Keep imports for type hinting if model object is used
from vestim.services.model_training.src.GRU_model import GRUModel # Add GRU model import
from vestim.services.model_training.src.FNN_model import FNNModel # Add FNN model import
from vestim.services import normalization_service as norm_svc # Added for normalization
import json # For potentially loading metadata

# Removed torch.serialization.add_safe_globals as we are reverting to weights_only=False

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
        model = torch.load(model_path) # Reverted to weights_only=False (default)
        model.to(self.device)
        model.eval()  # Set the model to evaluation mode
        return model

    def test_model(self, model, test_loader, hidden_units, num_layers, target_column_name: str,
                   task_info: dict = None, model_type: str = "LSTM"): # Added model_type parameter
        """
        Tests the model on the provided test data and calculates multiple evaluation metrics.
        If normalization was applied during training, attempts to inverse_transform predictions and true values.

        :param model: The loaded model.
        :param test_loader: DataLoader for the test set.
        :param hidden_units: Number of hidden units in the model (not used for FNN).
        :param num_layers: Number of layers in the model (not used for FNN).
        :param target_column_name: Name of the target column to determine units.
        :param task_info: Dictionary containing task details, potentially including normalization info.
        :param model_type: Type of model ("LSTM", "GRU", or "FNN")
        :return: A dictionary containing the predictions and evaluation metrics.
        """
        all_predictions, y_test_normalized_list = [], [] # y_test is initially normalized if data was normalized
    
        with torch.no_grad():
            if model_type == "FNN":
                # FNN testing: no hidden states, process all samples directly
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    # Forward pass (no hidden states for FNN)
                    y_out = model(X_batch)
                    
                    # Store predictions and true values
                    all_predictions.append(y_out.cpu().numpy())
                    y_test_normalized_list.append(y_batch.cpu().numpy())
            else:
                # RNN testing: requires hidden states (LSTM/GRU)
                h_s = torch.zeros(num_layers, 1, hidden_units).to(self.device)
                if model_type == "LSTM":
                    h_c = torch.zeros(num_layers, 1, hidden_units).to(self.device)
                else:  # GRU
                    h_c = None

                # Loop over the test set one sequence at a time
                for X_batch, y_batch in test_loader:
                    # Since test_loader is batched, process each sequence in the batch individually
                    for i in range(X_batch.size(0)):
                        # Extract a single sequence (shape: [1, lookback, features])
                        x_seq = X_batch[i].unsqueeze(0).to(self.device)
                        y_true = y_batch[i].unsqueeze(0).to(self.device)

                        # Forward pass with current hidden states
                        if model_type == "LSTM":
                            y_out, (h_s, h_c) = model(x_seq, h_s, h_c)
                            # Detach hidden states to avoid accumulation of gradients
                            h_s, h_c = h_s.detach(), h_c.detach()
                        elif model_type == "GRU":
                            y_out, h_s = model(x_seq, h_s)
                            # Detach hidden state to avoid accumulation of gradients
                            h_s = h_s.detach()
                        else:
                            raise ValueError(f"Unsupported model_type in test_model: {model_type}")

                        # Store the last timestep prediction and corresponding true value
                        all_predictions.append(y_out[:, -1].cpu().numpy())
                        y_test_normalized_list.append(y_true.cpu().numpy())

            # Convert all batch predictions to a single array
            if model_type == "FNN":
                # For FNN, predictions come in batches, so concatenate them directly
                y_pred_normalized = np.concatenate(all_predictions).flatten()
                y_test_normalized = np.concatenate(y_test_normalized_list).flatten()
            else:
                # For RNN, predictions are single values from last timestep
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
                # 'job_metadata' should now be directly available in task_info and contain the parsed JSON content.
                job_meta = task_info.get('job_metadata', {})
                normalization_applied_during_training = job_meta.get('normalization_applied', False)
                scaler_path_relative = job_meta.get('scaler_path')
                normalized_columns_during_training = job_meta.get('normalized_columns', [])
                
                # job_folder_augmented_from is now directly in task_info
                job_folder_for_scaler = task_info.get('job_folder_augmented_from')
                
                if normalization_applied_during_training and scaler_path_relative and job_folder_for_scaler and normalized_columns_during_training:
                    # Ensure scaler_path is absolute
                    if not os.path.isabs(scaler_path_relative):
                        scaler_path_absolute = os.path.join(job_folder_for_scaler, scaler_path_relative)
                    else:
                        scaler_path_absolute = scaler_path_relative
                    
                    print(f"Normalization was applied during training. Attempting to load scaler from: {scaler_path_absolute}")
                    scaler_loaded = norm_svc.load_scaler(scaler_path_absolute)
                    if scaler_loaded:
                        print(f"Scaler loaded. Applying inverse transform to target '{target_column_name}' using standardized method")
                        
                        # Use the standardized denormalization function instead of DataFrame method
                        y_pred_original_scale = norm_svc.inverse_transform_single_column(
                            y_pred_normalized, scaler_loaded, target_column_name, normalized_columns_during_training
                        )
                        y_test_original_scale = norm_svc.inverse_transform_single_column(
                            y_test_normalized, scaler_loaded, target_column_name, normalized_columns_during_training
                        )
                        
                        print("Standardized inverse transform applied to y_pred and y_test.")
                        print(f"Sample denormalized values - Pred: {y_pred_original_scale[:3] if len(y_pred_original_scale) > 3 else y_pred_original_scale}")
                        print(f"Sample denormalized values - True: {y_test_original_scale[:3] if len(y_test_original_scale) > 3 else y_test_original_scale}")
                        
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
                unit_display = "SOC"  # Match training task GUI format
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

            # Calculate error for the "Error (% SOC)" column
            # y_for_metrics_actual is y_test_original_scale (e.g., SOC 0-1)
            # y_for_metrics_pred is y_pred_original_scale (e.g., SOC 0-1)
            # multiplier is 100.0 for SOC, so error is in percentage points
            error_percent_soc_values = (y_for_metrics_actual - y_for_metrics_pred) * multiplier
            
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
                'multiplier': multiplier,  # Store the multiplier for potential reuse
                'error_percent_soc_values': error_percent_soc_values # New key for direct error values
            }
            return results_dict
 
    def run_testing(self, task, model_path, test_loader, test_file_path):
        """Runs testing for a given model and test file, returning results without saving."""
        print(f"Running testing for model: {model_path}")

        try:
            # Load the model weights
            state_dict = torch.load(model_path)
            model_type = task.get('model_metadata', {}).get('model_type', 'LSTM')
            hyperparams = task.get('hyperparams', {})
            
            if model_type == 'LSTM':
                model = LSTMModel(
                    input_size=hyperparams['INPUT_SIZE'],
                    hidden_units=hyperparams['HIDDEN_UNITS'],
                    num_layers=hyperparams['LAYERS'],
                    device=self.device,
                    dropout_prob=hyperparams.get('DROPOUT_PROB', 0.0),
                    apply_clipped_relu=hyperparams.get('normalization_applied', False)
                )
            elif model_type == 'GRU':
                model = GRUModel(
                    input_size=hyperparams['INPUT_SIZE'],
                    hidden_units=hyperparams['HIDDEN_UNITS'],
                    num_layers=hyperparams['LAYERS'],
                    device=self.device,
                    dropout_prob=hyperparams.get('DROPOUT_PROB', 0.0),
                    apply_clipped_relu=hyperparams.get('normalization_applied', False)
                )
            elif model_type == 'FNN':
                model = FNNModel(
                    input_size=hyperparams['INPUT_SIZE'],
                    output_size=hyperparams['OUTPUT_SIZE'],
                    hidden_layer_sizes=hyperparams['HIDDEN_LAYER_SIZES'],
                    dropout_prob=hyperparams.get('DROPOUT_PROB', 0.0),
                    apply_clipped_relu=hyperparams.get('normalization_applied', False)
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()  # Set the model to evaluation mode

            # Run the testing process (returns results but does NOT save them)
            target_column = task['data_loader_params'].get('target_column', 'unknown_target')
            model_type = task.get('model_metadata', {}).get('model_type', 'LSTM')  # Get model type from task
            
            # Get model-specific parameters safely
            if model_type in ['LSTM', 'GRU']:
                hidden_units = task['model_metadata']["hidden_units"]
                num_layers = task['model_metadata']["num_layers"]
            else:  # FNN or other models
                hidden_units = 0  # Not used for FNN
                num_layers = 0   # Not used for FNN
                
            # Pass the whole 'task' dictionary as 'task_info'
            results = self.test_model(
                model=model,
                test_loader=test_loader,
                hidden_units=hidden_units,
                num_layers=num_layers,
                target_column_name=target_column,
                task_info=task, # Pass the entire task dictionary
                model_type=model_type  # Pass model type
            )

            print(f"Model testing completed for file: {test_file_path}")
            return results  # Return results without saving

        except Exception as e:
            print(f"Error testing model {model_path}: {str(e)}")
            return None

