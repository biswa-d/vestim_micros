import torch
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class ContinuousTestingService:
    """
    New continuous testing service that processes one sample at a time
    and maintains hidden states across all test files for a single model.
    """
    
    def __init__(self, device='cpu'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hidden_states = None
        self.model_instance = None
        self.scaler = None
        
    def reset_for_new_model(self):
        """Reset all states for testing a new model."""
        self.hidden_states = None
        self.model_instance = None
        self.scaler = None
        print("Reset continuous testing service for new model")
    
    def run_continuous_testing(self, task, model_path, test_file_path, is_first_file=False, warmup_samples=400):
        """
        Continuous testing that processes one sample at a time as single timesteps.
        No sequences or lookback buffers - uses LSTM's natural recurrent memory.
        
        Args:
            task: Task configuration dictionary
            model_path: Path to the trained model
            test_file_path: Path to the current test file
            is_first_file: Whether this is the first file in the test sequence
            warmup_samples: Number of samples to use for warmup (skip predictions)
        
        Returns:
            Dictionary containing test results and metrics
        """
        print(f"Running continuous testing for: {test_file_path}")
        
        try:
            # Load model only once (on first file)
            if self.model_instance is None or is_first_file:
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Handle different model save formats
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    # Model saved with state_dict
                    self.model_instance = self._create_model_instance(task)
                    self.model_instance.load_state_dict(checkpoint['state_dict'])
                else:
                    # Model saved as complete object
                    self.model_instance = checkpoint
                
                self.model_instance.to(self.device)
                self.model_instance.eval()
                print(f"Model loaded: {model_path}")
            
            # Initialize hidden states only once for the first file (only for RNN models)
            if is_first_file or self.hidden_states is None:
                model_metadata = task.get('model_metadata', {})
                model_type = model_metadata.get('model_type', 'LSTM')
                
                if model_type == 'FNN':
                    # FNN models don't use hidden states
                    self.hidden_states = {
                        'model_type': 'FNN'
                    }
                    print(f"No hidden states needed for FNN model")
                elif model_type == 'GRU':
                    # GRU only has hidden state (h_s), no cell state (h_c)
                    self.hidden_states = {
                        'h_s': torch.zeros(self.model_instance.num_layers, 1, self.model_instance.hidden_units).to(self.device),
                        'model_type': 'GRU'
                    }
                    print(f"Hidden states initialized for continuous testing ({model_type})")
                else:
                    # LSTM has both hidden state (h_s) and cell state (h_c)
                    self.hidden_states = {
                        'h_s': torch.zeros(self.model_instance.num_layers, 1, self.model_instance.hidden_units).to(self.device),
                        'h_c': torch.zeros(self.model_instance.num_layers, 1, self.model_instance.hidden_units).to(self.device),
                        'model_type': 'LSTM'
                    }
                    print(f"Hidden states initialized for continuous testing ({model_type})")
            
            # Load scaler only once
            if self.scaler is None:
                self.scaler = self._load_scaler_if_available(task)
            
            # Get task parameters
            data_loader_params = task.get('data_loader_params', {})
            feature_cols = data_loader_params.get('feature_columns')
            target_col = data_loader_params.get('target_column')
            
            if not feature_cols or not target_col:
                raise ValueError(f"Missing feature_cols or target_col in task for {test_file_path}")
            
            # Load test data
            df = pd.read_csv(test_file_path)
            original_len = len(df)
            print(f"Processing {original_len} samples from {test_file_path}")
            
            # Check if normalization was applied during training
            job_meta = task.get('job_metadata', {})
            normalization_applied = job_meta.get('normalization_applied', False)
            
            # Fallback: check hyperparams if not in job_metadata
            if not normalization_applied:
                hyperparams = task.get('hyperparams', {})
                normalization_applied = hyperparams.get('NORMALIZATION_APPLIED', False)
                if normalization_applied:
                    print("Found normalization flag in hyperparams")
            
            print(f"Normalization applied during training: {normalization_applied}")
            
            # Load processed test data - processed files contain all data augmentations
            # including normalization if it was applied during data augmentation
            print(f"Loading processed test file: {test_file_path}")
            all_required_cols = feature_cols + [target_col]
            df_scaled = df[all_required_cols].select_dtypes(include=[np.number])
            
            # Debug: Check what the processed data looks like
            print(f"DEBUG - Processed test data analysis:")
            print(f"  Available columns: {df_scaled.columns.tolist()}")
            print(f"  Target column '{target_col}' sample values: {df_scaled[target_col].iloc[:5].values}")
            print(f"  Target column range: [{df_scaled[target_col].min():.6f}, {df_scaled[target_col].max():.6f}]")
            
            # Debug: Check if scaler is available and what type it is
            if self.scaler is not None:
                print(f"  Scaler type: {type(self.scaler)}")
                print(f"  Scaler feature names: {list(self.scaler.feature_names_in_) if hasattr(self.scaler, 'feature_names_in_') else 'None'}")
                if hasattr(self.scaler, 'data_min_') and hasattr(self.scaler, 'data_max_'):
                    print(f"  Scaler data_min_: {self.scaler.data_min_}")
                    print(f"  Scaler data_max_: {self.scaler.data_max_}")
                if hasattr(self.scaler, 'feature_range'):
                    print(f"  Scaler feature_range: {self.scaler.feature_range}")
            else:
                print(f"  Scaler: None (not loaded)")
            
            if normalization_applied:
                print("Using processed data (already normalized during data augmentation)")
            else:
                print("Using processed data (no normalization was applied during data augmentation)")
            
            # Always handle warmup for first file (regardless of normalization)
            if is_first_file:
                print(f"Adding {warmup_samples} warmup samples for first file")
                first_row = df_scaled.iloc[0]
                df_warmup = pd.DataFrame([first_row] * warmup_samples, columns=df_scaled.columns)
                df_test_full = pd.concat([df_warmup, df_scaled], ignore_index=True)
            else:
                df_test_full = df_scaled.copy()
            
            # Convert to tensor - each sample as a single timestep
            X_all = torch.tensor(
                df_test_full[feature_cols].values.astype(np.float32)
            ).view(-1, 1, len(feature_cols)).to(self.device)  # Shape: (samples, 1, features)
            
            # Allocate prediction buffer for actual test length
            y_preds = torch.zeros(original_len, dtype=torch.float32).to(self.device)
            
            # Process each sample one at a time
            with torch.no_grad():
                for t in range(len(df_test_full)):
                    x_t = X_all[t].unsqueeze(0)  # Shape: (1, 1, features) - single timestep
                    
                    # Forward pass based on model type
                    if self.hidden_states['model_type'] == 'FNN':
                        # FNN forward pass - reshape to (batch_size, features)
                        x_flat = x_t.view(1, -1)  # Shape: (1, features)
                        y_pred = self.model_instance(x_flat)
                    elif self.hidden_states['model_type'] == 'GRU':
                        # GRU forward pass with persistent hidden states
                        output, h_s = self.model_instance.gru(x_t, self.hidden_states['h_s'])
                        # Get final prediction
                        y_pred = self.model_instance.fc(output[:, -1, :])
                        # Update hidden state for next sample
                        self.hidden_states['h_s'] = h_s.detach()
                    else:
                        # LSTM forward pass with persistent hidden states
                        output, (h_s, h_c) = self.model_instance.lstm(x_t, (self.hidden_states['h_s'], self.hidden_states['h_c']))
                        # Get final prediction
                        y_pred = self.model_instance.linear(output[:, -1, :])
                        # Update hidden states for next sample
                        self.hidden_states['h_s'] = h_s.detach()
                        self.hidden_states['h_c'] = h_c.detach()
                    
                    # Skip predictions during warmup for first file
                    if is_first_file and t < warmup_samples:
                        continue
                    
                    # Store prediction
                    y_index = t if not is_first_file else t - warmup_samples
                    if y_index < original_len:
                        y_preds[y_index] = y_pred.squeeze()
            
            # Get true values and convert predictions to numpy
            y_true = df_scaled[target_col].iloc[:original_len].values.astype(np.float32)
            y_pred_arr = y_preds.cpu().numpy()
            
            print(f"Generated {len(y_pred_arr)} predictions (after warmup: {warmup_samples if is_first_file else 0})")
            print(f"Data processing mode: processed file (from data augmentation pipeline)")
            print(f"Normalization was applied during training: {normalization_applied}")
            print(f"Data shapes - Predictions: {y_pred_arr.shape}, True values: {y_true.shape}")
            print(f"Sample values - Pred: {y_pred_arr[:3]}, True: {y_true[:3]}")
            
            # Ensure proper alignment
            min_len = min(len(y_pred_arr), len(y_true))
            if len(y_pred_arr) != len(y_true):
                print(f"Warning: Length mismatch detected. Truncating to {min_len} samples.")
                y_pred_arr = y_pred_arr[:min_len]
                y_true = y_true[:min_len]
            
            # Check if normalization was applied during training to determine if denormalization is needed
            job_meta = task.get('job_metadata', {})
            normalization_applied = job_meta.get('normalization_applied', False)
            
            # Debug: Show raw values before any denormalization
            print(f"DEBUG - Before denormalization:")
            print(f"  Raw predictions (first 5): {y_pred_arr[:5]}")
            print(f"  Raw true values (first 5): {y_true[:5]}")
            print(f"  Prediction range: [{y_pred_arr.min():.6f}, {y_pred_arr.max():.6f}]")
            print(f"  True values range: [{y_true.min():.6f}, {y_true.max():.6f}]")
            
            # Denormalize only if normalization was applied during training AND scaler is available
            if normalization_applied and self.scaler is not None:
                print("Normalization was applied during training - denormalizing predictions and true values")
                y_pred_denorm, y_true_denorm = self._denormalize_values(y_pred_arr, y_true, target_col)
            else:
                if not normalization_applied:
                    print("No normalization was applied during training - using values as-is")
                else:
                    print("Normalization was applied but no scaler available - using scaled values")
                y_pred_denorm, y_true_denorm = y_pred_arr, y_true
            
            # Debug: Show final values after denormalization
            print(f"DEBUG - After denormalization (final values for metrics):")
            print(f"  Final predictions (first 5): {y_pred_denorm[:5]}")
            print(f"  Final true values (first 5): {y_true_denorm[:5]}")
            print(f"  Final prediction range: [{y_pred_denorm.min():.6f}, {y_pred_denorm.max():.6f}]")
            print(f"  Final true values range: [{y_true_denorm.min():.6f}, {y_true_denorm.max():.6f}]")
            print(f"  Values are in original physical units: {not normalization_applied or (normalization_applied and self.scaler is not None)}")
            
            # Calculate metrics and format results
            results = self._calculate_metrics_and_format_results(
                y_pred_denorm, y_true_denorm, target_col, task
            )
            
            print(f"Continuous testing completed for: {test_file_path}")
            return results
            
        except Exception as e:
            print(f"Error in continuous testing for {test_file_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _load_scaler_if_available(self, task):
        """Load scaler if normalization was applied during training."""
        try:
            job_meta = task.get('job_metadata', {})
            normalization_applied = job_meta.get('normalization_applied', False)
            
            if not normalization_applied:
                print("No normalization was applied during training")
                return None
            
            scaler_path_relative = job_meta.get('scaler_path')
            job_folder = task.get('job_folder_augmented_from')
            
            if not scaler_path_relative or not job_folder:
                print("Scaler path or job folder not available")
                return None
            
            scaler_path = os.path.join(job_folder, scaler_path_relative)
            
            if not os.path.exists(scaler_path):
                print(f"Scaler file not found at {scaler_path}")
                return None
            
            scaler = joblib.load(scaler_path)
            print(f"Loaded scaler from {scaler_path}")
            print(f"DEBUG - Scaler details:")
            print(f"  Type: {type(scaler)}")
            print(f"  Feature names: {list(scaler.feature_names_in_) if hasattr(scaler, 'feature_names_in_') else 'None'}")
            if hasattr(scaler, 'data_min_') and hasattr(scaler, 'data_max_'):
                print(f"  data_min_: {scaler.data_min_}")
                print(f"  data_max_: {scaler.data_max_}")
            if hasattr(scaler, 'feature_range'):
                print(f"  feature_range: {scaler.feature_range}")
            if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
                print(f"  mean_: {scaler.mean_}")
                print(f"  scale_: {scaler.scale_}")
            return scaler
            
        except Exception as e:
            print(f"Error loading scaler: {e}")
            return None
    
    def _denormalize_values(self, y_pred, y_true, target_col):
        """Denormalize predictions and true values using MinMaxScaler formula.
        This should only be called when normalization was applied during training."""
        try:
            if self.scaler is None:
                print("WARNING: Denormalization requested but no scaler available")
                return y_pred, y_true
            
            # Get target column index
            target_idx = list(self.scaler.feature_names_in_).index(target_col)
            
            # Handle both 1D and 2D scaler arrays (matching reference code exactly)
            if isinstance(self.scaler.data_min_, np.ndarray) and len(self.scaler.data_min_.shape) == 2:
                # 2D array case: shape (1, n_features)
                target_min = self.scaler.data_min_[0][target_idx]
                target_max = self.scaler.data_max_[0][target_idx]
            else:
                # 1D array case: shape (n_features,)
                target_min = self.scaler.data_min_[target_idx]
                target_max = self.scaler.data_max_[target_idx]
            
            target_scale = target_max - target_min
            
            print(f"Denormalization parameters for {target_col}:")
            print(f"  Target index: {target_idx}")
            print(f"  Scaler data_min_ shape: {self.scaler.data_min_.shape}")
            print(f"  Scaler data_max_ shape: {self.scaler.data_max_.shape}")
            print(f"  Data min: {target_min}")
            print(f"  Data max: {target_max}")
            print(f"  Scale (max-min): {target_scale}")
            
            # Show sample values before denormalization
            print(f"Sample normalized values - Pred: {y_pred[:3]}, True: {y_true[:3]}")
            
            # Apply MinMaxScaler denormalization formula: X_orig = X_scaled * (max - min) + min
            y_pred_denorm = y_pred * target_scale + target_min
            y_true_denorm = y_true * target_scale + target_min
            
            # Show sample values after denormalization
            print(f"Sample denormalized values - Pred: {y_pred_denorm[:3]}, True: {y_true_denorm[:3]}")
            print(f"Denormalized value ranges - Pred: [{y_pred_denorm.min():.3f}, {y_pred_denorm.max():.3f}], True: [{y_true_denorm.min():.3f}, {y_true_denorm.max():.3f}]")
            
            print(f"Successfully denormalized values for {target_col}")
            return y_pred_denorm, y_true_denorm
            
        except Exception as e:
            print(f"Error denormalizing values: {e}")
            import traceback
            traceback.print_exc()
            print("Returning original values")
            return y_pred, y_true
    
    def _calculate_metrics_and_format_results(self, y_pred, y_true, target_col, task):
        """Calculate metrics and format results."""
        target_column_name = target_col.lower()
        
        # Determine unit suffix and multiplier
        if "voltage" in target_column_name:
            unit_suffix = "_mv"
            unit_display = "mV"
            multiplier = 1000.0
        elif "soc" in target_column_name:
            unit_suffix = "_percent"
            unit_display = "SOC"
            multiplier = 100.0 if np.max(np.abs(y_true)) <= 1.0 else 1.0
        elif "temperature" in target_column_name or "temp" in target_column_name:
            unit_suffix = "_degC"
            unit_display = "Deg C"
            multiplier = 1.0
        else:
            unit_suffix = ""
            unit_display = ""
            multiplier = 1.0
        
        # Calculate metrics
        rms_error_val = np.sqrt(mean_squared_error(y_true, y_pred)) * multiplier
        mae_val = mean_absolute_error(y_true, y_pred) * multiplier
        mape_denominator = np.maximum(1e-10, np.abs(y_true))
        mape = np.mean(np.abs((y_true - y_pred) / mape_denominator)) * 100
        r2 = r2_score(y_true, y_pred)
        
        # Calculate error values for SOC
        error_percent_soc_values = (y_true - y_pred) * multiplier
        
        print(f"Metrics - RMS Error: {rms_error_val:.3f} {unit_display}, MAE: {mae_val:.3f} {unit_display}, MAPE: {mape:.2f}%, R²: {r2:.4f}")
        
        # Format results similar to original method
        results_dict = {
            'predictions': y_pred,
            'true_values': y_true,
            f'rms_error{unit_suffix}': rms_error_val,
            f'mae{unit_suffix}': mae_val,
            'mape_percent': mape,
            'r2': r2,
            'unit_display': unit_display,
            'multiplier': multiplier,
            'error_percent_soc_values': error_percent_soc_values
        }
        
        return results_dict
    
    def _create_model_instance(self, task):
        """Create model instance from task metadata when loading state_dict."""
        try:
            model_metadata = task.get('model_metadata', {})
            model_type = model_metadata.get('model_type', 'LSTM')
            
            # Import model classes
            from vestim.services.model_training.src.LSTM_model_service_test import LSTMModel, LSTMModelLN, LSTMModelBN
            from vestim.services.model_training.src.GRU_model import GRUModel
            
            # Get model parameters
            input_size = len(task.get('data_loader_params', {}).get('feature_columns', []))
            output_size = 1  # Single target prediction
            
            # Model-specific parameter handling
            if model_type in ['LSTM', 'LSTM_LN', 'LSTM_BN', 'GRU']:
                hidden_units = model_metadata.get('hidden_units', 64)
                num_layers = model_metadata.get('num_layers', 2)
                dropout = model_metadata.get('dropout', 0.2)
                
                # Create appropriate RNN model type
                if model_type == 'LSTM_LN':
                    model = LSTMModelLN(input_size, hidden_units, num_layers, output_size, dropout)
                elif model_type == 'LSTM_BN':
                    model = LSTMModelBN(input_size, hidden_units, num_layers, output_size, dropout)
                elif model_type == 'GRU':
                    model = GRUModel(input_size, hidden_units, num_layers, output_size, dropout)
                else:  # Default LSTM
                    model = LSTMModel(input_size, hidden_units, num_layers, output_size, dropout)
            elif model_type == 'FNN':
                # For FNN models, use different parameters
                from vestim.services.model_training.src.FNN_model import FNNModel
                hidden_layer_sizes = model_metadata.get('hidden_layer_sizes', [64, 32])
                dropout_prob = model_metadata.get('dropout_prob', 0.2)
                model = FNNModel(input_size, hidden_layer_sizes, output_size, dropout_prob)
            else:
                # Fallback for unknown model types
                hidden_units = model_metadata.get('hidden_units', 64)
                num_layers = model_metadata.get('num_layers', 2)
                dropout = model_metadata.get('dropout', 0.2)
                model = LSTMModel(input_size, hidden_units, num_layers, output_size, dropout)
            
            print(f"Created {model_type} model instance")
            return model
            
        except Exception as e:
            print(f"Error creating model instance: {e}")
            raise
