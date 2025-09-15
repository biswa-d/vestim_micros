import os
import json
import datetime
import pandas as pd
import joblib
import torch
import numpy as np
import time
from PyQt5.QtCore import QObject, pyqtSignal
from vestim.services.data_processor.src.data_augment_service import DataAugmentService
from vestim.services.model_training.src.FNN_model import FNNModel
from vestim.services.model_training.src.LSTM_model import LSTMModel
from vestim.services.model_training.src.GRU_model import GRUModel
from vestim.services.model_testing.src.testing_service import apply_inference_filter
from vestim.services import normalization_service as norm_svc

class VEstimStandaloneTestingManager(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    results_ready = pyqtSignal(dict)
    augmentation_required = pyqtSignal(pd.DataFrame, list)

    def __init__(self, job_folder_path, test_data_path, session_timestamp=None):
        super().__init__()
        self.job_folder_path = job_folder_path
        self.test_data_path = test_data_path
        self.session_timestamp = session_timestamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_augment_service = DataAugmentService()
        self.test_df = None
        self.overall_results = {}
    
    def start(self):
        """Start the testing process (called by test selection GUI)"""
        self.start_testing()
        
    def start_testing(self):
        """Public interface method for starting testing (called by GUI)"""
        self.run_test()

    def run_test(self):
        try:
            self.progress.emit("Starting test...")
            self.progress.emit("Loading configurations...")
            job_metadata_path = os.path.join(self.job_folder_path, 'job_metadata.json')
            self.aug_metadata_path = os.path.join(self.job_folder_path, 'augmentation_metadata.json')
            
            with open(job_metadata_path, 'r') as f:
                self.job_metadata = json.load(f)

            self.progress.emit(f"Loading test data from {os.path.basename(self.test_data_path)}...")
            self.test_df = pd.read_csv(self.test_data_path)
            self.original_test_df = self.test_df.copy()  # Store original for later saving
            
            # Try to automatically apply augmentation steps
            if os.path.exists(self.aug_metadata_path):
                self.progress.emit("Found augmentation metadata. Applying automatic augmentation...")
                augmented_df = self._apply_automatic_augmentation(self.test_df)
                self.resume_test_with_augmented_data(augmented_df)
            else:
                # Fallback to old validation approach
                if self.job_metadata.get('normalization_applied', False):
                    normalized_columns = self.job_metadata.get('normalized_columns')
                    missing_cols = [col for col in normalized_columns if col not in self.test_df.columns]
                    if missing_cols:
                        raise ValueError(f"Test data is missing columns required by the scaler: {missing_cols}")
                
                self.resume_test_with_augmented_data(self.test_df)

        except ValueError as e:
            if "missing columns required by the scaler" in str(e) or "Could not apply" in str(e):
                self.progress.emit(f"Automatic augmentation failed: {e}")
                self.progress.emit("Opening manual augmentation interface...")
                filter_configs = self._load_augmentation_configs()
                if filter_configs:
                    # Convert to legacy format for GUI compatibility
                    legacy_configs = [config for config in filter_configs if config.get("type") == "filter"]
                    self.augmentation_required.emit(self.test_df, legacy_configs)
                else:
                    self.progress.emit("Could not find augmentation steps. Please prepare the data manually.")
                    self.finished.emit()
            else:
                self.error.emit(f"Augmentation validation error: {e}")
                self.finished.emit()
        except Exception as e:
            self.error.emit(f"Testing failed: {e}")
            self.finished.emit()

    def resume_test_with_augmented_data(self, augmented_df):
        try:
            # Create consolidated test files directory using session timestamp
            test_file_basename = os.path.splitext(os.path.basename(self.test_data_path))[0]
            
            # Main job folder level: new_test_timestamp_files (single directory for all test files)
            new_tests_dir = os.path.join(self.job_folder_path, f'new_test_{self.session_timestamp}_files')
            os.makedirs(new_tests_dir, exist_ok=True)
            
            # Save raw test data for reference
            raw_test_file = os.path.join(new_tests_dir, f"raw_{test_file_basename}.csv")
            self.original_test_df.to_csv(raw_test_file, index=False)
            
            # Save augmented test data (the one that will be used for inference)
            augmented_test_file = os.path.join(new_tests_dir, f"augmented_{test_file_basename}.csv")
            augmented_df.to_csv(augmented_test_file, index=False)
            
            self.progress.emit(f"Test files saved in: {os.path.basename(new_tests_dir)}")
            self.progress.emit(f"Raw file: raw_{test_file_basename}.csv")
            self.progress.emit(f"Augmented file: augmented_{test_file_basename}.csv")
            
            # Actually save the files now
            if hasattr(self, 'original_test_df') and self.original_test_df is not None:
                self.original_test_df.to_csv(raw_test_file, index=False)
            else:
                # If original_test_df is not available, save from file
                original_df = pd.read_csv(self.test_data_path)
                original_df.to_csv(raw_test_file, index=False)
                
            # Save the augmented data
            augmented_df.to_csv(augmented_test_file, index=False)
            
            # Store test session metadata
            self.test_session_id = self.session_timestamp
            self.session_dir = new_tests_dir
            
            self.test_df = augmented_df
            self.progress.emit("Resuming test with augmented data...")
            



            scaler = None
            if self.job_metadata.get('normalization_applied', False):

                self.progress.emit("Loading scaler for denormalization...")
                scaler_path = os.path.normpath(os.path.join(self.job_folder_path, 'scalers', 'augmentation_scaler.joblib'))


                scaler = norm_svc.load_scaler(scaler_path)
                if scaler:
                    self.progress.emit("✓ Scaler loaded successfully")
                    normalized_columns = self.job_metadata.get('normalized_columns')

                    self.test_df[normalized_columns] = scaler.transform(self.test_df[normalized_columns])
                    self.progress.emit("Normalization applied successfully.")
                else:
                    self.progress.emit("⚠ Warning: Failed to load scaler, predictions will be on normalized scale")
            else:

                self.progress.emit("Normalization was not applied during training. Skipping.")

            models_dir = os.path.join(self.job_folder_path, 'models')


            if not os.path.exists(models_dir):
                self.progress.emit("No models directory found in job folder.")
                self.finished.emit()
                return
                
            # Scan all task directories in all model architecture folders
            self.progress.emit("Scanning for trained models...")

            task_directories = self._scan_task_directories(models_dir)

            
            if not task_directories:
                self.progress.emit("No trained models found.")

                self.finished.emit()
                return
                
            self.progress.emit(f"Found {len(task_directories)} trained models:")

            for i, task_info in enumerate(task_directories):
                arch_name = task_info.get('architecture_name', 'Unknown')
                task_name = task_info.get('task_name', 'Unknown')

                self.progress.emit(f"  - {arch_name}/{task_name}")
                model_type = task_info.get('model_type', 'Unknown')
                self.progress.emit(f"  - {arch_name}/{task_name} ({model_type})")
            
            # Use the session timestamp that was passed to constructor
            self.progress.emit(f"\nStandalone test session: {self.session_timestamp}")

            
            # Test each model using existing task structure
            self.progress.emit(f"\n{'='*60}")
            self.progress.emit("STARTING STANDALONE TESTING")
            self.progress.emit(f"{'='*60}")

            
            successful_tests = 0
            failed_tests = 0
            
            for i, task_info in enumerate(task_directories):
                try:

                    self.progress.emit(f"\n--- Testing {i+1}/{len(task_directories)}: {task_info['architecture_name']}/{task_info['task_name']} ---")
                    success = self._test_task_model(task_info, self.test_df.copy(), scaler, self.job_metadata)

                    if success:
                        successful_tests += 1
                    else:
                        failed_tests += 1
                except Exception as e:
                    self.progress.emit(f"ERROR: {e}")
                    failed_tests += 1
            
            # Final summary
            self.progress.emit(f"\n{'='*60}")
            self.progress.emit("STANDALONE TESTING COMPLETE")
            self.progress.emit(f"{'='*60}")
            self.progress.emit(f"Total models tested: {len(task_directories)}")
            self.progress.emit(f"Successful: {successful_tests}")
            self.progress.emit(f"Failed: {failed_tests}")
            self.progress.emit(f"Test session: {self.session_timestamp}")
            
            self.finished.emit()

        except Exception as e:
            self.progress.emit(f"\nERROR: {e}")
            self.finished.emit()

    def _apply_automatic_augmentation(self, df):
        """Automatically apply all augmentation steps from metadata."""
        try:
            with open(self.aug_metadata_path, 'r') as f:
                metadata = json.load(f)
            
            result_df = df.copy()
            
            # Apply resampling if needed
            resampling_info = metadata.get('resampling', {})
            if resampling_info.get('applied', False):
                self.progress.emit(f"Applying resampling to {resampling_info.get('frequency', 'unknown')} frequency...")
                result_df = self.data_augment_service.resample_data(result_df, resampling_info.get('frequency'))
            
            # Apply padding if needed
            padding_info = metadata.get('padding', {})
            if padding_info.get('applied', False):
                self.progress.emit(f"Applying padding (length: {padding_info.get('length')})...")
                result_df = self.data_augment_service.pad_data(
                    result_df, 
                    padding_info.get('length'),
                    resample_freq_for_time_padding=padding_info.get('resampling_frequency_for_padding')
                )
            
            # Apply filters
            applied_filters = metadata.get('applied_filters', [])
            for filter_config in applied_filters:
                self.progress.emit(f"Applying Butterworth filter to '{filter_config['column']}'...")
                result_df = self.data_augment_service.apply_butterworth_filter(
                    result_df,
                    column_name=filter_config['column'],
                    corner_frequency=filter_config['corner_frequency'],
                    sampling_rate=filter_config['sampling_rate'],
                    filter_order=filter_config['filter_order'],
                    output_column_name=filter_config['output_column_name']
                )
            
            # Apply calculated columns
            created_columns = metadata.get('created_columns', [])
            if created_columns:
                column_formulas = [(col['column_name'], col['formula']) for col in created_columns]
                self.progress.emit(f"Creating {len(column_formulas)} calculated columns...")
                result_df = self.data_augment_service.create_columns(result_df, column_formulas)
            
            self.progress.emit(f"✓ Automatic augmentation completed. Shape: {result_df.shape}")
            return result_df
            
        except Exception as e:
            raise ValueError(f"Could not apply automatic augmentation: {e}")

    def _load_augmentation_configs(self):
        """Load augmentation configurations, preferring structured JSON over text log."""
        # Try to load from structured JSON first
        if hasattr(self, 'aug_metadata_path') and os.path.exists(self.aug_metadata_path):
            try:
                self.progress.emit("Loading augmentation configuration from structured metadata...")
                with open(self.aug_metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                filter_configs = metadata.get('applied_filters', [])
                created_columns = metadata.get('created_columns', [])
                
                # Combine filters and created columns into a single list for processing
                all_configs = []
                
                # Add filter configurations
                for config in filter_configs:
                    all_configs.append({
                        "type": "filter",
                        "column": config.get("column"),
                        "output_column_name": config.get("output_column_name"),
                        "filter_order": config.get("filter_order"),
                        "corner_frequency": config.get("corner_frequency"),
                        "sampling_rate": config.get("sampling_rate")
                    })
                
                # Add created column configurations
                for config in created_columns:
                    all_configs.append({
                        "type": "calculated",
                        "column_name": config.get("column_name"),
                        "formula": config.get("formula")
                    })
                
                self.progress.emit(f"Found {len(filter_configs)} filters and {len(created_columns)} calculated columns in structured metadata.")
                return all_configs
                
            except Exception as e:
                self.progress.emit(f"Warning: Could not parse structured metadata: {e}")
        
        # Fallback - return empty list since we removed text log support
        return []

    def _scan_task_directories(self, models_dir):
        """Scan for all task directories in the existing job structure."""
        task_directories = []
        
        try:
            # Iterate through architecture folders (FNN_32_64, LSTM_64_128, etc.)
            for arch_folder in os.listdir(models_dir):
                arch_path = os.path.join(models_dir, arch_folder)
                if not os.path.isdir(arch_path):
                    continue
                
                # Iterate through task folders (B1024_rep-1, B256_rep-1, etc.)
                for task_folder in os.listdir(arch_path):
                    task_path = os.path.join(arch_path, task_folder)
                    if not os.path.isdir(task_path):
                        continue
                    
                    task_info_file = os.path.join(task_path, 'task_info.json')
                    best_model_file = os.path.join(task_path, 'best_model.pth')
                    
                    if os.path.exists(task_info_file) and os.path.exists(best_model_file):
                        try:
                            # Load task information
                            with open(task_info_file, 'r') as f:
                                task_info = json.load(f)
                            
                            # Add path information
                            task_info['architecture_name'] = arch_folder
                            task_info['task_name'] = task_folder
                            task_info['task_path'] = task_path
                            task_info['model_file'] = best_model_file
                            task_info['task_info_file'] = task_info_file
                            
                            task_directories.append(task_info)
                            
                        except Exception as e:
                            self.progress.emit(f"Warning: Could not load task info from {task_info_file}: {e}")
                    else:
                        if not os.path.exists(task_info_file):
                            self.progress.emit(f"Warning: Missing task_info.json in {task_path}")
                        if not os.path.exists(best_model_file):
                            self.progress.emit(f"Warning: Missing best_model.pth in {task_path}")
        
        except Exception as e:
            self.progress.emit(f"Error scanning task directories: {e}")
        
        return task_directories

    def _test_task_model(self, task_info, test_df, scaler, job_metadata):
        """Test a single model using the existing task structure."""
        try:
            arch_name = task_info['architecture_name']
            task_name = task_info['task_name'] 
            task_path = task_info['task_path']
            model_file = task_info['model_file']
            
            self.progress.emit(f"  Architecture: {arch_name}")
            self.progress.emit(f"  Task: {task_name}")
            
            # Extract model configuration from task_info
            model_type = task_info.get('model_type', 'FNN')
            hyperparams = task_info.get('hyperparams', {})
            data_config = task_info.get('data_config', {})
            training_config = task_info.get('training_config', {})
            
            # Feature and target columns are in hyperparams (not data_config)
            feature_columns = hyperparams.get('FEATURE_COLUMNS', data_config.get('feature_columns', []))
            target_column = hyperparams.get('TARGET_COLUMN', data_config.get('target_column'))
            
            # Handle LOOKBACK which might be "N/A" for FNN models
            lookback_val = hyperparams.get('LOOKBACK', data_config.get('lookback', 0))
            if lookback_val == "N/A" or lookback_val is None:
                lookback = 0
            else:
                lookback = int(lookback_val)
                
            training_method = hyperparams.get('TRAINING_METHOD', training_config.get('training_method', 'Sequential'))
            
            self.progress.emit(f"  Model Type: {model_type}")
            self.progress.emit(f"  Target: {target_column}")
            self.progress.emit(f"  Features: {len(feature_columns)} columns")
            if lookback > 0:
                self.progress.emit(f"  Lookback: {lookback} time steps")
            
            # Validate required columns
            missing_cols = [col for col in feature_columns if col not in test_df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Prepare test data
            data_to_process = test_df[feature_columns].values
            if training_method == 'WholeSequenceFNN':
                X_test = torch.tensor(data_to_process, dtype=torch.float32)
                effective_samples = len(data_to_process)
            else:
                if len(data_to_process) < lookback:
                    raise ValueError(f"Test data length ({len(data_to_process)}) < lookback ({lookback})")
                X_test_np = self._create_sequences(data_to_process, lookback)
                X_test = torch.tensor(X_test_np, dtype=torch.float32)
                effective_samples = len(X_test_np)
            
            # Create model instance based on architecture and hyperparams
            input_size = len(feature_columns)
            if model_type == 'FNN':
                model_input_size = input_size * lookback if training_method != 'WholeSequenceFNN' else input_size
                # Get hidden layer sizes - it's stored as a list in hyperparams
                hidden_sizes = hyperparams.get('HIDDEN_LAYER_SIZES', hyperparams.get('hidden_layer_sizes', [64, 32]))
                
                # CRITICAL: Use apply_clipped_relu logic exactly like training loop
                apply_clipped_relu = hyperparams.get('normalization_applied', False)
                
                model = FNNModel(
                    input_size=model_input_size, 
                    output_size=hyperparams.get('OUTPUT_SIZE', hyperparams.get('output_size', 1)),
                    hidden_layer_sizes=hidden_sizes,
                    activation_function=hyperparams.get('activation', 'ReLU'),
                    dropout_prob=float(hyperparams.get('DROPOUT_PROB', hyperparams.get('dropout_prob', 0.0))),
                    apply_clipped_relu=apply_clipped_relu
                )
            elif model_type == 'LSTM':
                # CRITICAL: Use apply_clipped_relu logic exactly like training loop
                apply_clipped_relu = hyperparams.get('normalization_applied', False)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                model = LSTMModel(
                    input_size=input_size, 
                    hidden_units=int(hyperparams.get('HIDDEN_UNITS', hyperparams.get('hidden_size', 64))),
                    num_layers=int(hyperparams.get('LAYERS', hyperparams.get('num_layers', 2))),
                    device=device,
                    dropout_prob=float(hyperparams.get('DROPOUT_PROB', hyperparams.get('dropout_prob', 0.0))),
                    apply_clipped_relu=apply_clipped_relu
                )
            elif model_type == 'GRU':
                # CRITICAL: Use apply_clipped_relu logic exactly like training loop
                apply_clipped_relu = hyperparams.get('normalization_applied', False)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                model = GRUModel(
                    input_size=input_size,
                    hidden_units=int(hyperparams.get('GRU_HIDDEN_UNITS', hyperparams.get('hidden_size', 64))),
                    num_layers=int(hyperparams.get('GRU_LAYERS', hyperparams.get('num_layers', 2))),
                    device=device,
                    dropout_prob=float(hyperparams.get('GRU_DROPOUT_PROB', hyperparams.get('dropout_prob', 0.0))),
                    apply_clipped_relu=apply_clipped_relu
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Load trained weights
            checkpoint = torch.load(model_file, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
            model.eval()
            
            # Count model parameters and store in task_info
            total_params = sum(p.numel() for p in model.parameters())
            task_info['model_parameters'] = total_params
            
            # Run inference
            self.progress.emit(f"  Running inference on {effective_samples} samples...")
            start_time = time.time()
            with torch.no_grad():
                predictions_normalized = model(X_test)
            predictions_normalized = predictions_normalized.cpu().numpy()
            inference_time = time.time() - start_time
            
            # CRITICAL: Apply inference filters exactly like main training loop
            filter_type = hyperparams.get('INFERENCE_FILTER_TYPE', 'None')
            if filter_type != 'None':
                self.progress.emit(f"  Applying {filter_type} inference filter...")
                predictions_normalized = apply_inference_filter(predictions_normalized.flatten(), {'hyperparams': hyperparams})
                predictions_normalized = predictions_normalized.reshape(-1, 1)
                self.progress.emit(f"  ✓ {filter_type} filter applied")
            
            # Denormalize predictions
            if scaler:
                predictions_final = norm_svc.inverse_transform_single_column(
                    predictions_normalized, scaler, target_column, job_metadata.get('normalized_columns')
                )
                self.progress.emit("  ✓ Predictions denormalized")
            else:
                predictions_final = predictions_normalized.flatten()
            
            # Create individual task result directory using session timestamp: new_test_result_{session_timestamp}
            test_result_dir = os.path.join(task_path, f'new_test_result_{self.session_timestamp}')
            os.makedirs(test_result_dir, exist_ok=True)
            
            test_file_name = os.path.splitext(os.path.basename(self.test_data_path))[0]
            
            # Determine target column display name and error units (match main loop logic)
            if "voltage" in target_column.lower():
                target_display = "Voltage"
                error_unit = "mV"
                error_multiplier = 1000.0  # Convert V to mV for error display
            elif "soc" in target_column.lower():
                target_display = "SOC"
                error_unit = "% SOC"
                error_multiplier = 100.0  # Convert 0-1 to percentage for error display
            elif "temperature" in target_column.lower() or "temp" in target_column.lower():
                target_display = "Temperature"
                error_unit = "°C"
                error_multiplier = 1.0  # Temperature already in correct scale
            else:
                target_display = target_column.title()
                error_unit = "units"
                error_multiplier = 1.0
            
            if training_method == 'WholeSequenceFNN':
                # For FNN, use full test data - use ORIGINAL test data for actual values
                final_df = self.original_test_df.copy()  # Use original (not normalized) data
                final_df[f'Predicted_{target_display}'] = predictions_final
                
                if target_column in final_df.columns:
                    # Get actual values from ORIGINAL test data (not normalized)
                    actual_values_original = final_df[target_column].values
                    # Rename target column to match main loop format  
                    final_df[f'True_{target_display}'] = actual_values_original
                    actual_values = actual_values_original  # These are already in original scale
                    predicted_values = predictions_final   # These are denormalized
            else:
                # For RNN, adjust for lookback - use ORIGINAL test data for actual values
                prediction_start_index = lookback - 1
                final_df = self.original_test_df.iloc[prediction_start_index:].copy()  # Original test_df
                final_df = final_df.iloc[:len(predictions_final)]
                final_df[f'Predicted_{target_display}'] = predictions_final
                
                if target_column in final_df.columns:
                    # Get actual values from ORIGINAL test data (not normalized)
                    actual_values_original = final_df[target_column].values
                    # Rename target column to match main loop format
                    final_df[f'True_{target_display}'] = actual_values_original
                    actual_values = actual_values_original  # These are already in original scale
                    predicted_values = predictions_final   # These are denormalized
            
            # Calculate error in appropriate units (like main loop)
            if target_column in test_df.columns:
                errors_raw = predicted_values - actual_values
                errors_display = errors_raw * error_multiplier
                final_df[f'Error ({error_unit})'] = errors_display
                
                # Remove the original target column since we renamed it
                if target_column in final_df.columns and f'True_{target_display}' in final_df.columns:
                    final_df = final_df.drop(columns=[target_column])
                
                # Keep only essential columns (like main loop): features, True_X, Predicted_X, Error
                essential_columns = []
                
                # Add feature columns (non-target columns from original data)
                for col in final_df.columns:
                    if col not in [f'True_{target_display}', f'Predicted_{target_display}', f'Error ({error_unit})']:
                        # Keep if it's a feature column (was in original test data)
                        if col in test_df.columns and col != target_column:
                            essential_columns.append(col)
                
                # Add target and prediction columns in main loop order
                essential_columns.extend([f'True_{target_display}', f'Predicted_{target_display}', f'Error ({error_unit})'])
                
                # Create clean final DataFrame with only essential columns
                final_df = final_df[essential_columns]
            else:
                # If no target column, just keep features and predictions
                essential_columns = []
                for col in final_df.columns:
                    if col != f'Predicted_{target_display}':
                        if col in test_df.columns:
                            essential_columns.append(col)
                essential_columns.append(f'Predicted_{target_display}')
                final_df = final_df[essential_columns]
                predicted_values = predictions_final
                actual_values = None
            
            # Save the clean predictions file (like main loop)
            predictions_file = os.path.join(test_result_dir, f"{test_file_name}_predictions.csv")
            final_df.to_csv(predictions_file, index=False)
            self.progress.emit(f"  ✓ Predictions saved: {predictions_file}")
            
            # Calculate error statistics for GUI (keep in memory, don't save extra files)
            if actual_values is not None:
                # Calculate comprehensive error metrics (raw values for accurate calculation)
                mae = np.mean(np.abs(predicted_values - actual_values))
                mse = np.mean((predicted_values - actual_values) ** 2)
                rmse = np.sqrt(mse)
                mape = np.mean(np.abs((actual_values - predicted_values) / np.maximum(1e-10, np.abs(actual_values)))) * 100
                r2 = 1 - (np.sum((actual_values - predicted_values) ** 2) / np.sum((actual_values - np.mean(actual_values)) ** 2))
                max_error = np.max(np.abs(predicted_values - actual_values))  # Add max absolute error
                
                # Extract training metrics for GUI display
                training_metrics = self._extract_training_metrics(task_path, task_info)
                
                # Get parameter count (exact number, not abbreviated)
                num_params = hyperparams.get('NUM_LEARNABLE_PARAMS', 'N/A')
                if isinstance(num_params, (int, float)):
                    num_params = int(num_params)  # Ensure exact integer
                
                # Emit results for GUI display
                results_data = {
                    'predictions': predicted_values,
                    'actual_values': actual_values,
                    'MAE': mae,
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAPE': mape,
                    'R²': r2,
                    'max_error': max_error,  # Add max error for summary
                    'model_type': model_type,
                    'architecture': arch_name,
                    'task': task_name,
                    'target_column': target_column,
                    'target_display': target_display,
                    'error_unit': error_unit,
                    'model_file_path': model_file,  # Add model file path for training metrics
                    'test_data_file': self.test_data_path,  # Add test data file path for display
                    'task_info': task_info,  # Add task info for model parameters
                    'predictions_file': predictions_file,  # Add file path for plotting
                    'inference_time': inference_time,
                    'training_info': training_metrics,  # Include training metrics for GUI
                    'num_params': num_params  # Add exact parameter count
                }
                self.results_ready.emit(results_data)
                self.progress.emit(f"  ✓ Results: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
                
                # Create error statistics for metadata (but don't save separate file)
                error_stats = {
                    'mae': mae,
                    'mse': mse, 
                    'rmse': rmse,
                    'mape': mape,
                    'r2_score': r2,
                    'samples_count': len(actual_values),
                    'error_unit': error_unit,
                    'error_multiplier': error_multiplier
                }
            else:
                # No target column available for error calculation
                error_stats = {
                    'mae': 'N/A',
                    'mse': 'N/A', 
                    'rmse': 'N/A',
                    'mape': 'N/A',
                    'r2_score': 'N/A',
                    'samples_count': len(predictions_final)
                }
                
                # Still emit results for GUI even without actual values
                results_data = {
                    'predictions': predictions_final,
                    'actual_values': None,
                    'MAE': 'N/A',
                    'MSE': 'N/A',
                    'RMSE': 'N/A',
                    'MAPE': 'N/A',
                    'R²': 'N/A',
                    'model_type': model_type,
                    'architecture': arch_name,
                    'task': task_name,
                    'target_column': target_column,
                    'target_display': target_display,
                    'error_unit': error_unit,
                    'model_file_path': model_file,  # Add model file path for training metrics
                    'test_data_file': self.test_data_path,  # Add test data file path for display
                    'task_info': task_info,  # Add task info for model parameters
                    'predictions_file': predictions_file,  # Add file path for plotting
                    'inference_time': inference_time,
                    'training_info': training_metrics
                }
                self.results_ready.emit(results_data)
                
            # Save essential test metadata
            training_metrics = self._extract_training_metrics(task_path, task_info)
            
            test_metadata = {
                'test_type': 'standalone_test',
                'test_timestamp': self.session_timestamp,
                'test_file': os.path.basename(self.test_data_path),
                'architecture': arch_name,
                'task': task_name,
                'model_type': model_type,
                'target_column': target_column,
                'target_display': target_display,
                'error_unit': error_unit,
                'samples_processed': effective_samples,
                'inference_time_seconds': inference_time,
                'training_info': training_metrics,
                'prediction_stats': {
                    'mean': float(np.mean(predictions_final)),
                    'std': float(np.std(predictions_final)),
                    'min': float(np.min(predictions_final)),
                    'max': float(np.max(predictions_final))
                }
            }
            
            # No individual metadata files needed - consolidated summary will be created by GUI
            
            # Display training metrics (like normal tool run)
            training_history = task_info.get('training_history', {})
            epochs = training_history.get('epochs_trained', 'N/A')
            best_train_loss = training_history.get('best_train_loss', 'N/A')
            best_val_loss = training_history.get('best_val_loss', 'N/A')
            
            self.progress.emit(f"  ✓ Training Info - Epochs: {epochs}, Best Train Loss: {best_train_loss}, Best Val Loss: {best_val_loss}")
            self.progress.emit(f"  ✓ Results saved to: {test_result_dir}")
            self.progress.emit(f"  ✓ Processing time: {time.time() - start_time:.2f}s")
            
            return True
            
        except Exception as e:
            self.progress.emit(f"  ✗ Error: {e}")
            return False

    def _extract_training_metrics(self, task_path, task_info):
        """Extract training metrics from training logs and task results."""
        training_metrics = {
            'epochs_trained': 'N/A',
            'best_train_loss': 'N/A',
            'best_val_loss': 'N/A',
            'final_train_loss': 'N/A',
            'final_val_loss': 'N/A',
            'training_time': 'N/A',
            'early_stopped': False
        }
        
        try:
            # First check task_info results
            results = task_info.get('results', {})
            if results.get('completed', False):
                training_metrics['best_val_loss'] = results.get('best_val_loss', 'N/A')
                training_metrics['best_epoch'] = results.get('best_epoch', 'N/A') 
                training_metrics['training_time'] = results.get('training_time', 'N/A')
                training_metrics['early_stopped'] = results.get('early_stopped', False)
            
            # Try to read training progress CSV for more detailed metrics
            logs_dir = os.path.join(task_path, 'logs')
            training_csv = os.path.join(logs_dir, 'training_progress.csv')
            
            if os.path.exists(training_csv):
                try:
                    df = pd.read_csv(training_csv)
                    if not df.empty:
                        training_metrics['epochs_trained'] = int(df['epoch'].max())
                        training_metrics['final_train_loss'] = float(df['train_loss_norm'].iloc[-1])
                        training_metrics['final_val_loss'] = float(df['val_loss_norm'].iloc[-1])
                        training_metrics['best_val_loss'] = float(df['best_val_loss_norm'].min())
                        
                        # Find best epoch
                        best_idx = df['best_val_loss_norm'].idxmin()
                        training_metrics['best_train_loss'] = float(df.loc[best_idx, 'train_loss_norm'])
                        
                except Exception as e:
                    self.progress.emit(f"    Warning: Could not parse training CSV: {e}")
            
            return training_metrics
                        
        except Exception as e:
            self.progress.emit(f"    Warning: Could not extract training metrics: {e}")
            return training_metrics
    
    def _create_sequences(self, data, lookback):
        X = []
        for i in range(len(data) - lookback + 1):
            X.append(data[i:(i + lookback)])
        return np.array(X)