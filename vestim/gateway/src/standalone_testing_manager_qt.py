import os
import json
import datetime
import pandas as pd
import joblib
import torch
import numpy as np
import time
import gc, sys
from PyQt5.QtCore import QObject, pyqtSignal
from vestim.services.data_processor.src.data_augment_service import DataAugmentService
from vestim.services.model_training.src.FNN_model import FNNModel
from vestim.services.model_training.src.LSTM_model import LSTMModel
from vestim.services.model_training.src.GRU_model import GRUModel
from vestim.services.model_testing.src.testing_service import apply_inference_filter
from vestim.services.data_processor.src import normalization_service as norm_svc

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
        self.padding_length = 0
    
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
            
            # Add support for different file types
            file_extension = os.path.splitext(self.test_data_path)[1].lower()
            if file_extension == '.csv':
                self.test_df = pd.read_csv(self.test_data_path)
            elif file_extension in ['.xlsx', '.xls']:
                self.test_df = pd.read_excel(self.test_data_path, sheet_name=0) # Default to first sheet
            else:
                # Fallback or error
                self.error.emit(f"Unsupported file type: {file_extension}. Please use .csv or .xlsx.")
                self.finished.emit()
                return

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
                    
                    # Pre-process columns before scaling to handle non-numeric types like timedelta
                    for col in normalized_columns:
                        if col in self.test_df.columns and self.test_df[col].dtype == 'object':
                            try:
                                # Attempt to convert timedelta-like strings to total seconds
                                self.test_df[col] = pd.to_timedelta(self.test_df[col]).dt.total_seconds()
                                self.progress.emit(f"Converted timedelta column '{col}' to seconds before scaling.")
                            except (ValueError, TypeError):
                                try:
                                    # Fallback for other non-numeric objects
                                    self.test_df[col] = pd.to_numeric(self.test_df[col], errors='coerce')
                                    self.progress.emit(f"Coerced object column '{col}' to numeric before scaling.")
                                except (ValueError, TypeError):
                                    self.error.emit(f"Column '{col}' could not be converted to a numeric type for scaling.")
                                    return

                    self.test_df[normalized_columns] = scaler.transform(self.test_df[normalized_columns])
                    self.progress.emit("Normalization applied successfully.")
                else:
                    self.progress.emit("Warning: Failed to load scaler, predictions will be on normalized scale")
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
            self.padding_length = 0
            
            # Apply resampling if needed
            resampling_info = metadata.get('resampling', {})
            if resampling_info.get('applied', False):
                self.progress.emit(f"Applying resampling to {resampling_info.get('frequency', 'unknown')} frequency...")
                result_df = self.data_augment_service.resample_data(result_df, resampling_info.get('frequency'))
            
            # Apply padding if needed
            padding_info = metadata.get('padding', {})
            if padding_info.get('applied', False):
                padding_length = padding_info.get('length', 0)
                self.padding_length = padding_length
                self.progress.emit(f"Applying padding (length: {padding_length})...")
                result_df = self.data_augment_service.pad_data(
                    result_df,
                    padding_length,
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
        model = None
        try:
            arch_name = task_info['architecture_name']
            task_name = task_info['task_name']
            task_path = task_info['task_path']
            model_file = task_info['model_file']
            
            self.progress.emit(f"  Architecture: {arch_name}")
            self.progress.emit(f"  Task: {task_name}")
            
            # Define device for model and tensors
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Extract model configuration from task_info
            model_metadata = task_info.get('model_metadata', {})
            model_type = model_metadata.get('model_type', task_info.get('model_type', 'FNN'))
            hyperparams = task_info.get('hyperparams', {})
            data_config = task_info.get('data_config', {})
            training_config = task_info.get('training_config', {})
            
            feature_columns = hyperparams.get('FEATURE_COLUMNS', data_config.get('feature_columns', []))
            target_column = hyperparams.get('TARGET_COLUMN', data_config.get('target_column'))
            
            lookback_val = hyperparams.get('LOOKBACK', data_config.get('lookback', 0))
            lookback = 0 if lookback_val == "N/A" or lookback_val is None else int(lookback_val)
            
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
            
            # Create model instance based on architecture and hyperparams
            input_size = len(feature_columns)
            if model_type == 'FNN':
                model_input_size = input_size * lookback if training_method != 'WholeSequenceFNN' else input_size
                hidden_sizes = hyperparams.get('HIDDEN_LAYER_SIZES', [64, 32])
                apply_clipped_relu = hyperparams.get('normalization_applied', False)
                model = FNNModel(
                    input_size=model_input_size,
                    output_size=hyperparams.get('OUTPUT_SIZE', 1),
                    hidden_layer_sizes=hidden_sizes,
                    activation_function=hyperparams.get('activation', 'ReLU'),
                    dropout_prob=float(hyperparams.get('DROPOUT_PROB', 0.0)),
                    apply_clipped_relu=apply_clipped_relu
                )
            elif model_type == 'LSTM':
                apply_clipped_relu = hyperparams.get('normalization_applied', False)
                model = LSTMModel(
                    input_size=input_size,
                    hidden_units=int(hyperparams.get('HIDDEN_UNITS', 64)),
                    num_layers=int(hyperparams.get('LAYERS', 2)),
                    device=device,
                    dropout_prob=float(hyperparams.get('DROPOUT_PROB', 0.0)),
                    apply_clipped_relu=apply_clipped_relu
                )
            elif model_type == 'GRU':
                apply_clipped_relu = hyperparams.get('normalization_applied', False)
                model = GRUModel(
                    input_size=input_size,
                    hidden_units=int(hyperparams.get('GRU_HIDDEN_UNITS', 64)),
                    num_layers=int(hyperparams.get('GRU_LAYERS', 2)),
                    device=device,
                    dropout_prob=float(hyperparams.get('GRU_DROPOUT_PROB', 0.0)),
                    apply_clipped_relu=apply_clipped_relu
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Load trained weights and move model to the correct device
            checkpoint = torch.load(model_file, map_location=device)
            model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
            model.to(device)
            model.eval()
            
            # Count model parameters
            total_params = sum(p.numel() for p in model.parameters())
            task_info['model_parameters'] = total_params
            
            # Prepare and run inference
            self.progress.emit(f"  Running inference on {len(test_df)} samples...")
            start_time = time.time()
            predictions_normalized = None
            effective_samples = len(test_df)

            with torch.no_grad():
                # Use sample-by-sample inference for RNNs to avoid OOM
                if model_type in ['LSTM', 'GRU']:
                    data_to_process = test_df[feature_columns].values
                    
                    # Add warmup samples to initialize hidden states
                    warmup_samples = lookback
                    if len(data_to_process) > 0:
                        first_row = data_to_process[0]
                        warmup_data = np.array([first_row] * warmup_samples)
                        data_full = np.vstack([warmup_data, data_to_process])
                    else:
                        data_full = data_to_process # Handle empty test file
                    
                    if data_full.size == 0:
                        predictions_normalized = np.array([])
                    else:
                        X_all = torch.tensor(data_full.astype(np.float32)).view(-1, 1, len(feature_columns))
                        
                        y_preds_list = []
                        h_s = None  # Hidden state for GRU/LSTM
                        c_s = None  # Cell state for LSTM
                        
                        # Process data in chunks to avoid overwhelming the CPU->GPU transfer
                        chunk_size = 10000
                        for i in range(0, len(X_all), chunk_size):
                            chunk = X_all[i:i+chunk_size].to(device)
                            
                            for t in range(len(chunk)):
                                x_t = chunk[t].unsqueeze(0)  # Shape (1, 1, features)
                                
                                if model_type == 'LSTM':
                                    y_pred, (h_s, c_s) = model(x_t, h_s, c_s)
                                    if h_s is not None: h_s = h_s.detach()
                                    if c_s is not None: c_s = c_s.detach()
                                elif model_type == 'GRU':
                                    y_pred, h_s = model(x_t, h_s)
                                    if h_s is not None: h_s = h_s.detach()

                                # Store predictions after warmup period
                                if i + t >= warmup_samples:
                                    y_preds_list.append(y_pred.squeeze().cpu().numpy())

                        predictions_normalized = np.array(y_preds_list).reshape(-1, 1)

                else:  # FNN and other models - use batched inference
                    from torch.utils.data import TensorDataset, DataLoader
                    
                    data_to_process = test_df[feature_columns].values
                    if training_method == 'WholeSequenceFNN':
                        X_test_np = data_to_process
                    else: # Sequential FNN
                        if len(data_to_process) < lookback:
                            raise ValueError(f"Test data length ({len(data_to_process)}) < lookback ({lookback})")
                        X_test_np = self._create_sequences(data_to_process, lookback)
                    
                    if len(X_test_np) > 0:
                        X_test = torch.tensor(X_test_np, dtype=torch.float32)
                        test_dataset = TensorDataset(X_test)
                        batch_size = 4096
                        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                        
                        y_preds_list = []
                        for (x_batch,) in test_loader:
                            x_batch = x_batch.to(device)
                            y_pred_batch = model(x_batch)
                            y_preds_list.append(y_pred_batch.cpu().numpy())
                        
                        if y_preds_list:
                            predictions_normalized = np.vstack(y_preds_list)
                        else:
                            predictions_normalized = np.array([])
                    else:
                        predictions_normalized = np.array([])

            # Ensure predictions are numpy array for subsequent processing
            if isinstance(predictions_normalized, torch.Tensor):
                predictions_normalized = predictions_normalized.cpu().numpy()

            # Remove padding from predictions if it was applied
            if hasattr(self, 'padding_length') and self.padding_length > 0:
                if len(predictions_normalized) > self.padding_length:
                    self.progress.emit(f"  Removing {self.padding_length} padding predictions...")
                    predictions_normalized = predictions_normalized[self.padding_length:]
                    self.progress.emit(f"  ✓ Padding removed.")
                else:
                    self.progress.emit(f"  Warning: Predictions length ({len(predictions_normalized)}) is less than padding ({self.padding_length}). Cannot remove padding.")
            
            inference_time = time.time() - start_time
            
            # CRITICAL: Apply inference filters exactly like main training loop
            filter_type = hyperparams.get('INFERENCE_FILTER_TYPE', 'None')
            if filter_type != 'None' and len(predictions_normalized) > 0:
                self.progress.emit(f"  Applying {filter_type} inference filter...")
                predictions_normalized = apply_inference_filter(predictions_normalized.flatten(), {'hyperparams': hyperparams})
                predictions_normalized = predictions_normalized.reshape(-1, 1)
                self.progress.emit(f"  ✓ {filter_type} filter applied")
            
            # Denormalize predictions
            if scaler and len(predictions_normalized) > 0:
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
                error_multiplier = 1000.0
            elif "soc" in target_column.lower():
                target_display = "SOC"
                error_unit = "% SOC"
                error_multiplier = 100.0
            elif "temperature" in target_column.lower() or "temp" in target_column.lower():
                target_display = "Temperature"
                error_unit = "°C"
                error_multiplier = 1.0
            else:
                target_display = target_column.title()
                error_unit = "units"
                error_multiplier = 1.0
            
            # --- New, unified data alignment logic ---
            self.progress.emit("  Aligning predictions with original data...")
            final_df = self.original_test_df.copy()
            
            # Initialize actual_values from the full original dataframe
            if target_column in final_df.columns:
                actual_values = final_df[target_column].values
                final_df[f'True_{target_display}'] = actual_values
            else:
                actual_values = None

            # Determine where predictions should start
            # Determine where predictions should start, now accounting for padding
            padding_offset = self.padding_length if hasattr(self, 'padding_length') else 0

            if model_type in ['LSTM', 'GRU']:
                # For RNNs, warmup is handled during inference, so predictions start at index 0 of the original data
                prediction_start_index = 0
            elif training_method == 'WholeSequenceFNN':
                # Predictions align with original data after padding is removed, so start at 0
                prediction_start_index = 0
            else:  # Sequential FNN
                # The first valid prediction corresponds to the end of the first sequence
                # that is completely free of padded data.
                # This is at `padding_length + lookback - 1` in the padded data,
                # which corresponds to `lookback - 1` in the original data.
                prediction_start_index = lookback - 1 if lookback > 0 else 0

            # Create a NaN-filled array for predictions that matches the original df length
            predictions_aligned = np.full(len(final_df), np.nan)

            # Calculate how many predictions can be placed into the aligned array
            num_preds_to_place = min(len(predictions_final), len(predictions_aligned) - prediction_start_index)
            
            # Place the predictions at the correct starting index
            if num_preds_to_place > 0:
                predictions_aligned[prediction_start_index : prediction_start_index + num_preds_to_place] = predictions_final[:num_preds_to_place]

            # Assign the aligned predictions to the dataframe
            final_df[f'Predicted_{target_display}'] = predictions_aligned
            predicted_values = predictions_aligned # This array now has NaNs

            # --- End of new logic ---

            # Calculate error and finalize DataFrame for CSV
            if actual_values is not None and predicted_values is not None:
                # Important: For metric calculation, we must ignore NaNs
                # Create a mask for valid (non-NaN) prediction entries
                valid_indices = ~np.isnan(predicted_values)
                
                # Filter both actual and predicted values using the mask
                actual_values_for_metrics = actual_values[valid_indices]
                predicted_values_for_metrics = predicted_values[valid_indices]

                # Calculate error for the entire column (with NaNs where appropriate)
                # Suppress warnings for NaN calculations, as this is expected
                with np.errstate(invalid='ignore'):
                    errors_raw = predicted_values - actual_values
                errors_display = errors_raw * error_multiplier
                final_df[f'Error ({error_unit})'] = errors_display
                
                if target_column in final_df.columns and f'True_{target_display}' in final_df.columns:
                    final_df = final_df.drop(columns=[target_column])
                
                essential_columns = [col for col in self.original_test_df.columns if col in final_df.columns and col != target_column]
                essential_columns.extend([f'True_{target_display}', f'Predicted_{target_display}', f'Error ({error_unit})'])
                final_df = final_df[[col for col in essential_columns if col in final_df.columns]]
            else:
                actual_values_for_metrics = None # Ensure this is defined
            
            # Save predictions file
            predictions_file = os.path.join(test_result_dir, f"{test_file_name}_predictions.csv")
            final_df.to_csv(predictions_file, index=False)
            self.progress.emit(f"  ✓ Predictions saved: {predictions_file}")
            
            # Calculate metrics and emit results for GUI
            # Use the NaN-filtered arrays for metric calculation
            mae, mse, rmse, r2, max_error = ('N/A', 'N/A', 'N/A', 'N/A', 'N/A')
            try:
                if actual_values_for_metrics is not None and len(actual_values_for_metrics) > 0:
                    mae = np.mean(np.abs(predicted_values_for_metrics - actual_values_for_metrics))
                    mse = np.mean((predicted_values_for_metrics - actual_values_for_metrics) ** 2)
                    rmse = np.sqrt(mse)
                    r2 = 1 - (np.sum((actual_values_for_metrics - predicted_values_for_metrics) ** 2) / np.sum((actual_values_for_metrics - np.mean(actual_values_for_metrics)) ** 2))
                    max_error = np.max(np.abs(predicted_values_for_metrics - actual_values_for_metrics))
            except Exception as e:
                self.progress.emit(f"  Warning: Could not calculate metrics: {e}")

            training_metrics = self._extract_training_metrics(task_path, task_info)
            num_params = hyperparams.get('NUM_LEARNABLE_PARAMS', 'N/A')
            
            results_data = {
                'MAE': mae, 'RMSE': rmse, 'R²': r2, 'max_error': max_error,
                'predictions': predicted_values, # Send the full array with NaNs for plotting
                'actual_values': actual_values,   # Send the full actuals array for plotting
                'model_type': model_type, 'architecture': arch_name, 'task': task_name,
                'target_column': target_column, 'target_display': target_display, 'error_unit': error_unit,
                'model_file_path': model_file, 'test_data_file': self.test_data_path,
                'task_info': task_info, 'predictions_file': predictions_file,
                'inference_time': inference_time, 'training_info': training_metrics,
                'num_params': int(num_params) if isinstance(num_params, (int, float)) else 'N/A'
            }
            self.results_ready.emit(results_data)
            self.progress.emit(f"  ✓ Results: MAE={mae if isinstance(mae, str) else f'{mae:.4f}'}, RMSE={rmse if isinstance(rmse, str) else f'{rmse:.4f}'}, R²={r2 if isinstance(r2, str) else f'{r2:.4f}'}")
            
            # Display training metrics
            training_metrics = self._extract_training_metrics(task_path, task_info)
            self.progress.emit(f"  ✓ Training Info - Epochs: {training_metrics.get('epochs_trained', 'N/A')}, Best Train Loss: {training_metrics.get('best_train_loss', 'N/A')}, Best Val Loss: {training_metrics.get('best_val_loss', 'N/A')}")
            self.progress.emit(f"  ✓ Results saved to: {test_result_dir}")
            self.progress.emit(f"  ✓ Processing time: {time.time() - start_time:.2f}s")
            
            return True
            
        except Exception as e:
            self.progress.emit(f"  ✗ Error: {e}")
            return False
        finally:
            # Aggressively clean up memory after each model test
            del model
            if 'X_test' in locals(): del X_test
            if 'predictions_normalized' in locals(): del predictions_normalized
            if 'predictions_final' in locals(): del predictions_final
            if 'torch' in sys.modules and torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            task_name = task_info.get('task_name', 'Unknown')
            self.progress.emit(f"  ✓ Cleaned up memory for task {task_name}")

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
                    df = pd.read_csv(training_csv, comment='#', on_bad_lines='warn', engine='python')
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