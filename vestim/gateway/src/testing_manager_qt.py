# ---------------------------------------------------------------------------------
# Author: Biswanath Dehury
# Date: `{{date:2023-03-02}}`
# Version: 1.0.0
# Description: Description of the script
#Descrition: This is the batchtesting without padding implementation for the unscaled data where the batch-size is used for testloader preparation but the model is tested
# one sequence at a time like a running window. The first part of the test file is padded with data to avoid the size mismatch and get the final prediction the same
# shape as the test file.
# Commit code
# Copyright (c) 2024 Biswanath Dehury, Dr. Phil Kollmeyer's Battery Lab at McMaster University
# ---------------------------------------------------------------------------------




import torch
import os
import json, hashlib, sqlite3, csv, traceback, gc, sys
from threading import Thread, Lock
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np

from vestim.gateway.src.job_manager_qt import JobManager
from vestim.services.model_testing.src.testing_service import VEstimTestingService # Corrected import
from vestim.services.model_testing.src.test_data_service import VEstimTestDataService # Corrected import
from vestim.services.model_testing.src.continuous_testing_service import ContinuousTestingService # New continuous testing
from vestim.gateway.src.training_setup_manager_qt import VEstimTrainingSetupManager
import logging

class VEstimTestingManager:
    def __init__(self, job_manager=None, params=None, task_list=None, training_results=None):
        print("Initializing VEstimTestingManager...")
        self.logger = logging.getLogger(__name__)
        self.job_manager = job_manager if job_manager else JobManager()
        self.training_setup_manager = VEstimTrainingSetupManager(job_manager=self.job_manager)
        self.testing_service = VEstimTestingService()  # Keep old service for fallback
        # self.continuous_testing_service = ContinuousTestingService()  # Removed to prevent race conditions
        self.test_data_service = VEstimTestDataService()
        
        # Get device selection from hyperparameters if available, fallback to CUDA detection
        self.params = params if params is not None else {}
        device_selection = self.params.get('DEVICE_SELECTION', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Store the functional device string for backend services
        self.functional_device = device_selection if torch.cuda.is_available() and 'cuda' in device_selection else 'cpu'
        
        # Determine the full device name for display
        if 'cuda' in self.functional_device and torch.cuda.is_available():
            try:
                gpu_idx = int(self.functional_device.split(':')[-1]) if ':' in self.functional_device else 0
                full_device_name = torch.cuda.get_device_name(gpu_idx)
                self.display_device = f"{self.functional_device.upper()} ({full_device_name})"
            except Exception as e:
                self.logger.error(f"Could not get GPU name for {self.functional_device}: {e}")
                self.display_device = self.functional_device.upper()
        else:
            self.display_device = 'CPU'
        
        self.params['CURRENT_DEVICE'] = self.display_device
        self.logger.info(f"TestingManager: Functional device: {self.functional_device}, Display device: {self.display_device}")
        
        self.max_workers = 4  # Number of concurrent threads
        self.queue = None  # Initialize the queue attribute
        self.stop_flag = False  # Initialize the stop flag attribute
        self.task_list = task_list if task_list is not None else []
        self.training_results = training_results if training_results is not None else {}
        self.results_summary = []
        
        self.lock = Lock()
        self.successful_tests = 0
        self.failed_tests = 0
        print("Initialization complete.")

    def start_testing(self, queue):
        """Start the testing process and store the queue for results."""
        self.queue = queue  # Store the queue instance
        self.stop_flag = False  # Reset stop flag when starting testing
        print("Starting testing process...")

        # Create the thread to handle testing
        self.testing_thread = Thread(target=self._run_testing_tasks)
        self.testing_thread.setDaemon(True)
        self.testing_thread.start()

    def _run_testing_tasks(self):
        """The main function that runs testing tasks."""
        try:
            print("Getting test folder and results save directory...")
            test_folder = self.job_manager.get_test_folder()
            self.results_summary.clear()
            
            # Reset counters
            self.successful_tests = 0
            self.failed_tests = 0

            # Retrieve task list
            print("Retrieving task list from TrainingSetupManager...")
            task_list = self.task_list if self.task_list else self.training_setup_manager.get_task_list()

            if not task_list:
                raise ValueError("Task list is not available.")

            self.logger.info(f"Found {len(task_list)} tasks to run.")

            # Execute tasks in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_task = {
                    executor.submit(self._test_single_model, task, idx, test_folder): task
                    for idx, task in enumerate(task_list)
                }

                # Wait for tasks to complete
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        future.result()  # Retrieve the result
                    except Exception as exc:
                        self.logger.error(f"Task {task.get('task_id', 'unknown')} generated an exception in future: {exc}", exc_info=True)
                        self.queue.put({'task_error': f"Task {task.get('task_id', 'unknown')} failed: {exc}"})

            # Log summary
            total_run = self.successful_tests + self.failed_tests
            self.logger.info("--- TESTING SUMMARY ---")
            self.logger.info(f"Total tasks processed: {len(task_list)}")
            self.logger.info(f"Total tests attempted: {total_run}")
            self.logger.info(f"Successful tests: {self.successful_tests}")
            self.logger.info(f"Failed tests: {self.failed_tests}")
            self.logger.info("-----------------------")

            # Signal to the queue that all tasks are completed
            print("All tasks completed. Sending signal to GUI...")
            self.queue.put({'all_tasks_completed': True})

        except Exception as e:
            print(f"An error occurred during testing: {str(e)}")
            self.queue.put({'task_error': str(e)})


    def _test_single_model(self, task, idx, test_folder):
        """Test a single model and save the result."""
        try:
            self.logger.info(f"--- Starting _test_single_model for task_id: {task.get('task_id', 'UnknownTask')} (list index: {idx}) ---")
            
            # Get all test files first to accurately count failures if model is missing
            test_files = [f for f in os.listdir(test_folder) if f.endswith('.csv')]
            if not test_files:
                self.logger.warning(f"No test files found in {test_folder} for task {task.get('task_id')}. Skipping model.")
                return
            self.logger.info(f"Task {task.get('task_id')}: Found {len(test_files)} test files.")

            # Check for the trained model file
            model_path = task.get('training_params', {}).get('best_model_path')
            if not model_path or not os.path.exists(model_path):
                self.logger.error(f"Best model not found for task {task.get('task_id')}: {model_path}. Skipping all {len(test_files)} tests for this model.")
                self.queue.put({'task_error': f"Best model not found for task {task.get('task_id')}"})
                with self.lock:
                    self.failed_tests += len(test_files)  # All tests for this model failed
                return

            # --- If model exists, proceed with testing ---
            task_dir = task['task_dir']
            lookback = task['hyperparams']['LOOKBACK']
            num_learnable_params = task['hyperparams']['NUM_LEARNABLE_PARAMS']
            
            log_model_path = os.path.relpath(model_path, self.job_manager.get_job_folder())
            print(f"Testing model: {log_model_path} with lookback: {lookback}")
            
            test_results_dir = os.path.join(task_dir, 'test_results')
            os.makedirs(test_results_dir, exist_ok=True)

            for test_file_index, test_file in enumerate(test_files):
                continuous_testing_service = None
                file_results = None
                try:
                    # Create a new service for each file to ensure complete memory isolation
                    continuous_testing_service = ContinuousTestingService(device=self.functional_device)
                    file_name = os.path.splitext(test_file)[0]
                    test_file_path = os.path.join(test_folder, test_file)
                    
                    # IMPORTANT: Two different testing approaches:
                    # 1. CONTINUOUS TESTING (NEW - DEFAULT): Processes one sample at a time as single timesteps
                    #    - No sequence creation, no DataLoader, no lookback buffer
                    #    - Each sample fed as single timestep to LSTM: (1, 1, features)
                    #    - Hidden states persist across all test files for a single model
                    #    - Uses LSTM's natural recurrent memory for temporal dependencies
                    #    - More realistic for deployment scenarios (streaming inference)
                    # 2. SEQUENCE-BASED TESTING (OLD - FALLBACK): Creates sequences with padding, uses DataLoader
                    #    - Pads data and creates sequences of lookback length
                    #    - Resets hidden states for each sequence but maintains across batches
                    #    - More traditional approach
                    
                    # Default to new method, fallback to old dataloader method if needed
                    use_continuous_testing = True  # Set to False to use old dataloader method
                    
                    # Get lookback value for warmup
                    lookback_val = task.get('hyperparams', {}).get('LOOKBACK', 200) # Default if not found
                    
                    if use_continuous_testing:
                        # Use continuous testing - no test loader needed
                        test_df = pd.read_csv(test_file_path)
                        
                        file_results = continuous_testing_service.run_continuous_testing(
                            task=task,
                            model_path=model_path,
                            test_file_path=test_file_path,
                            is_first_file=(test_file_index == 0),
                            warmup_samples=lookback_val  # Use lookback as warmup period
                        )
                        
                        # Get timestamps from the corresponding RAW test file since processed files may not have them
                        # Try to find the raw file that corresponds to this processed file
                        timestamps = None
                        try:
                            # Extract base filename (remove path and extension)
                            processed_filename = os.path.basename(test_file_path)
                            base_name = os.path.splitext(processed_filename)[0]
                            
                            # Look for the raw file in the raw_data directory
                            raw_data_dir = test_file_path.replace('processed_data', 'raw_data')
                            raw_file_dir = os.path.dirname(raw_data_dir)
                            
                            # Try common raw file extensions
                            raw_extensions = ['.csv', '.xlsx', '.mat']
                            raw_file_path = None
                            
                            for ext in raw_extensions:
                                potential_raw_file = os.path.join(raw_file_dir, base_name + ext)
                                if os.path.exists(potential_raw_file):
                                    raw_file_path = potential_raw_file
                                    break
                            
                            if raw_file_path and raw_file_path.endswith('.csv'):
                                print(f"Loading timestamps from raw file: {raw_file_path}")
                                raw_df = pd.read_csv(raw_file_path)
                                
                                # Handle different possible timestamp column names
                                for col in raw_df.columns:
                                    if 'time' in col.lower().replace(" ", ""):
                                        timestamps = raw_df[col].values
                                        print(f"Found timestamp column '{col}' in raw file")
                                        break
                            else:
                                print(f"Raw file not found or not CSV, falling back to processed file timestamps")
                                # Fallback to processed file timestamps
                                for col in test_df.columns:
                                    if 'time' in col.lower().replace(" ", ""):
                                        timestamps = test_df[col].values
                                        print(f"Found timestamp column '{col}' in processed file")
                                        break
                                        
                        except Exception as e:
                            print(f"Error loading timestamps from raw file: {e}")
                            # Final fallback to processed file
                            for col in test_df.columns:
                                if 'time' in col.lower().replace(" ", ""):
                                    timestamps = test_df[col].values
                                    break
                        
                        if timestamps is None:
                            print("Warning: No timestamp column found, using index as fallback")
                            timestamps = np.arange(len(file_results.get('predictions', [])))
                        
                        # Store timestamps in file_results for later use
                        if timestamps is not None:
                            if file_results is not None and timestamps is not None:
                                file_results['timestamps'] = timestamps
                    else:
                        # Fallback to old dataloader method - create test loader only when needed
                        data_loader_params = task.get('data_loader_params', {})
                        feature_cols = data_loader_params.get('feature_columns')
                        target_col = data_loader_params.get('target_column')
                        # LOOKBACK and BATCH_SIZE for test loader might come from task hyperparams or a specific test config
                        # Using hyperparams for now, ensure these are appropriate for test data creation
                        batch_size_val = task.get('hyperparams', {}).get('BATCH_SIZE', 100) # Default if not found

                        if not feature_cols or not target_col:
                            self.logger.error(f"Missing feature_cols or target_col in task for {test_file_path}")
                            # Optionally, put an error in the queue or skip this file
                            continue

                        test_loader = self.test_data_service.create_test_file_loader(
                            test_file_path=test_file_path,
                            lookback=int(lookback_val),
                            batch_size=int(batch_size_val),
                            feature_cols=feature_cols,
                            target_col=target_col
                        )
                        
                        file_results = self.testing_service.run_testing(
                            task=task,
                            model_path=model_path,
                            test_loader=test_loader,
                            test_file_path=test_file_path
                        )
                    
                    if file_results is None:
                        self.logger.error(f"Testing service returned None for {test_file_path}. Skipping.")
                        with self.lock:
                            self.failed_tests += 1
                        continue

                    # The VEstimTestingService.test_model (called by run_testing) now returns results with dynamic keys
                    # e.g., 'rms_error_mv', 'rms_error_percent', 'mae_degC'
                    # It also returns 'predictions' and 'true_values'

                    target_column_name = task.get('data_loader_params', {}).get('target_column', 'value')
                    unit_suffix = ""
                    csv_unit_display = "" # For CSV column names like "True Values (V)"
                    error_unit_display = "" # For error column names like "RMS Error (mV)"

                    if "voltage" in target_column_name.lower():
                        unit_suffix = "_mv"
                        csv_unit_display = "(V)"
                        error_unit_display = "(mV)"  # Consistent with training GUI - errors in mV
                    elif "soc" in target_column_name.lower() or "soe" in target_column_name.lower() or "sop" in target_column_name.lower():
                        unit_suffix = "_percent"
                        if "soc" in target_column_name.lower():
                            csv_unit_display = "(SOC)"
                            error_unit_display = "(% SOC)"
                        elif "soe" in target_column_name.lower():
                            csv_unit_display = "(SOE)"
                            error_unit_display = "(% SOE)"
                        else: # sop
                            csv_unit_display = "(SOP)"
                            error_unit_display = "(% SOP)"
                    elif "temperature" in target_column_name.lower() or "temp" in target_column_name.lower():
                        unit_suffix = "_degC"
                        csv_unit_display = "(Deg C)"   # Match training GUI format
                        error_unit_display = "(Deg C)" # Consistent with training GUI
                    
                    # Define dynamic metric keys based on the unit suffix
                    rms_key = f"rms_error{unit_suffix}"
                    mae_key = f"mae{unit_suffix}"
                    max_error_key = f"max_abs_error{unit_suffix}"
                    
                    # Calculate difference for CSV
                    # Predictions and true values are in their original scale from file_results
                    y_true_scaled = file_results['true_values']
                    y_pred_scaled = file_results['predictions']
                    
                    difference = y_true_scaled - y_pred_scaled
                    # Apply appropriate multiplier based on target type for consistent error reporting
                    if "voltage" in target_column_name.lower():
                        difference *= 1000  # Convert V difference to mV for CSV display consistency with error metrics
                    elif "soc" in target_column_name.lower() or "soe" in target_column_name.lower() or "sop" in target_column_name.lower():
                        # Use the multiplier calculated earlier for SOC/SOE/SOP error
                        multiplier = file_results.get('multiplier', 100)
                        difference *= multiplier
                    
                    # Save predictions with dynamic column names - matching training GUI conventions
                    predictions_file = os.path.join(test_results_dir, f"{file_name}_predictions.csv")
                    
                    # Get timestamps from file_results or create fallback
                    timestamps = file_results.get('timestamps', np.arange(len(y_true_scaled)))
                    
                    # Debug: Check timestamp data
                    if len(timestamps) > 0:
                        print(f"Timestamp data: first={timestamps[0]}, last={timestamps[-1]}, length={len(timestamps)}")
                        # Check if timestamps are empty/null
                        non_null_timestamps = sum(1 for ts in timestamps if ts not in [None, '', 'nan', 'NaN'])
                        print(f"Non-null timestamps: {non_null_timestamps}/{len(timestamps)}")
                    else:
                        print("Warning: Empty timestamps array")
                    
                    # Prepare data for DataFrame
                    data_for_csv = {
                        'Timestamp': timestamps,
                        f'True {target_column_name} {csv_unit_display}': y_true_scaled,
                        f'Predicted {target_column_name} {csv_unit_display}': y_pred_scaled,
                    }
                    
                    # For all cases, use the calculated and scaled 'difference'
                    data_for_csv[f'Error {error_unit_display}'] = difference
                    
                    pd.DataFrame(data_for_csv).to_csv(predictions_file, index=False)
                    
                    # Add results to summary file with dynamic headers matching training GUI
                    # Calculate max absolute error with appropriate scaling
                    max_abs_error_val = np.max(np.abs(difference)) if difference.size > 0 else 0

                    # Generate shorthand name for the model task
                    # Use the new descriptive names if available (for Optuna), otherwise generate the old ones
                    # The model name is the parent directory of the task directory (e.g., FNN_32_16)
                    task_dir = task.get('task_dir')
                    if task_dir:
                        display_model_name = os.path.basename(os.path.dirname(task_dir))
                        # The task name is the descriptive task directory name (e.g., B128_LR_SLR_VP20)
                        display_task_name = os.path.basename(task_dir)
                    else:
                        # Fallback for safety
                        display_model_name = task.get('model_name', self.generate_shorthand_name(task))
                        display_task_name = task.get('task_id')

                    # Best losses are now read from training summary files or passed from training GUI
                    best_train_loss = 'N/A'
                    best_valid_loss = 'N/A'
                    completed_epochs = 'N/A'
                    
                    # Try to get training results from the passed dictionary first
                    task_id = task.get('task_id')
                    print(f"[TESTING MGR] Looking for training results for task {task_id}")
                    print(f"[TESTING MGR] Available task IDs in training_results: {list(self.training_results.keys())}")
                    training_result = self.training_results.get(task_id, {})
                    print(f"[TESTING MGR] Retrieved training_result: {training_result}")
                    
                    best_train_loss = 'N/A'
                    best_valid_loss = 'N/A'
                    completed_epochs = 'N/A'
                    
                    if training_result:
                        best_train_loss = training_result.get('best_train_loss', 'N/A')
                        best_valid_loss = training_result.get('best_validation_loss', 'N/A')
                        completed_epochs = training_result.get('completed_epochs', 'N/A')
                        print(f"[TESTING MGR] Using in-memory results: train={best_train_loss}, valid={best_valid_loss}, epochs={completed_epochs}")
                    
                    # If not available in memory (or empty), ALWAYS try to read from training summary JSON file on disk
                    print(f"[TESTING MGR] Disk fallback check: training_result={bool(training_result)}, best_train_loss={best_train_loss}, best_valid_loss={best_valid_loss}")
                    if not training_result or best_train_loss == 'N/A' or best_valid_loss == 'N/A':
                        print(f"[TESTING MGR] *** TRIGGERING DISK FALLBACK *** Attempting to read from disk...")
                        try:
                            # training_summary.json is saved in task_dir (includes hyperparam subfolder) for per-config results
                            task_dir = task.get('task_dir') or task.get('model_dir')
                            print(f"[TESTING MGR] Task directory: {task_dir}")
                            if task_dir:
                                summary_file = os.path.join(task_dir, 'training_summary.json')
                                print(f"[TESTING MGR] Looking for summary file: {summary_file}")
                                if os.path.exists(summary_file):
                                    print(f"[TESTING MGR] *** FOUND FILE *** Reading training_summary.json...")
                                    with open(summary_file, 'r') as f:
                                        summary_data = json.load(f)
                                        print(f"[TESTING MGR] Raw summary_data keys: {summary_data.keys()}")
                                        best_train_loss = summary_data.get('best_train_loss_denormalized', 'N/A')
                                        best_valid_loss = summary_data.get('best_validation_loss_denormalized', 'N/A')
                                        completed_epochs = summary_data.get('completed_epochs', 'N/A')
                                        print(f"[TESTING MGR] *** SUCCESSFULLY LOADED *** train={best_train_loss}, valid={best_valid_loss}, epochs={completed_epochs}")
                                        self.logger.info(f"Loaded training results from disk for task {task['task_id']}: train={best_train_loss}, val={best_valid_loss}, epochs={completed_epochs}")
                                else:
                                    print(f"[TESTING MGR] *** FILE NOT FOUND *** training_summary.json NOT FOUND at {summary_file}")
                            else:
                                print(f"[TESTING MGR] *** NO DIRECTORY *** No task_dir or model_dir found in task")
                        except Exception as e:
                            print(f"[TESTING MGR] *** ERROR READING FILE *** {e}")
                            import traceback
                            traceback.print_exc()
                            self.logger.warning(f"Could not read training summary from disk for task {task['task_id']}: {e}")
                    else:
                        print(f"[TESTING MGR] *** SKIPPING DISK FALLBACK *** Using in-memory results")

                    summary_row = {
                        "Sl.No": f"{idx + 1}.{test_file_index + 1}",
                        "Task ID": display_task_name,
                        "Model": display_model_name,
                        "File Name": test_file,
                        "#W&Bs": num_learnable_params,
                        "Best Train Loss": best_train_loss,
                        "Best Valid Loss": best_valid_loss,
                        "Epochs Trained": completed_epochs,
                        f"Test RMSE {error_unit_display}": f"{file_results.get(rms_key, float('nan')):.2f}",
                        f"Test MAXE {error_unit_display}": f"{max_abs_error_val:.2f}",
                        "R2": f"{file_results.get('r2', float('nan')):.4f}"
                    }
                    self.results_summary.append(summary_row)
                    
                    # Data to send to GUI for this specific test file
                    gui_result_data = {
                        'saved_dir': test_results_dir,
                        'task_name': display_task_name, # Use the descriptive task name
                        'sl_no': f"{idx + 1}.{test_file_index + 1}", # Unique Sl.No for GUI based on task and test file
                        'model_name': display_model_name, # Use the descriptive model name
                        'file_name': test_file, # Current test file name
                        '#params': num_learnable_params,
                        'best_train_loss': best_train_loss,
                        'best_valid_loss': best_valid_loss,
                        'completed_epochs': completed_epochs,
                        # Create a more concise task_info for the GUI
                        'task_info': {
                            'task_id': task.get('task_id'),
                            'model_type': task.get('model_metadata', {}).get('model_type'),
                            'lookback': task.get('hyperparams', {}).get('LOOKBACK'),
                            'repetitions': task.get('hyperparams', {}).get('REPETITIONS'),
                            # Add other key identifiers if needed by GUI, but avoid full hyperparam dict
                            'layers': task.get('hyperparams', {}).get('LAYERS'),
                            'hidden_units': task.get('hyperparams', {}).get('HIDDEN_UNITS'),
                        },
                        'target_column': target_column_name, # Add target column for plotting
                        'predictions_file': predictions_file, # Correct path to the predictions file for plotting
                        'unit_display': error_unit_display, # Pass error unit display for GUI consistency
                        'csv_unit_display': csv_unit_display # Pass value unit display for GUI consistency
                    }
                    
                    # Add dynamic metrics from file_results directly with better error handling
                    gui_result_data[rms_key] = file_results.get(rms_key, 'N/A')
                    gui_result_data[mae_key] = file_results.get(mae_key, 'N/A')
                    gui_result_data[f'max_abs_error{unit_suffix}'] = max_abs_error_val
                    gui_result_data['r2'] = file_results.get('r2', 'N/A')
                    
                    # Include the unit suffix directly for GUI use
                    gui_result_data['unit_suffix'] = unit_suffix
                    gui_result_data['unit_display'] = error_unit_display
                    
                    # For backward compatibility with GUI
                    if unit_suffix == "_mv":
                        gui_result_data['rms_error_mv'] = file_results.get(rms_key, 'N/A')
                        gui_result_data['mae_mv'] = file_results.get(mae_key, 'N/A')
                        gui_result_data['max_error_mv'] = max_abs_error_val
                    
                    # Print debug information to help with troubleshooting
                    print(f"Sending results for test file: {test_file}")
                    try:
                        output_dir_for_log_preds = os.path.dirname(self.job_manager.get_job_folder()) # Gets 'output'
                        log_predictions_path = os.path.relpath(predictions_file, output_dir_for_log_preds)
                    except Exception:
                        log_predictions_path = predictions_file
                    print(f"Predictions file path: {log_predictions_path}")
                    print(f"Target column: {target_column_name}")
                    
                    # Send results to GUI for this specific test file
                    self.queue.put({'task_completed': gui_result_data})
                    with self.lock:
                        self.successful_tests += 1
                finally:
                    # Aggressively clean up memory after each file test for each model
                    if file_results is not None:
                        del file_results
                    if continuous_testing_service is not None:
                        del continuous_testing_service
                    if 'torch' in sys.modules:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    gc.collect()
                    self.logger.info(f"Cleaned up memory for task {task.get('task_id', 'UnknownTask')} on file {test_file}")

        except Exception as e:
            self.logger.error(f"CRITICAL ERROR in _test_single_model for task {task.get('task_id', 'UnknownTask')}: {e}")
            self.logger.error(traceback.format_exc())
            self.queue.put({'task_error': f"Critical error in task {task.get('task_id', 'UnknownTask')}: {e}"})

    @staticmethod
    def generate_shorthand_name(task):
        """Generate a shorthand name for the task based on hyperparameters."""
        hyperparams = task['hyperparams']
        model_type = hyperparams.get('MODEL_TYPE', 'LSTM')
        
        # Get common parameters
        batch_size = hyperparams.get('BATCH_SIZE', 'NA')
        max_epochs = hyperparams.get('MAX_EPOCHS', 'NA')
        lr_val = hyperparams.get('INITIAL_LR', 'NA')
        try:
            # Attempt to convert to float and format, otherwise use original string
            lr = f"{float(lr_val):.1e}"
        except (ValueError, TypeError):
            lr = lr_val
        lr_drop_period = hyperparams.get('LR_DROP_PERIOD', 'NA')
        valid_patience = hyperparams.get('VALID_PATIENCE', 'NA')
        valid_frequency = hyperparams.get('ValidFrequency', 'NA')
        lookback = hyperparams.get('LOOKBACK', 'NA')
        repetitions = hyperparams.get('REPETITIONS', 'NA')
        
        # Get model-specific parameters
        if model_type in ['LSTM', 'GRU']:
            layers = hyperparams.get('LAYERS', 'NA')
            hidden_units = hyperparams.get('HIDDEN_UNITS', 'NA')
            arch_suffix = f"L{layers}_H{hidden_units}"
        elif model_type == 'FNN':
            # For FNN, use hidden layer config as architecture identifier
            hidden_layers = hyperparams.get('HIDDEN_LAYER_SIZES', 'NA')
            # Create a short representation of the layer config
            if isinstance(hidden_layers, list):
                arch_suffix = f"FNN{'_'.join(map(str, hidden_layers))}"
            else:
                arch_suffix = f"FNN{str(hidden_layers).replace(',', '_')}"
        else:
            arch_suffix = f"{model_type}_NA"

        short_name = (f"{arch_suffix}_B{batch_size}_Lk{lookback}_"
                      f"E{max_epochs}_LR{lr}_LD{lr_drop_period}_VP{valid_patience}_"
                      f"VF{valid_frequency}_R{repetitions}")

        # Create parameter string for hash (include model-specific params)
        if model_type in ['LSTM', 'GRU']:
            param_string = f"{layers}_{hidden_units}_{batch_size}_{lookback}_{lr}_{valid_patience}_{max_epochs}"
        else:  # FNN
            param_string = f"{hidden_layers}_{batch_size}_{lookback}_{lr}_{valid_patience}_{max_epochs}"
            
        short_hash = hashlib.md5(param_string.encode()).hexdigest()[:3]  # First 3 chars for uniqueness
        shorthand_name = f"{short_name}_{short_hash}"
        return shorthand_name
    
    def get_results_summary(self):
        """Return the summary of results."""
        return self.results_summary
    
    def log_test_to_sqlite(self, task, results, db_log_file):
        """Log test results to SQLite database."""
        conn = sqlite3.connect(db_log_file)
        cursor = conn.cursor()

        # Create the test_logs table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_logs (
                task_id TEXT,
                file_name TEXT,
                model TEXT,
                rms_error REAL,
                mae REAL,
                max_error REAL,
                mape REAL,
                r2 REAL,
                PRIMARY KEY(task_id, file_name)
            )
        ''')

        # Insert test results with file name to track individual file results
        cursor.execute('''
            INSERT OR REPLACE INTO test_logs (task_id, file_name, model, rms_error, mae, max_error, mape, r2)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            task['task_id'], 
            results.get('file_name', 'unknown'), 
            task['model_path'], 
            results.get('rms_error_mv', results.get('rms_error_percent', results.get('rms_error_degC', 0))),
            results.get('mae_mv', results.get('mae_percent', results.get('mae_degC', 0))),
            results.get('max_abs_error_mv', results.get('max_abs_error_percent', results.get('max_abs_error_degC', 0))),
            results.get('r2', 0)
        ))

        conn.commit()
        conn.close()

    def log_test_to_csv(self, task, results, csv_log_file):
        """Log test results to CSV file."""
        fieldnames = ['Task ID', 'File Name', 'Model', 'RMS Error', 'MAE', 'Max Error', 'R2', 'Units']
        file_exists = os.path.isfile(csv_log_file)
        
        # Determine units based on task target
        target_column = task.get('data_loader_params', {}).get('target_column', '')
        units = 'mV' if 'voltage' in target_column.lower() else '%' if 'soc' in target_column.lower() else '°C' if ('temperature' in target_column.lower() or 'temp' in target_column.lower()) else ''

        with open(csv_log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()  # Write header only once
            
            writer.writerow({
                'Task ID': task['task_id'],
                'File Name': results.get('file_name', 'unknown'),
                'Model': task['model_path'],
                'RMS Error': results.get('rms_error_mv', results.get('rms_error_percent', results.get('rms_error_degC', 'N/A'))),
                'MAE': results.get('mae_mv', results.get('mae_percent', results.get('mae_degC', 'N/A'))),
                'Max Error': results.get('max_abs_error_mv', results.get('max_abs_error_percent', results.get('max_abs_error_degC', 'N/A'))),
                'R2': results.get('r2', 'N/A'),
                'Units': units
            })
