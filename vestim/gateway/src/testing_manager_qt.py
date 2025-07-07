# ---------------------------------------------------------------------------------
# Author: Biswanath Dehury
# Date: `{{date:2023-03-02}}`
# Version: 1.0.0
# Description: Description of the script
#Descrition: This is the batchtesting without padding implementation for the unscaled data where the batch-size is used for testloader preparation but the model is tested
# one sequence at a time like a running window. The first part of the test file is padded with data to avoid the size mismatch and get the final prediction the same
# shape as the test file.
#
# Copyright (c) 2024 Biswanath Dehury, Dr. Phil Kollmeyer's Battery Lab at McMaster University
# ---------------------------------------------------------------------------------


import torch
import os
import json, hashlib, sqlite3, csv
from threading import Thread
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
    def __init__(self):
        print("Initializing VEstimTestingManager...")
        self.logger = logging.getLogger(__name__)
        self.job_manager = JobManager()  # Singleton instance of JobManager
        self.training_setup_manager = VEstimTrainingSetupManager()
        self.testing_service = VEstimTestingService()  # Keep old service for fallback
        self.continuous_testing_service = ContinuousTestingService()  # New continuous testing
        self.test_data_service = VEstimTestDataService()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_workers = 4  # Number of concurrent threads
        self.queue = None  # Initialize the queue attribute
        self.stop_flag = False  # Initialize the stop flag attribute
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
            # save_dir = self.job_manager.get_test_results_folder()
            # print(f"Test folder: {test_folder}, Save directory: {save_dir}")

            # Retrieve task list
            print("Retrieving task list from TrainingSetupManager...")
            task_list = self.training_setup_manager.get_task_list()

            if not task_list:
                task_summary_file = os.path.join(self.job_manager.get_job_folder(), 'training_tasks_summary.json')
                if os.path.exists(task_summary_file):
                    with open(task_summary_file, 'r') as f:
                        task_list = json.load(f)
                else:
                    raise ValueError("Task list is not available in memory or on disk.")

            print(f"Total tasks to run: {len(task_list)}")

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
                        print(f"Task {task} generated an exception: {exc}")
                        self.queue.put({'task_error': f'Task {task} generated an exception: {exc}'})

            # Signal to the queue that all tasks are completed
            print("All tasks completed. Sending signal to GUI...")
            self.queue.put({'all_tasks_completed': True})

        except Exception as e:
            print(f"An error occurred during testing: {str(e)}")
            self.queue.put({'task_error': str(e)})


    def _test_single_model(self, task, idx, test_folder):
        """Test a single model and save the result."""
        try:
            self.logger.info(f"--- Starting _test_single_model for task_id: {task.get('task_id', 'UnknownTask')} (list index: {idx}) ---") # Added detailed log
            print(f"Preparing test data for Task {idx + 1}...")
            
            # Reset continuous testing service for each new model
            self.continuous_testing_service.reset_for_new_model()
            
            # Get required paths and parameters
            lookback = task['hyperparams']['LOOKBACK']
            # model_path = task['model_path'] # This was pointing to the untrained template
            task_dir = task['task_dir'] # This is the specific directory for the task (e.g., .../task_XYZ_rep_1/)
            
            # Correct model_path should point to the best_model.pth within the task_dir
            # This path is set in task['training_params']['best_model_path'] by TrainingSetupManager
            # and used by TrainingTaskManager to save the best model.
            model_path = task.get('training_params', {}).get('best_model_path')

            if not model_path or not os.path.exists(model_path):
                self.logger.error(f"Best model path not found or file does not exist for task {task.get('task_id', 'UnknownTask')}: {model_path}. Skipping testing for this task.")
                self.queue.put({'task_error': f"Best model not found for task {task.get('task_id', 'UnknownTask')}"})
                return # Skip this task if its best model isn't available

            num_learnable_params = task['hyperparams']['NUM_LEARNABLE_PARAMS']
            
            # Make model_path relative for logging
            try:
                output_dir_for_log = os.path.dirname(self.job_manager.get_job_folder()) # Gets 'output'
                log_model_path = os.path.relpath(model_path, output_dir_for_log)
            except Exception: # Fallback if path manipulation fails
                log_model_path = model_path
            print(f"Testing model: {log_model_path} with lookback: {lookback}")
            
            # Create test_results directory within task directory
            test_results_dir = os.path.join(task_dir, 'test_results')
            os.makedirs(test_results_dir, exist_ok=True)

            # Get all test files
            test_files = [f for f in os.listdir(test_folder) if f.endswith('.csv')]
            if not test_files:
                print(f"No test files found in {test_folder}")
                return

            print(f"Found {len(test_files)} test files. Running tests...")
            
            # Process each test file
            for test_file_index, test_file in enumerate(test_files):
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
                    file_results = self.continuous_testing_service.run_continuous_testing(
                        task=task,
                        model_path=model_path,
                        test_file_path=test_file_path,
                        is_first_file=(test_file_index == 0),
                        warmup_samples=lookback_val  # Use lookback as warmup period
                    )
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
                
                if file_results is None: # Handle case where testing service run_testing fails
                    self.logger.error(f"Testing failed for {test_file_path}, skipping results processing for this file.")
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
                elif "soc" in target_column_name.lower():
                    unit_suffix = "_percent"
                    csv_unit_display = "(SOC)"   # Changed from (% SOC)
                    error_unit_display = "(% SOC)" # Stays as (% SOC) for the error column name
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
                elif "soc" in target_column_name.lower() and np.max(np.abs(y_true_scaled)) <= 1.0:  
                    # Check if SOC is in 0-1 range and needs percentage conversion
                    difference *= 100  # Convert 0-1 difference to percentage for CSV display
                
                # Save predictions with dynamic column names - matching training GUI conventions
                predictions_file = os.path.join(test_results_dir, f"{file_name}_predictions.csv")
                
                # Prepare data for DataFrame
                data_for_csv = {
                    f'True Value {csv_unit_display}': y_true_scaled, # Changed "Values" to "Value"
                    f'Predictions {csv_unit_display}': y_pred_scaled,
                }
                
                if "soc" in target_column_name.lower():
                    # Use the pre-calculated error from testing_service for SOC
                    data_for_csv[f'Error {error_unit_display}'] = file_results['error_percent_soc_values']
                else:
                    # Use the existing difference calculation for other types
                    data_for_csv[f'Error {error_unit_display}'] = difference
                
                pd.DataFrame(data_for_csv).to_csv(predictions_file, index=False)
                
                # Add results to summary file with dynamic headers matching training GUI
                # summary_file = os.path.join(task_dir, 'test_summary.csv')
                # header = ['File', f'RMS Error {error_unit_display}', f'MAE {error_unit_display}', f'Max Abs Error {error_unit_display}', f'MAPE (%)', 'R2']
                
                # Calculate max absolute error with appropriate scaling
                max_abs_error_val = np.max(np.abs(difference)) if difference.size > 0 else 0

                # write_header = not os.path.exists(summary_file) or os.path.getsize(summary_file) == 0
                # with open(summary_file, 'a', newline='') as f:
                #     writer = csv.writer(f)
                #     if write_header:
                #         writer.writerow(header)
                #     writer.writerow([
                #         test_file,
                #         f"{file_results.get(rms_key, float('nan')):.2f}",
                #         f"{file_results.get(mae_key, float('nan')):.2f}",
                #         f"{max_abs_error_val:.2f}", # Log calculated max_abs_error_val
                #         f"{file_results.get('mape_percent', float('nan')):.2f}",
                #         f"{file_results.get('r2', float('nan')):.4f}"
                #     ])
                
                # Generate shorthand name for the model task
                shorthand_name = self.generate_shorthand_name(task)
                
                # Data to send to GUI for this specific test file
                gui_result_data = {
                    'saved_dir': test_results_dir,
                    'task_id': task['task_id'],
                    'sl_no': f"{idx + 1}.{test_file_index + 1}", # Unique Sl.No for GUI based on task and test file
                    'model': shorthand_name,
                    'file_name': test_file, # Current test file name
                    '#params': num_learnable_params,
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
                gui_result_data['mape_percent'] = file_results.get('mape_percent', 'N/A')
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

            # The overall task completion signal (all_tasks_completed) is sent after the outer loop in _run_testing_tasks

        except Exception as e:
            try:
                output_dir_for_log_err = os.path.dirname(self.job_manager.get_job_folder())
                log_model_path_err = os.path.relpath(model_path, output_dir_for_log_err)
            except Exception:
                log_model_path_err = model_path
            print(f"Error testing model {log_model_path_err}: {str(e)}")
            self.logger.error(f"Error testing model {log_model_path_err}: {str(e)}", exc_info=True)
            self.queue.put({'task_error': str(e)})

    @staticmethod
    def generate_shorthand_name(task):
        """Generate a shorthand name for the task based on hyperparameters."""
        hyperparams = task['hyperparams']
        model_type = hyperparams.get('MODEL_TYPE', 'LSTM')
        
        # Get common parameters
        batch_size = hyperparams.get('BATCH_SIZE', 'NA')
        max_epochs = hyperparams.get('MAX_EPOCHS', 'NA')
        lr = hyperparams.get('INITIAL_LR', 'NA')
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
            results.get('mape_percent', 0),
            results.get('r2', 0)
        ))

        conn.commit()
        conn.close()

    def log_test_to_csv(self, task, results, csv_log_file):
        """Log test results to CSV file."""
        fieldnames = ['Task ID', 'File Name', 'Model', 'RMS Error', 'MAE', 'Max Error', 'MAPE', 'R2', 'Units']
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
                'MAPE': results.get('mape_percent', 'N/A'),
                'R2': results.get('r2', 'N/A'),
                'Units': units
            })
