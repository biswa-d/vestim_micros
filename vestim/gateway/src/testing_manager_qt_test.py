# ---------------------------------------------------------------------------------
# Author: Biswanath Dehury
# Date: `{{date:2023-03-02}}`
# Version: 1.0.0
# Description: Description of the script
#Descrition: This is the batchtesting without padding implementation for the unscaled data where the batch-size is used for testloader preparation but the model is tested
# one sequence at a time like a running window. The first part of the test file is padded with data to avoid the size mismatch and get the final prediction the same
# shape as the test file.

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
from vestim.services.model_testing.src.testing_service_test import VEstimTestingService
from vestim.services.model_testing.src.test_data_service_test_digatron_csv import VEstimTestDataService
from vestim.gateway.src.training_setup_manager_qt_test import VEstimTrainingSetupManager
import logging

class VEstimTestingManager:
    def __init__(self):
        print("Initializing VEstimTestingManager...")
        self.logger = logging.getLogger(__name__)
        self.job_manager = JobManager()  # Singleton instance of JobManager
        self.training_setup_manager = VEstimTrainingSetupManager()
        self.testing_service = VEstimTestingService()
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
            print(f"Preparing test data for Task {idx + 1}...")
            
            # Get required paths and parameters
            lookback = task['hyperparams']['LOOKBACK']
            model_path = task['model_path']
            task_dir = task['task_dir']
            num_learnable_params = task['hyperparams']['NUM_LEARNABLE_PARAMS']
            
            print(f"Testing model: {model_path} with lookback: {lookback}")
            
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
            all_results = []
            for test_file in test_files:
                file_name = os.path.splitext(test_file)[0]
                test_file_path = os.path.join(test_folder, test_file)
                
                # Run test for single file
                file_results = self.testing_service.test_single_file(
                    task, 
                    model_path, 
                    test_file_path
                )
                
                # Calculate max error
                errors = np.abs(file_results['y_test'] - file_results['predictions'])
                max_error = np.max(errors)
                file_results['max_error_mv'] = max_error
                
                # Save predictions with correct column names for plotting
                predictions_file = os.path.join(test_results_dir, f"{file_name}_predictions.csv")
                pd.DataFrame({
                    'True Values (V)': file_results['y_test'],
                    'Predictions (V)': file_results['predictions'],
                    'Error (mV)': errors
                }).to_csv(predictions_file, index=False)
                
                # Add results to summary file
                summary_file = os.path.join(task_dir, 'test_summary.csv')
                if not os.path.exists(summary_file):
                    with open(summary_file, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['File', 'RMS Error (mV)', 'MAE (mV)', 'Max Error (mV)', 'MAPE (%)', 'R2'])
                
                with open(summary_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        test_file,
                        file_results['rms_error_mv'],
                        file_results['mae_mv'],
                        max_error,
                        file_results['mape'],
                        file_results['r2']
                    ])
                
                all_results.append(file_results)

            # Calculate average metrics across all files
            avg_results = {
                'rms_error_mv': np.mean([r['rms_error_mv'] for r in all_results]),
                'mae_mv': np.mean([r['mae_mv'] for r in all_results]),
                'max_error_mv': np.max([r['max_error_mv'] for r in all_results]),
                'mape': np.mean([r['mape'] for r in all_results]),
                'r2': np.mean([r['r2'] for r in all_results])
            }

            # Generate shorthand name
            shorthand_name = self.generate_shorthand_name(task)
            
            # Put the results in the queue for the GUI
            self.queue.put({
                'task_completed': {
                    'saved_dir': test_results_dir,  # Changed to test_results_dir for plotting
                    'task_id': task['task_id'],
                    'sl_no': idx + 1,
                    'model': shorthand_name,
                    'file_name': test_file,  # Add filename to display
                    '#params': num_learnable_params,
                    'rms_error_mv': avg_results['rms_error_mv'],
                    'mae_mv': avg_results['mae_mv'],
                    'max_error_mv': avg_results['max_error_mv'],
                    'mape': avg_results['mape'],
                    'r2': avg_results['r2']
                }
            })

        except Exception as e:
            print(f"Error testing model {model_path}: {str(e)}")
            self.queue.put({'task_error': str(e)})


    @staticmethod
    def generate_shorthand_name(task):
        """Generate a shorthand name for the task based on hyperparameters."""
        hyperparams = task['hyperparams']
        layers = hyperparams.get('LAYERS', 'NA')
        hidden_units = hyperparams.get('HIDDEN_UNITS', 'NA')
        batch_size = hyperparams.get('BATCH_SIZE', 'NA')
        max_epochs = hyperparams.get('MAX_EPOCHS', 'NA')
        lr = hyperparams.get('INITIAL_LR', 'NA')
        lr_drop_period = hyperparams.get('LR_DROP_PERIOD', 'NA')
        valid_patience = hyperparams.get('VALID_PATIENCE', 'NA')
        valid_frequency = hyperparams.get('ValidFrequency', 'NA')
        lookback = hyperparams.get('LOOKBACK', 'NA')
        repetitions = hyperparams.get('REPETITIONS', 'NA')

        short_name = (f"L{layers}_H{hidden_units}_B{batch_size}_Lk{lookback}_"
                      f"E{max_epochs}_LR{lr}_LD{lr_drop_period}_VP{valid_patience}_"
                      f"VF{valid_frequency}_R{repetitions}")

        param_string = f"{layers}_{hidden_units}_{batch_size}_{lookback}_{lr}_{valid_patience}_{max_epochs}"
        short_hash = hashlib.md5(param_string.encode()).hexdigest()[:3]  # First 6 chars for uniqueness
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
                model TEXT,
                rms_error_mv REAL,
                mae_mv REAL,
                mape REAL,
                r2 REAL,
                PRIMARY KEY(task_id)
            )
        ''')

        # Insert test results
        cursor.execute('''
            INSERT OR REPLACE INTO test_logs (task_id, model, rms_error_mv, mae_mv, mape, r2)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (task['task_id'], task['model_path'], results['rms_error_mv'], results['mae_mv'], results['mape'], results['r2']))

        conn.commit()
        conn.close()

    def log_test_to_csv(self, task, results, csv_log_file):
        """Log test results to CSV file."""
        fieldnames = ['Task ID', 'Model', 'RMS Error (mV)', 'MAE (mV)', 'MAPE', 'R2']
        file_exists = os.path.isfile(csv_log_file)

        with open(csv_log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()  # Write header only once
            writer.writerow({
                'Task ID': task['task_id'],
                'Model': task['model_path'],
                'RMS Error (mV)': results['rms_error_mv'],
                'MAE (mV)': results['mae_mv'],
                'MAPE': results['mape'],
                'R2': results['r2']
            })