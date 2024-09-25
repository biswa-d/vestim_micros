#----------------------------------------------------------------------------------------
#Descrition: This file _1 is to implement the testing service without sequential data preparationfor testing the LSTM model
#
# Created On: Tue Sep 24 2024 16:51:14
# Author: Biswanath Dehury
# Company: Dr. Phil Kollmeyer's Battery Lab at McMaster University
# Copyright (c) 2024 Biswanath Dehury, Dr. Phil Kollmeyer's Battery Lab at McMaster University
#----------------------------------------------------------------------------------------

import torch
import os
import json, hashlib, sqlite3, csv
from threading import Thread
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed

from vestim.gateway.src.job_manager_qt import JobManager
from vestim.services.model_testing.src.testing_service_test_1 import VEstimTestingService
from vestim.services.model_testing.src.test_data_service_test_pouch_1 import VEstimTestDataService
from vestim.gateway.src.training_setup_manager_qt_test import VEstimTrainingSetupManager

class VEstimTestingManager:
    def __init__(self):
        print("Initializing VEstimTestingManager...")
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
            save_dir = self.job_manager.get_test_results_folder()
            print(f"Test folder: {test_folder}, Save directory: {save_dir}")

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
                    executor.submit(self._test_single_model, task, idx, test_folder, save_dir): task
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


    def _test_single_model(self, task, idx, test_folder, save_dir):
        """Test a single model and save the result."""
        try:
            print(f"Preparing test data for Task {idx + 1}...")

            # Extract lookback, learnable params and model path from the task
            lookback = task['hyperparams']['LOOKBACK']
            model_path = task['model_path']
            num_learnable_params = task['hyperparams']['NUM_LEARNABLE_PARAMS']
            print(f"Testing model: {model_path} with lookback: {lookback}")

            print("Loading and processing test data...")
            X_test, y_test = self.test_data_service.load_and_process_data(test_folder)

            print("Generating shorthand name for model...")
            shorthand_name = self.generate_shorthand_name(task)

            # Run the testing process and get results
            results = self.testing_service.run_testing(task, model_path, X_test, y_test, save_dir)

            # Log the test results to CSV and SQLite
            csv_log_file = task['csv_log_file']
            db_log_file = task['db_log_file']

            self.log_test_to_csv(task, results, csv_log_file)
            self.log_test_to_sqlite(task, results, db_log_file)

            self.testing_service.save_test_results(results, shorthand_name, save_dir)

            # Put the results in the queue for the GUI
            print(f"Results for model {shorthand_name}: {results}")
            self.queue.put({
                'task_completed': {
                    'sl_no': idx + 1,
                    'model': shorthand_name,
                    '#params': num_learnable_params,
                    'rms_error_mv': results['rms_error_mv'],
                    'mae_mv': results['mae_mv'],
                    'mape': results['mape'],
                    'r2': results['r2']
                }
            })

        except Exception as e:
            print(f"Error testing model {task['model_path']}: {str(e)}")
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


