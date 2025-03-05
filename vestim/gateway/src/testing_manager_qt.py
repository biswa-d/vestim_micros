import torch
import os
import numpy as np
import pandas as pd
import json, hashlib, sqlite3, csv
from threading import Thread
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed

from vestim.gateway.src.job_manager_qt import JobManager
from vestim.services.model_testing.src.testing_service import VEstimTestingService
from vestim.services.model_testing.src.test_data_service import VEstimTestDataService
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
        """Test a single model on all test files in the folder and save individual + averaged results."""
        try:
            print(f"Preparing test data for Task {idx + 1}...")

            # Extract lookback, batch size, and model path from the task
            learnable_params = task['hyperparams']['NUM_LEARNABLE_PARAMS']
            lookback = task['hyperparams']['LOOKBACK']
            batch_size = task['hyperparams']['BATCH_SIZE']
            model_path = task['model_path']
            save_dir = task['task_dir']
            feature_cols = task['data_loader_params']['feature_columns']
            target_col = task['data_loader_params']['target_column']
            task_id = task['task_id']
            print(f"Testing model: {model_path} with lookback: {lookback}")

            print("Generating shorthand name for model...")
            shorthand_name = self.generate_shorthand_name(task)

            # Get all test files in the test folder
            test_files = [f for f in os.listdir(test_folder) if f.endswith('.csv')]  # Adjust extension if needed
            if not test_files:
                print(f"No test files found in {test_folder}")
                return
            
            print(f"Found {len(test_files)} test files. Running tests...")

            # Initialize lists to store metrics for averaging
            rms_errors, mae_errors, mape_errors, r2_scores = [], [], [], []

            for test_file in test_files:
                test_file_path = os.path.join(test_folder, test_file)
                print(f"Processing test file: {test_file}")

                # Load and process test data for the specific file
                test_file_loader = self.test_data_service.create_test_file_loader(test_file_path, lookback, batch_size, feature_cols, target_col)

                # Run the testing process
                results = self.testing_service.run_testing(task, model_path, test_file_loader, test_file_path)

                # Log the test results to CSV and SQLite
                csv_log_file = task['csv_log_file']
                db_log_file = task['db_log_file']

                self.log_test_to_csv(task, results, csv_log_file)
                self.log_test_to_sqlite(task, results, db_log_file)

                # Save individual test results
                self.save_test_results(results, save_dir, test_file_path)

                # Store error metrics for averaging later
                rms_errors.append(results['rms_error_mv'])
                mae_errors.append(results['mae_mv'])
                mape_errors.append(results['mape'])
                r2_scores.append(results['r2'])

            # Compute the average metrics across all test files
            avg_rms_error = np.mean(rms_errors) if rms_errors else None
            avg_mae = np.mean(mae_errors) if mae_errors else None
            avg_mape = np.mean(mape_errors) if mape_errors else None
            avg_r2 = np.mean(r2_scores) if r2_scores else None

            print(f"Averaged results for model {shorthand_name}: RMS Error = {avg_rms_error}, MAE = {avg_mae}, MAPE = {avg_mape}, R² = {avg_r2}")

            # Put the averaged results in the queue for the GUI
            self.queue.put({
                'task_completed': {
                    'sl_no': idx + 1,
                    'model': shorthand_name,
                    'avg_rms_error_mv': avg_rms_error,
                    'avg_mae_mv': avg_mae,
                    'avg_mape': avg_mape,
                    'avg_r2': avg_r2,
                    '#params': learnable_params,
                    'saved_dir': save_dir
                }
            })

        except Exception as e:
            print(f"Error testing model {task['model_path']}: {str(e)}")
            self.queue.put({'task_error': str(e)})

    def save_test_results(self, results, save_dir, test_file_path):
        """
        Saves the test results for each test file to a model-specific subdirectory.

        :param results: Dictionary containing predictions, true values, and metrics.
        :param model_name: Name of the model (or model file) to label the results.
        :param save_dir: Directory where the results will be saved.
        :param test_file_path: The path of the test file used for generating predictions.
        """

        # Extract the test file name without extension for unique result filenames
        test_file_name = os.path.splitext(os.path.basename(test_file_path))[0]

        # Create a DataFrame to store the predictions and true values
        df = pd.DataFrame({
            'True Values (V)': results['true_values'],
            'Predictions (V)': results['predictions'],
            'Difference (mV)': (results['true_values'] - results['predictions']) * 1000  # Difference in mV
        })

        # Save the DataFrame as a CSV file named after the test file
        prediction_file = os.path.join(save_dir, f"{test_file_name}_predictions.csv")
        df.to_csv(prediction_file, index=False)

        # Save the metrics separately for the test file
        metrics_file = os.path.join(save_dir, f"{test_file_name}_metrics.txt")
        with open(metrics_file, 'w') as f:
            f.write(f"Test File: {test_file_name}\n")
            f.write(f"RMS Error (mV): {results['rms_error_mv']:.2f}\n")
            f.write(f"MAE (mV): {results['mae_mv']:.2f}\n")
            f.write(f"MAPE (%): {results['mape']:.2f}\n")
            f.write(f"R²: {results['r2']:.4f}\n")

        print(f"Results and metrics for test file '{test_file_name}' saved in {save_dir}")

    
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


