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
            task_dir = task['task_dir']
            feature_cols = task['data_loader_params']['feature_columns']
            target_col = task['data_loader_params']['target_column']
            task_id = task['task_id']
            print(f"Testing model: {model_path} with lookback: {lookback}")

            # Create test_results directory
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

                # Save predictions with correct column names for plotting
                predictions_file = os.path.join(test_results_dir, f"{file_name}_predictions.csv")
                pd.DataFrame({
                    'True Values (V)': file_results['y_test'],
                    'Predictions (V)': file_results['predictions'],
                    'Error (mV)': errors * 1000  # Convert to mV
                }).to_csv(predictions_file, index=False)
                
                file_results['max_error_mv'] = max_error * 1000  # Convert to mV
                all_results.append(file_results)

            # Save test summary
            summary_file = os.path.join(task_dir, 'test_summary.csv')
            with open(summary_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['File', 'RMS Error (mV)', 'MAE (mV)', 'Max Error (mV)', 'MAPE (%)', 'R2'])
                for test_file, result in zip(test_files, all_results):
                    writer.writerow([
                        test_file,
                        result['rms_error_mv'],
                        result['mae_mv'],
                        result['max_error_mv'],
                        result['mape'],
                        result['r2']
                    ])

            # Calculate average metrics
            avg_results = {
                'rms_error_mv': np.mean([r['rms_error_mv'] for r in all_results]),
                'mae_mv': np.mean([r['mae_mv'] for r in all_results]),
                'max_error_mv': np.max([r['max_error_mv'] for r in all_results]),
                'mape': np.mean([r['mape'] for r in all_results]),
                'r2': np.mean([r['r2'] for r in all_results])
            }

            print("Generating shorthand name for model...")
            shorthand_name = self.generate_shorthand_name(task)

            # Put the results in the queue for the GUI
            print(f"Results for model {shorthand_name}: {avg_results}")
            self.queue.put({
                'task_completed': {
                    'saved_dir': test_results_dir,  # Directory containing predictions
                    'task_id': task_id,
                    'model': shorthand_name,
                    'file_name': test_file,  # Name of the test file
                    '#params': learnable_params,
                    'rms_error_mv': avg_results['rms_error_mv'],  # Already in mV
                    'max_error_mv': avg_results['max_error_mv'],  # Already in mV
                    'mape': avg_results['mape'],  # In percentage
                    'r2': avg_results['r2']  # R-squared value
                }
            })

        except Exception as e:
            print(f"Error testing model {model_path}: {str(e)}")
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


