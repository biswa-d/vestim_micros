import torch
import os
import json, hashlib
from threading import Thread
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.gateway.src.job_manager import JobManager
from src.services.model_testing.src.testing_service import VEstimTestingService
from src.services.model_testing.src.test_data_service import VEstimTestDataService
from src.gateway.src.training_setup_manager import VEstimTrainingSetupManager

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
            # Initialize task tracking
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

                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        future.result()  # Retrieve the result
                    except Exception as exc:
                        print(f"Task {task} generated an exception: {exc}")
                        self.queue.put({'task_error': f'Task {task} generated an exception: {exc}'})

        except Exception as e:
            print(f"An error occurred during testing: {str(e)}")
            self.queue.put({'task_error': str(e)})

    def _test_single_model(self, task, idx, test_folder, save_dir):
        """Test a single model and save the result."""
        try:
            print(f"Preparing test data for Task {idx + 1}...")

            # Extract lookback and model path from the task
            lookback = task['hyperparams']['LOOKBACK']
            model_path = task['model_path']
            print(f"Testing model: {model_path} with lookback: {lookback}")

            print("Loading and processing test data...")
            X_test, y_test = self.test_data_service.load_and_process_data(test_folder, lookback)

            print("Generating shorthand name for model...")
            shorthand_name = self.generate_shorthand_name(task)

            # Run the testing process
            results = self.testing_service.run_testing(task, model_path, X_test, y_test, save_dir)
            self.testing_service.save_test_results(results, shorthand_name, save_dir)

            # Put the results in the queue for the GUI
            print(f"Results for model {shorthand_name}: {results}")
            self.queue.put({
                'task_completed': {
                    'sl_no': idx + 1,
                    'model': shorthand_name,
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

        short_name = (f"Lr{layers}_H{hidden_units}_B{batch_size}_Lk{lookback}_"
                      f"E{max_epochs}_LR{lr}_LDR{lr_drop_period}_ValP{valid_patience}_"
                      f"ValFr{valid_frequency}_Rep{repetitions}")

        param_string = f"{layers}_{hidden_units}_{batch_size}_{lookback}_{lr}_{valid_patience}_{max_epochs}"
        short_hash = hashlib.md5(param_string.encode()).hexdigest()[:6]  # First 6 chars for uniqueness
        shorthand_name = f"{short_name}_{short_hash}"
        return shorthand_name
