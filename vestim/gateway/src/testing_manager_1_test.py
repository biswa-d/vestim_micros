import torch
import os
import json
from threading import Thread
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed

from vestim.gateway.src.job_manager import JobManager
from vestim.services.model_testing.src.testing_service_test import VEstimTestingService
from vestim.services.model_testing.src.test_data_service_test import VEstimTestDataService
from vestim.gateway.src.training_setup_manager_test import VEstimTrainingSetupManager

class VEstimTestingManager:
    def __init__(self):
        print("Initializing VEstimTestingManager...")
        self.job_manager = JobManager()  # Singleton instance of JobManager
        self.training_setup_manager = VEstimTrainingSetupManager()
        self.testing_service = VEstimTestingService()
        self.test_data_service = VEstimTestDataService()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_workers = 4  # Number of concurrent threads
        self.testing_thread = None  # Initialize the thread attribute
        self.queue = None  # Initialize the queue attribute
        self.stop_flag = False  # Initialize the stop flag attribute
        print("Initialization complete.")

    def start_testing(self, queue, update_progress_callback):
        print("Starting testing process...")
        self.queue = queue  # Store the queue instance
        self.stop_flag = False  # Reset stop flag when starting testing
        print("Creating testing thread...")
        self.testing_thread = Thread(target=self._run_testing_tasks, args=(queue, update_progress_callback))
        self.testing_thread.setDaemon(True)
        self.testing_thread.start()
        print("Testing thread started.")

    def _run_testing_tasks(self, queue, update_progress_callback):
        try:
            print("Getting test folder and results save directory...")
            test_folder = self.job_manager.get_test_folder()
            save_dir = self.job_manager.get_test_results_folder()
            print(f"Test folder: {test_folder}, Save directory: {save_dir}")

            print("Retrieving task list from TrainingSetupManager...")
            task_list = self.training_setup_manager.get_task_list()
            print(f"Retrieved task list: {task_list}")

            if not task_list:
                print("No task list in memory. Attempting to load from JSON file...")
                task_summary_file = os.path.join(self.job_manager.get_job_folder(), 'training_tasks_summary.json')
                if os.path.exists(task_summary_file):
                    print(f"Loading task list from {task_summary_file}...")
                    with open(task_summary_file, 'r') as f:
                        task_list = json.load(f)
                    print("Task list loaded from JSON file.")
                else:
                    print("Task summary file does not exist.")

            if not task_list:
                raise ValueError("Task list is not available in memory or on disk.")
            
            print(f"Total tasks to run: {len(task_list)}")
            # Create a status update queue
            status_queue = Queue()

            def status_updater():
                while True:
                    status_message = status_queue.get()
                    if status_message is None or self.stop_flag:
                        break  # Exit loop if None is received or stop flag is set
                    print(f"Status Update: {status_message}")
                    update_progress_callback({'status': status_message})
                    status_queue.task_done()

            print("Starting status updater thread...")
            status_thread = Thread(target=status_updater)
            status_thread.start()
            print("Status updater thread started.")

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                print("Submitting tasks for testing...")
                future_to_task = {
                    executor.submit(self._test_single_model, task, idx, test_folder, save_dir, queue, status_queue): task
                    for idx, task in enumerate(task_list)
                }

                for future in as_completed(future_to_task):
                    if self.stop_flag:
                        print("Stop flag detected. Exiting testing loop.")
                        break
                    task = future_to_task[future]
                    try:
                        future.result()  # Retrieve the result, raises exception if occurred in thread
                    except Exception as exc:
                        print(f"Task {task} generated an exception: {exc}")
                        queue.put({'task_error': f'Task {task} generated an exception: {exc}'})

            print("All tasks submitted. Waiting for completion...")
            status_queue.put(None)  # Signal the status updater to stop
            status_queue.join()  # Ensure all status messages have been processed
            
            if not self.stop_flag:
                print("Testing completed successfully.")
                update_progress_callback({'status': 'Testing completed!'})

        except Exception as e:
            print(f"An error occurred during testing: {str(e)}")
            queue.put({'task_error': str(e)})


    def _test_single_model(self, task, idx, test_folder, save_dir, queue, status_queue):
        try:
            if self.stop_flag:
                print("Stop flag is set. Exiting test early.")
                return  # Exit early if stop is requested
            print(f"Preparing test data for Task {idx + 1}...")

            status_queue.put(f'Preparing test data for Task {idx + 1}...')

            if self.stop_flag:
                print("Stop flag is set. Exiting test early after data preparation.")
                return  # Exit early if stop is requested
            
            # Extract lookback and model path from the task
            lookback = task['hyperparams']['LOOKBACK']
            model_path = task['model_path']
            print(f"Testing model: {model_path} with lookback: {lookback}")

            print("Loading and processing test data...")
            X_test, y_test = self.test_data_service.load_and_process_data(test_folder, lookback)

            print("Generating shorthand name for model...")
            shorthand_name = self.generate_shorthand_name(model_path)

            print(f"Running testing on model: {shorthand_name}...")
            status_queue.put(f'Testing model: {shorthand_name}')
            
            # Run testing and save results
            results = self.testing_service.run_testing(task, model_path, X_test, y_test, save_dir)
            self.testing_service.save_test_results(results, shorthand_name, save_dir)

            # Continue to check stop_flag during long-running operations
            if self.stop_flag:
                print("Stop flag is set. Exiting test early after running the test.")
                return  # Exit early if stop is requested

            # Update the queue with the test results for the current task
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
            queue.put({'task_error': str(e)})


    def generate_shorthand_name(self, model_path):
        # Split the model path into components
        path_parts = model_path.split(os.sep)

        # Extract relevant directory names
        model_dir = path_parts[-2]  # Directory containing model.pth
        parent_dir = path_parts[-3]  # Parent directory of model_dir

        # Parse key parameters from the directory names
        params = model_dir.split('_')
        shorthand_name = f"{params[1]}_hu_{params[2]}..look_{params[-2]}..bat_{params[-1]}"

        print(f"Generated shorthand name: {shorthand_name}")
        return shorthand_name



if __name__ == "__main__":
    queue = Queue()
    testing_manager = VEstimTestingManager()
    testing_manager.start_testing(queue, lambda x: print(x))
