import torch
import os
import json
from threading import Thread
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed

from job_manager import JobManager
from src.services.model_testing.src.testing_service_test import VEstimTestingService
from src.services.model_testing.src.test_data_service_test import VEstimTestDataService
from src.gateway.src.training_setup_manager_test import VEstimTrainingSetupManager


class VEstimTestingManager:
    def __init__(self):
        self.job_manager = JobManager()  # Singleton instance of JobManager
        self.training_setup_manager = VEstimTrainingSetupManager()
        self.testing_service = VEstimTestingService()
        self.test_data_service = VEstimTestDataService()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_workers = 4  # Number of concurrent threads

    def start_testing(self, queue, update_progress_callback):
        # Start the testing process in a separate thread
        testing_thread = Thread(target=self._run_testing_tasks, args=(queue, update_progress_callback))
        testing_thread.setDaemon(True)
        testing_thread.start()

    def _run_testing_tasks(self, queue, update_progress_callback):
        try:
            # Get test folder and results save directory from JobManager
            test_folder = self.job_manager.get_test_folder()
            save_dir = self.job_manager.get_test_results_folder()

            # Try to get the task list from TrainingSetupManager
            task_list = self.training_setup_manager.get_task_list()

            if not task_list:
                # If task_list is not available in memory, load it from the saved JSON file
                task_summary_file = os.path.join(self.job_manager.get_job_folder(), 'training_tasks_summary.json')
                if os.path.exists(task_summary_file):
                    with open(task_summary_file, 'r') as f:
                        task_list = json.load(f)

            if not task_list:
                raise ValueError("Task list is not available in memory or on disk.")
            
            # Create a status update queue
            status_queue = Queue()

            def status_updater():
                while True:
                    status_message = status_queue.get()
                    if status_message is None:
                        break  # Exit loop if None is received
                    update_progress_callback({'status': status_message})
                    status_queue.task_done()

            # Start a thread to handle status updates
            status_thread = Thread(target=status_updater)
            status_thread.start()

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_task = {
                    executor.submit(self._test_single_model, task, idx, test_folder, save_dir, queue, update_progress_callback): task
                    for idx, task in enumerate(task_list)
                }

                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        future.result()  # Retrieve the result, raises exception if occurred in thread
                    except Exception as exc:
                        queue.put({'task_error': f'Task {task} generated an exception: {exc}'})
            # Once all tasks are done, signal the status updater to stop
            status_queue.put(None)
            status_queue.join()  # Ensure all status messages have been processed
            
            # Once all tasks are done
            update_progress_callback({'status': 'Testing completed!'})

        except Exception as e:
            queue.put({'task_error': str(e)})

    def _test_single_model(self, task, idx, test_folder, save_dir, queue, status_queue):
        try:
            # Update progress for preparing test data
            status_queue.put(f'Preparing test data for Task {idx + 1}...')

            # Extract lookback and model path from the task
            lookback = task['hyperparams']['LOOKBACK']
            model_path = task['model_path']

            # Load and process the test data with the extracted lookback
            X_test, y_test = self.testing_service.load_and_process_data(test_folder, lookback)

            # Generate shorthand name from model path
            shorthand_name = self.generate_shorthand_name(model_path)

            # Update progress for testing model
            status_queue.put(f'Testing model: {shorthand_name}')

            # Run testing and save results
            results = self.testing_service.run_testing(model_path, X_test, y_test, save_dir)
            self.testing_service.save_test_results(results, shorthand_name, save_dir)

            # Update the queue with the test results for the current task
            queue.put({
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
            queue.put({'task_error': str(e)})

    def generate_shorthand_name(self, model_path):
        """
        Generates a shorthand version of the model name based on the directory structure.

        :param model_path: Full path to the model.pth file.
        :return: Shorthand version of the model name.
        """
        # Split the model path into components
        path_parts = model_path.split(os.sep)

        # Extract relevant directory names
        model_dir = path_parts[-2]  # Directory containing model.pth
        parent_dir = path_parts[-3]  # Parent directory of model_dir

        # Parse key parameters from the directory names
        params = model_dir.split('_')
        shorthand_name = f"{params[1]}_hu_{params[2]}..look_{params[-2]}..bat_{params[-1]}"

        return shorthand_name


if __name__ == "__main__":
    queue = Queue()
    testing_manager = VEstimTestingManager()
    testing_manager.start_testing(queue, lambda x: print(x))
