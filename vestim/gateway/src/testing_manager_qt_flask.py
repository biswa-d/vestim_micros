from flask import Flask, jsonify, Blueprint, request
import torch, requests
import os, json, hashlib, time
from threading import Thread
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from vestim.services.model_testing.src.testing_service_test import VEstimTestingService
from vestim.services.model_testing.src.test_data_service_test_pouch import VEstimTestDataService


# Flask Blueprint for Testing Manager
testing_manager_blueprint = Blueprint('testing_manager', __name__)

class VEstimTestingManager:
    def __init__(self):
        print("Initializing VEstimTestingManager...")
        self.testing_service = VEstimTestingService()
        self.test_data_service = VEstimTestDataService()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_workers = 4  # Number of concurrent threads
        self.testing_thread = None  # Initialize the thread attribute
        self.queue = None  # Initialize the queue attribute
        self.stop_flag = False  # Initialize the stop flag attribute
        print("Initialization complete.")

        self.task_list = None
        self.test_folder = None
        self.save_dir = None
        self.job_folder = None
    
    def fetch_job_folder(self):
        """Fetches and stores the job folder from the Job Manager API."""
        if self.job_folder is None:
            try:
                response_job = requests.get("http://localhost:5000/job_manager/get_job_folder")
                if response_job.status_code == 200:
                    self.job_folder = response_job.json()['job_folder']
                else:
                    raise Exception("Failed to fetch job folder")
            except Exception as e:
                self.logger.error(f"Error fetching job folder: {str(e)}")
                raise e
        
    def fetch_test_folder(self):
        """Fetches and stores the test folder from the Job Manager API."""
        if not hasattr(self, 'test_folder') or self.test_folder is None:
            try:
                response = requests.get("http://localhost:5000/job_manager/get_test_folder")
                if response.status_code == 200:
                    self.test_folder = response.json()['test_folder']
                else:
                    raise Exception("Failed to fetch test folder")
            except Exception as e:
                self.logger.error(f"Error fetching test folder: {str(e)}")
                raise e
        return self.test_folder

    def fetch_save_dir(self):
        """Fetches and stores the test results save directory from the Job Manager API."""
        if not hasattr(self, 'save_dir') or self.save_dir is None:
            try:
                response = requests.get("http://localhost:5000/job_manager/get_test_results_folder")
                if response.status_code == 200:
                    self.save_dir = response.json()['test_results_folder']
                else:
                    raise Exception("Failed to fetch test results folder")
            except Exception as e:
                self.logger.error(f"Error fetching test results folder: {str(e)}")
                raise e
        return self.save_dir

    def fetch_task_list(self):
        """Fetches and stores the task list from the Training Setup Manager API."""
        if not hasattr(self, 'task_list') or self.task_list is None:
            try:
                response = requests.get("http://localhost:5000/training_setup/get_tasks")
                if response.status_code == 200:
                    self.task_list = response.json()
                else:
                    raise Exception("Failed to fetch task list")
            except Exception as e:
                self.logger.error(f"Error fetching task list: {str(e)}")
                raise e
        return self.task_list
    

    def start_testing(self, queue):
        """Starts the testing process."""
        print("Starting testing process...")
        self.queue = queue  # Store the queue instance
        self.stop_flag = False  # Reset stop flag when starting testing
        print("Creating testing thread...")
        self.testing_thread = Thread(target=self._run_testing_tasks, args=(queue,))
        self.testing_thread.setDaemon(True)
        self.testing_thread.start()
        print("Testing thread started.")

    def _run_testing_tasks(self, queue):
        """Run testing tasks sequentially."""
        try:
            print("Getting test folder and results save directory...")
            self.fetch_task_list()
            self.fetch_test_folder()
            self.fetch_save_dir()
            self.fetch_job_folder()

            # Load task list from memory or file
            if not self.task_list:
                task_summary_file = os.path.join(self.job_folder, 'training_tasks_summary.json')
                if os.path.exists(task_summary_file):
                    with open(task_summary_file, 'r') as f:
                        self.task_list = json.load(f)

            if not self.task_list:
                raise ValueError("Task list is not available in memory or on disk.")

            # Initialize a queue to manage task status
            status_queue = Queue()

            # Process tasks with thread pool
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_task = {
                    executor.submit(self._test_single_model, task, idx, self.test_folder, self.save_dir, queue, status_queue): task
                    for idx, task in enumerate(self.task_list)
                }

                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        future.result()
                    except Exception as exc:
                        queue.put({'task_error': f'Task {task} generated an exception: {exc}'})

            status_queue.put(None)  # Signal the end of status updates
            status_queue.join()

            if not self.stop_flag:
                queue.put({'status': 'Testing completed!'})

        except Exception as e:
            queue.put({'task_error': str(e)})


    def _test_single_model(self, task, idx, test_folder, save_dir, queue, status_queue):
        """Test a single model and update the queue with the status."""
        try:
            status_queue.put(f'Preparing test data for Task {idx + 1}...')
            lookback = task['hyperparams']['LOOKBACK']
            model_path = task['model_path']

            # Load and process test data
            X_test, y_test = self.test_data_service.load_and_process_data(test_folder, lookback)
            shorthand_name = self.generate_shorthand_name(task)
            status_queue.put(f'Testing model: {shorthand_name}')

            # Run testing and save results
            results = self.testing_service.run_testing(task, model_path, X_test, y_test, save_dir)
            self.testing_service.save_test_results(results, shorthand_name, save_dir)

            # Update the queue with completed task status
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


    def get_current_task_status(self):
        current_time = time.time()
        elapsed_time = current_time - self.start_time if hasattr(self, 'start_time') else 0
        elapsed_hms = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

        # Add the number of completed tasks and results
        completed_tasks = len([task for task in self.task_list if task.get('status') == 'completed'])
        results = [{'model': task.get('model_path'), 'result': task.get('result')} for task in self.task_list if task.get('status') == 'completed']

        # Task status to be returned in the API response
        task_status = {
            "current_task_index": self.task_index,
            "completed_tasks": completed_tasks,
            "total_tasks": len(self.task_list),
            "results": results,
            "progress_data": self.current_task_status.get('progress_data', {}),
            "status": self.current_task_status.get('status', 'running'),
            "elapsed_time": elapsed_hms
        }
        return task_status

    @staticmethod
    def generate_shorthand_name(task):
        hyperparams = task['hyperparams']
        layers = hyperparams.get('LAYERS', 'NA')
        hidden_units = hyperparams.get('HIDDEN_UNITS', 'NA')
        batch_size = hyperparams.get('BATCH_SIZE', 'NA')
        max_epochs = hyperparams.get('MAX_EPOCHS', 'NA')
        lr = hyperparams.get('INITIAL_LR', 'NA')
        param_string = f"{layers}_{hidden_units}_{batch_size}_{lr}_{max_epochs}"
        short_hash = hashlib.md5(param_string.encode()).hexdigest()[:6]
        shorthand_name = f"Lr{layers}_H{hidden_units}_B{batch_size}_{short_hash}"
        return shorthand_name

    def stop_testing(self):
        self.stop_flag = True
        if self.testing_thread is not None and self.testing_thread.is_alive():
            self.testing_thread.join(timeout=5)

# Singleton instance of the VEstimTestingManager
testing_manager = VEstimTestingManager()

# Flask Endpoints
@testing_manager_blueprint.route('/start_testing', methods=['POST'])
def start_testing():
    queue = Queue()
    try:
        testing_manager.start_testing(queue, lambda status: print(status))
        return jsonify({"message": "Testing started successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@testing_manager_blueprint.route('/stop_testing', methods=['POST'])
def stop_testing():
    try:
        testing_manager.stop_testing()
        return jsonify({"message": "Testing stopped successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@testing_manager_blueprint.route('/task_status', methods=['GET'])
def task_status():
    try:
        # Mock-up status retrieval (you can add a real method here)
        status = {"status": "running"}
        return jsonify(status), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
