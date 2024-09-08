import torch
import os
import json, hashlib
from PyQt5.QtCore import QThread, pyqtSignal
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.gateway.src.job_manager import JobManager
from src.services.model_testing.src.testing_service import VEstimTestingService
from src.services.model_testing.src.test_data_service_test import VEstimTestDataService
from src.gateway.src.training_setup_manager_qt import VEstimTrainingSetupManager


class VEstimTestingManager(QThread):
    def __init__(self):
        super().__init__()
        self.job_manager = JobManager()
        self.training_setup_manager = VEstimTrainingSetupManager()
        self.testing_service = VEstimTestingService()
        self.test_data_service = VEstimTestDataService()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_workers = 4  # Max number of threads
        self.stop_flag = False
        self.queue = Queue()

    def run(self):
        """Main method that starts the testing process when the thread is started."""
        self.start_testing()


    def start_testing(self):
        self.stop_flag = False
        test_folder = self.job_manager.get_test_folder()
        save_dir = self.job_manager.get_test_results_folder()

        task_list = self.training_setup_manager.get_task_list()
        print(f"Retrieved task list: {task_list}")

        if not task_list:
            task_summary_file = os.path.join(self.job_manager.get_job_folder(), 'training_tasks_summary.json')
            if os.path.exists(task_summary_file):
                with open(task_summary_file, 'r') as f:
                    task_list = json.load(f)

        if not task_list:
            self.update_status_signal.emit("Task list is not available in memory or on disk.")
            return

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(self._test_single_model, task, idx, test_folder, save_dir): task
                for idx, task in enumerate(task_list)
            }

            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    future.result()
                except Exception as exc:
                    self.queue.put({'task_error': f'Task {task} generated an exception: {exc}'})

        if not self.stop_flag:
            self.update_status_signal.emit("Testing completed!")

    def _test_single_model(self, task, idx, test_folder, save_dir):
        print(f"Testing service instance: {self.testing_service}")
        print(f"Has run_testing: {hasattr(self.testing_service, 'run_testing')}")

        """Test a single model and save the result."""
        try:
            lookback = task['hyperparams']['LOOKBACK']
            model_path = task['model_path']
            print(f"Testing model: {model_path}")

            X_test, y_test = self.test_data_service.load_and_process_data(test_folder, lookback)
            print(f"Loaded test data with shapes: {X_test.shape}, {y_test.shape}")

            shorthand_name = self.generate_shorthand_name(task)
            print(f"Generated shorthand name: {shorthand_name}")

            # Notify the status signal
            self.update_status_signal.emit(f'Testing model: {shorthand_name}')

            print(f"Testing service: {self.testing_service}")
            print(f"Has run_testing: {hasattr(self.testing_service, 'run_testing')}")
            results = self.testing_service.run_testing(task, model_path, X_test, y_test, save_dir)
            self.testing_service.save_test_results(results, shorthand_name, save_dir)

            # Emit the task result to be processed by the GUI
            self.result_signal.emit({
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
            self.result_signal.emit({'task_error': str(e)})

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

    def stop_testing(self):
        """Stop the testing process."""
        self.stop_flag = True  # Set the stop flag

