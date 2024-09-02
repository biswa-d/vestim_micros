import time
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.gateway.src.hyper_param_manager import VEstimHyperParamManager
from services.model_training.src.training_service_m import TrainingService
from src.services.model_training.src.LSTM_model_service import LSTMModelService
from src.services.model_training.src.data_loader_service import DataLoaderService
from src.gateway.src.job_manager import JobManager

class VEstimTrainingManager:
    def __init__(self):
        self.params = None
        self.current_hyper_params = None  # Initialize as None initially
        self.hyper_param_manager = VEstimHyperParamManager()  # Retrieve hyperparameters from the manager
        self.training_service = TrainingService()
        self.lstm_model_service = LSTMModelService()
        self.job_manager = JobManager()  # Manages job-related directories and files
        self.data_loader_service = DataLoaderService()  # Initialize the data loader service
        self.start_time = None
        self.models = []  # Store models in memory
        self.data_loaders = []  # Store data loaders in memory
        self.executor = None  # Thread pool executor
        self.training_futures = []  # Store futures to manage training tasks
        self.training_active = True  # Flag to indicate if training should continue
        self.training_tasks = []  # To store the mapping of hyperparameters, models, and data loaders


    def build_models(self):
        self.params = self.hyper_param_manager.get_current_params()
        self.current_hyper_params = self.params

        input_size = 3
        output_size = 1
        layers = int(self.params['LAYERS'])
        hidden_units_list = [int(h) for h in self.params['HIDDEN_UNITS'].split(',')]
        lr_drop_factors = [float(lr) for lr in self.params['LR_DROP_FACTOR'].split(',')]
        repetitions = int(self.params['REPETITIONS'])

        for hidden_units in hidden_units_list:
            for lr_drop_factor in lr_drop_factors:
                for repetition in range(1, repetitions + 1):  # Loop over repetitions
                    model_dir = os.path.join(self.job_manager.get_job_folder(), 'models', f'model_lstm_hu_{hidden_units}_lrd_{lr_drop_factor}_rep_{repetition}')
                    os.makedirs(model_dir, exist_ok=True)

                    model_name = f"model_lstm_hu_{hidden_units}_lrd_{lr_drop_factor}_rep_{repetition}.pth"
                    model_path = os.path.join(model_dir, model_name)

                    model = self.lstm_model_service.create_and_save_lstm_model(input_size, output_size, layers, hidden_units, model_path)

                    task = {
                        'model': model,
                        'data_loader': None,  # Placeholder until data_loader is assigned
                        'hyperparams': {
                            'LAYERS': layers,
                            'HIDDEN_UNITS': hidden_units,
                            'BATCH_SIZE': None,  # Placeholder until data_loader is assigned
                            'LOOKBACK': None,  # Placeholder until data_loader is assigned
                            'INITIAL_LR': self.params['INITIAL_LR'],
                            'LR_DROP_FACTOR': lr_drop_factor,
                            'LR_DROP_PERIOD': self.params['LR_DROP_PERIOD'],
                            'VALID_PATIENCE': self.params['VALID_PATIENCE'],
                            'ValidFrequency': self.params['ValidFrequency'],
                            'REPETITIONS': repetition,
                            'model_path': model_path
                        }
                    }
                    self.training_tasks.append(task)


    def create_data_loaders(self):
        lookbacks = [int(lb) for lb in self.current_hyper_params['LOOKBACK'].split(',')]
        batch_sizes = [int(bs) for bs in self.current_hyper_params['BATCH_SIZE'].split(',')]
        num_workers = 4

        task_index = 0  # To track which task we're updating

        for lookback in lookbacks:
            for batch_size in batch_sizes:
                data_loader = self.data_loader_service.create_data_loader(
                    train_folder=os.path.join(self.job_manager.get_job_folder(), 'train', 'processed_data'),
                    lookback=lookback,
                    batch_size=batch_size,
                    num_workers=num_workers
                )

                # Update the corresponding task with data_loader and specific hyperparams
                self.training_tasks[task_index]['data_loader'] = data_loader
                self.training_tasks[task_index]['hyperparams']['LOOKBACK'] = lookback
                self.training_tasks[task_index]['hyperparams']['BATCH_SIZE'] = batch_size
                task_index += 1


    def start_training(self):
        # Build models first
        self.build_models()

        # Create data loaders
        self.create_data_loaders()

        # Start the training process for each task in self.training_tasks
        self.executor = ThreadPoolExecutor(max_workers=1)  # Sequential execution

        for task in self.training_tasks:
            model = task['model']
            data_loader = task['data_loader']
            hyperparams = task['hyperparams']

            if self.training_active:
                future = self.executor.submit(
                    self.training_service.start_training,
                    model,
                    data_loader,
                    hyperparams,
                    self.update_progress
                )
                self.training_futures.append(future)
            else:
                break

        # Handle completed training tasks
        for future in as_completed(self.training_futures):
            if not self.training_active:
                break
            try:
                future.result()  # Wait for the model training to complete
            except Exception as exc:
                print(f"Model training generated an exception: {exc}")
            else:
                print(f"Model training completed successfully.")
                # Save the trained model after training
                self.lstm_model_service.save_model(task['model'], task['hyperparams']['model_path'])


    def stop_training(self):
        # Stop the ongoing training process
        self.training_active = False
        # Shut down the executor and cancel any remaining tasks
        if self.executor:
            self.executor.shutdown(wait=False)
            print("Training has been stopped and future tasks have been canceled.")

        # Save model states or any other necessary information
        for model in self.models:
            # Assuming the model save logic is already implemented in the service
            print(f"Saving state for model: {model}")

    def update_progress(self, task, repetition, epoch, train_loss, validation_error):
        if not self.training_active:
            return  # Stop updating if training is stopped

        # Calculate elapsed time
        elapsed_time = time.time() - self.start_time
        hours, minutes, seconds = self._format_time(elapsed_time)

        # Add task-specific information to the progress update
        task_info = {
            'model_name': task['hyperparams']['model_path'],
            'hidden_units': task['hyperparams']['hidden_units'],
            'lr_drop_factor': task['hyperparams']['lr_drop_factor'],
            # Include other relevant hyperparameters as needed
        }

        # Update GUI with the progress and task-specific info
        self._update_gui_with_progress(task_info, repetition, epoch, train_loss, validation_error, hours, minutes, seconds)

    def _update_gui_with_progress(self, task_info, repetition, epoch, train_loss, validation_error, hours, minutes, seconds):
        # Interface with the GUI to update the training progress, including task-specific info
        pass  # This would be implemented based on your specific GUI framework


    def _format_time(self, elapsed_time):
        # Format the elapsed time into hours, minutes, and seconds
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        return int(hours), int(minutes), int(seconds)
