import time
import os,json
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.gateway.src.hyper_param_manager import VEstimHyperParamManager
from src.services.model_training.src.training_service import TrainingService
from src.services.model_training.src.LSTM_model_service_test import LSTMModelService
from src.services.model_training.src.data_loader_service_test import DataLoaderService
from src.gateway.src.job_manager import JobManager

class VEstimTrainingManager:
    def __init__(self,update_status_callback):
        self.params = None
        self.current_hyper_params = None
        self.hyper_param_manager = VEstimHyperParamManager()
        self.training_service = TrainingService()
        self.lstm_model_service = LSTMModelService()
        self.job_manager = JobManager()
        self.data_loader_service = DataLoaderService()
        self.start_time = None
        self.models = []
        self.data_loaders = []
        self.executor = None
        self.training_futures = []
        self.training_active = True
        self.training_tasks = []
        self.update_status = update_status_callback

    def setup_training(self):
        """Set up the training process, including building models and creating data loaders."""
        self.params = self.hyper_param_manager.get_current_params()
        print(f"Params after loading: {self.params}")
        
        self.current_hyper_params = self.params
        print(f"Params after updating: {self.current_hyper_params}")
        # Update status to indicate model building is starting
        self.update_status("Building models...")
        # Build models
        self.build_models()
        # Update status to indicate data loader creation is starting
        self.update_status("Creating data loaders...")
        # Create data loaders
        self.create_data_loaders()
        # Update status to indicate training task creation is starting
        self.update_status("Creating training tasks...")
        # Create training tasks
        self.create_training_tasks()
        # Update status to indicate setup is complete
        self.update_status(f"Setup complete! Task info saved in {self.job_manager.get_job_folder()}.")


    def build_models(self):
        hidden_units_list = [int(h) for h in self.params['HIDDEN_UNITS'].split(',')]
        layers = int(self.params['LAYERS'])

        for hidden_units in hidden_units_list:
            print(f"Creating model with hidden_units: {hidden_units}")
            
            model_dir = os.path.join(self.job_manager.get_job_folder(), 'models', f'model_lstm_hu_{hidden_units}')
            os.makedirs(model_dir, exist_ok=True)

            model_name = f"model_lstm_hu_{hidden_units}.pth"
            model_path = os.path.join(model_dir, model_name)

            model_params = {
                "INPUT_SIZE": 3,  # Modify as needed
                "HIDDEN_UNITS": hidden_units,
                "LAYERS": layers
            }

            model = self.lstm_model_service.create_and_save_lstm_model(model_params, model_path)
            # Store the model along with its directory and hyperparameters
            self.models.append({
                'model': model,
                'model_dir': model_dir,
                'hyperparams': {
                    'LAYERS': layers,
                    'HIDDEN_UNITS': hidden_units,
                    'model_path': model_path
                }
            })


    def create_data_loaders(self):
        lookbacks = [int(lb) for lb in self.current_hyper_params['LOOKBACK'].split(',')]
        batch_sizes = [int(bs) for bs in self.current_hyper_params['BATCH_SIZE'].split(',')]
        num_workers = 4

        self.data_loaders = []  # Initialize a list to store the combined data loaders

        for lookback in lookbacks:
            for batch_size in batch_sizes:
                # Create both train and validation loaders for the same lookback and batch size
                train_loader, val_loader = self.data_loader_service.create_data_loaders(
                    folder_path=os.path.join(self.job_manager.get_job_folder(), 'train', 'processed_data'),
                    lookback=lookback,
                    batch_size=batch_size,
                    num_workers=num_workers
                )

                # Store the pair of loaders with their specific hyperparameters
                self.data_loaders.append({
                    'train_loader': train_loader,
                    'val_loader': val_loader,
                    'lookback': lookback,
                    'batch_size': batch_size
                })


    def create_training_tasks(self):
        """
        Create a list of training tasks by combining models, data loaders, and relevant training parameters.
        This method will also save the task information to disk for future use.
        """
        task_list = []  # Initialize a list to store tasks

        # Retrieve relevant hyperparameters for training
        learning_rates = [float(lr) for lr in self.current_hyper_params['INITIAL_LR'].split(',')]
        lr_drop_periods = [int(drop) for drop in self.current_hyper_params['LR_DROP_PERIOD'].split(',')]
        valid_patience_values = [int(vp) for vp in self.current_hyper_params['VALID_PATIENCE'].split(',')]
        repetitions = int(self.current_hyper_params['REPETITIONS'])
        batch_sizes = [int(bs) for bs in self.current_hyper_params['BATCH_SIZE'].split(',')]
        lookbacks = [int(lb) for lb in self.current_hyper_params['LOOKBACK'].split(',')]

        # Iterate through models, data loaders, and training parameters
        for model_task in self.models:  # Iterate through the list of models
            model = model_task['model']
            model_metadata = {
                'model_type': 'LSTMModel',
                'input_size': model.input_size,
                'hidden_units': model.hidden_units,
                'num_layers': model.num_layers,
                # Add more metadata if necessary
            }

            for data_loader_task in self.data_loaders:  # Iterate through the combined loaders
                for lr in learning_rates:
                    for drop_period in lr_drop_periods:
                        for patience in valid_patience_values:
                            for rep in range(1, repetitions + 1):
                                for batch_size in batch_sizes:
                                    for lookback in lookbacks:

                                        # Create a unique directory for each task based on all parameters, including batch and lookback
                                        task_dir = os.path.join(
                                            model_task['model_dir'],
                                            f'lr_{lr}_drop_{drop_period}_patience_{patience}_rep_{rep}_lookback_{lookback}_batch_{batch_size}'
                                        )
                                        os.makedirs(task_dir, exist_ok=True)

                                        # Define task information, including model metadata and training parameters
                                        task_info = {
                                            'model_metadata': model_metadata,  # Use metadata instead of the full model
                                            'data_loader_params': {
                                                'lookback': lookback,
                                                'batch_size': batch_size,
                                                # Include other necessary parameters here
                                            },
                                            'model_dir': task_dir,
                                            'hyperparams': {
                                                'LAYERS': self.current_hyper_params['LAYERS'],
                                                'HIDDEN_UNITS': model_metadata['hidden_units'],
                                                'BATCH_SIZE': batch_size,
                                                'LOOKBACK': lookback,
                                                'INITIAL_LR': lr,
                                                'LR_DROP_PERIOD': drop_period,
                                                'VALID_PATIENCE': patience,
                                                'ValidFrequency': self.current_hyper_params['ValidFrequency'],
                                                'REPETITIONS': rep,
                                                'model_path': os.path.join(task_dir, 'model.pth')  # Where the model will be saved later
                                            }
                                        }

                                        # Append the task to the task list
                                        task_list.append(task_info)

                                        # Save the task info to disk as a JSON file
                                        task_info_file = os.path.join(task_dir, 'task_info.json')
                                        with open(task_info_file, 'w') as f:
                                            json.dump({k: v for k, v in task_info.items() if k != 'model'}, f, indent=4)

        # Replace the existing training tasks with the newly created task list
        self.training_tasks = task_list

        # Optionally, save the entire task list for future reference at the root level
        tasks_summary_file = os.path.join(self.job_manager.get_job_folder(), 'training_tasks_summary.json')
        with open(tasks_summary_file, 'w') as f:
            json.dump([{k: v for k, v in task.items() if k != 'model'} for task in self.training_tasks], f, indent=4)

        print(f"Created {len(self.training_tasks)} training tasks and saved to disk.")

    def start_training(self):
        """Start the training process for all tasks."""
        self.setup_training()
        #self.executor = ThreadPoolExecutor(max_workers=1)

        for task in self.training_tasks:
            if self.training_active:
                future = self.executor.submit(
                    self.training_service.start_training,
                    task['model'],
                    task['train_loader'],
                    task['val_loader'],
                    task['hyperparams'],
                    self.update_progress,
                    task['model_dir']
                )
                self.training_futures.append(future)
            else:
                break

        for future in as_completed(self.training_futures):
            if not self.training_active:
                break
            try:
                future.result()
            except Exception as exc:
                print(f"Model training generated an exception: {exc}")
            else:
                print("Model training completed successfully.")

    def stop_training(self):
        self.training_active = False
        if self.executor:
            self.executor.shutdown(wait=False)
            print("Training has been stopped and future tasks have been canceled.")

        for model in self.models:
            print(f"Saving state for model: {model}")

    def update_progress(self, task, repetition, epoch, train_loss, validation_error):
        if not self.training_active:
            return

        elapsed_time = time.time() - self.start_time
        hours, minutes, seconds = self._format_time(elapsed_time)

        task_info = {
            'model_name': task['hyperparams']['model_path'],
            'hidden_units': task['hyperparams']['HIDDEN_UNITS'],
            'lr_drop_period': task['hyperparams']['LR_DROP_PERIOD'],
        }

        self._update_gui_with_progress(task_info, repetition, epoch, train_loss, validation_error, hours, minutes, seconds)

    def _update_gui_with_progress(self, task_info, repetition, epoch, train_loss, validation_error, hours, minutes, seconds):
        pass  # Implement based on your specific GUI framework

    def _format_time(self, elapsed_time):
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        return int(hours), int(minutes), int(seconds)
