import time
import os
import json
from vestim.gateway.src.hyper_param_manager import VEstimHyperParamManager
from vestim.services.model_training.src.LSTM_model_service import LSTMModelService
from vestim.gateway.src.job_manager import JobManager

class VEstimTrainingSetupManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(VEstimTrainingSetupManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, update_status_callback=None):
        if not hasattr(self, 'initialized'):  # Check if already initialized
            self.params = None
            self.current_hyper_params = None
            self.hyper_param_manager = VEstimHyperParamManager()
            self.lstm_model_service = LSTMModelService()
            self.job_manager = JobManager()
            self.start_time = None
            self.models = []
            self.training_tasks = []
            self.update_status = update_status_callback or (lambda message: None)
            self.load_cached_task_list()  # Load cached task list on initialization
            self.initialized = True  # Mark as initialized

    def load_cached_task_list(self):
        """Loads the cached task list from disk if available."""
        tasks_summary_file = os.path.join(self.job_manager.get_job_folder(), 'training_tasks_summary.json')
        if os.path.exists(tasks_summary_file):
            with open(tasks_summary_file, 'r') as f:
                self.training_tasks = json.load(f)
            print(f"Loaded {len(self.training_tasks)} tasks from cache.")
        else:
            print("No cached tasks found.")

    def setup_training(self):
        """Set up the training process, including building models and creating training tasks."""
        self.params = self.hyper_param_manager.get_current_params()
        print(f"Params after loading: {self.params}")

        self.current_hyper_params = self.params
        print(f"Params after updating: {self.current_hyper_params}")

        # Update status to indicate model building is starting
        self.update_status("Building models...")
        # Build models
        self.build_models()

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

    def create_training_tasks(self):
        """
        Create a list of training tasks by combining models and relevant training hyperparameters.
        This method will also save the task information to disk for future use.
        """
        task_list = []  # Initialize a list to store tasks

        # Retrieve relevant hyperparameters for training
        learning_rates = [float(lr) for lr in self.current_hyper_params['INITIAL_LR'].split(',')]
        lr_drop_factors = [float(drop_factor) for drop_factor in self.current_hyper_params['LR_DROP_FACTOR'].split(',')]
        lr_drop_periods = [int(drop) for drop in self.current_hyper_params['LR_DROP_PERIOD'].split(',')]
        valid_patience_values = [int(vp) for vp in self.current_hyper_params['VALID_PATIENCE'].split(',')]
        repetitions = int(self.current_hyper_params['REPETITIONS'])
        lookbacks = [int(lb) for lb in self.current_hyper_params['LOOKBACK'].split(',')]
        batch_sizes = [int(bs) for bs in self.current_hyper_params['BATCH_SIZE'].split(',')]
        max_epochs = int(self.current_hyper_params['MAX_EPOCHS'])  # Ensure MAX_EPOCHS is included

        # Iterate through each model
        for model_task in self.models:
            model = model_task['model']
            model_metadata = {
                'model_type': 'LSTMModel',
                'input_size': model.input_size,
                'hidden_units': model.hidden_units,
                'num_layers': model.num_layers,
            }

            # Iterate through hyperparameters
            for lr in learning_rates:
                for drop_factor in lr_drop_factors:
                    for drop_period in lr_drop_periods:
                        for patience in valid_patience_values:
                            for lookback in lookbacks:
                                for batch_size in batch_sizes:
                                    for rep in range(1, repetitions + 1):
                                        # Create a unique directory for each task based on all parameters
                                        task_dir = os.path.join(
                                            model_task['model_dir'],
                                            f'lr_{lr}_drop_{drop_period}_factor_{drop_factor}_patience_{patience}_rep_{rep}_lookback_{lookback}_batch_{batch_size}'
                                        )
                                        os.makedirs(task_dir, exist_ok=True)

                                        # Define task information
                                        task_info = {
                                            'model': model,  # Ensure this model instance is correctly initialized
                                            'model_metadata': model_metadata,  # Use metadata instead of the full model
                                            'data_loader_params': {
                                                'lookback': lookback,
                                                'batch_size': batch_size,
                                            },
                                            'model_dir': task_dir,
                                            'model_path': os.path.join(task_dir, 'model.pth'),
                                            'hyperparams': {
                                                'LAYERS': self.current_hyper_params['LAYERS'],
                                                'HIDDEN_UNITS': model_metadata['hidden_units'],
                                                'BATCH_SIZE': batch_size,
                                                'LOOKBACK': lookback,
                                                'INITIAL_LR': lr,
                                                'LR_DROP_FACTOR': drop_factor,
                                                'LR_DROP_PERIOD': drop_period,
                                                'VALID_PATIENCE': patience,
                                                'ValidFrequency': self.current_hyper_params['ValidFrequency'],
                                                'REPETITIONS': rep,
                                                'MAX_EPOCHS': max_epochs,  # Include MAX_EPOCHS here
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


    def get_task_list(self):
        """Returns the list of training tasks."""
        return self.training_tasks
