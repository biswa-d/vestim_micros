import os, uuid, time
import json
from vestim.gateway.src.hyper_param_manager_qt import VEstimHyperParamManager
from vestim.services.model_training.src.LSTM_model_service_test import LSTMModelService
from vestim.gateway.src.job_manager_qt import JobManager
import logging
import torch

class VEstimTrainingSetupManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(VEstimTrainingSetupManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, progress_signal=None, job_manager=None):
        if not hasattr(self, 'initialized'):  # Ensure initialization only happens once
            self.logger = logging.getLogger(__name__)  # Initialize logger
            self.params = None
            self.current_hyper_params = None
            self.hyper_param_manager = VEstimHyperParamManager()  # Initialize your hyperparameter manager here
            self.lstm_model_service = LSTMModelService()  # Initialize your model service here
            self.job_manager = job_manager  # JobManager should be passed in or initialized separately
            self.models = []  # Store model information
            self.training_tasks = []  # Store created tasks
            self.progress_signal = progress_signal  # Signal to communicate progress with the GUI
            self.initialized = True  # Mark as initialized

    def setup_training(self):
        print("Setting up training by the manager...")
        self.logger.info("Setting up training...")
        self.logger.info("Fetching hyperparameters...")
        """Set up the training process, including building models and creating training tasks."""
        try:
            print("Fetching hyperparameters...")
            self.params = self.hyper_param_manager.get_hyper_params()
            self.current_hyper_params = self.params
            self.logger.info(f"Params after updating: {self.current_hyper_params}")

            # Emit progress signal to indicate model building is starting
            if self.progress_signal:
                self.progress_signal.emit("Building models...", "", 0)

            # Build models
            self.build_models()

            # Emit progress signal to indicate training task creation is starting
            if self.progress_signal:
                self.progress_signal.emit("Creating training tasks...", "", 0)

            # Create training tasks
            self.create_training_tasks()

            # Emit final progress signal after tasks are created
            task_count = len(self.training_tasks)
            if self.progress_signal:
                self.progress_signal.emit(
                    f"Setup complete! Task info saved in {self.job_manager.get_job_folder()}.",
                    self.job_manager.get_job_folder(),
                    task_count
                )

        except Exception as e:
            self.logger.error(f"Error during setup: {str(e)}")
            # Handle any error during setup and pass it to the GUI
            if self.progress_signal:
                self.progress_signal.emit(f"Error during setup: {str(e)}", "", 0)

    def create_selected_model(self, model_type, model_params, model_path):
        """Creates and saves the selected model based on the dropdown selection."""
        
        model_map = {
            "LSTM": self.lstm_model_service.create_and_save_lstm_model,
            #"LSTM Batch Norm": self.lstm_model_service.create_and_save_lstm_model_with_BN,
            "LSTM Layer Norm": self.lstm_model_service.create_and_save_lstm_model_with_LN,
            #"Transformer": self.transformer_model_service.create_and_save_transformer_model,
            #"FCNN": self.fcnn_model_service.create_and_save_fcnn_model,
            #"GRU": self.gru_model_service.create_and_save_gru_model,
        }

        # Call the function from the dictionary or raise an error if not found
        if model_type in model_map:
            # Determine the target device from global params
            selected_device_str = self.params.get('DEVICE_SELECTION', 'cuda:0') # Default to cuda:0 if not found
            try:
                if selected_device_str.startswith("cuda") and not torch.cuda.is_available():
                    self.logger.warning(f"TrainingSetupManager: CUDA device {selected_device_str} selected, but CUDA not available. Model will be built for CPU.")
                    target_device = torch.device("cpu")
                elif selected_device_str.startswith("cuda"):
                    target_device = torch.device(selected_device_str)
                elif selected_device_str == "CPU":
                    target_device = torch.device("cpu")
                else:
                    self.logger.warning(f"TrainingSetupManager: Unrecognized device '{selected_device_str}'. Defaulting model build to cuda:0 if available, else CPU.")
                    target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            except Exception as e:
                self.logger.error(f"TrainingSetupManager: Error determining target device '{selected_device_str}': {e}. Defaulting model build to CPU.")
                target_device = torch.device("cpu")
            
            self.logger.info(f"TrainingSetupManager: Passing target_device {target_device} to model creation for {model_type}")
            return model_map[model_type](model_params, model_path, target_device)
        
        raise ValueError(f"Unsupported model type: {model_type}")

    def build_models(self):
        """Build and store the LSTM models based on hyperparameters."""
        try:
            # Ensure HIDDEN_UNITS and LAYERS are properly formatted
            hidden_units_value = str(self.params['HIDDEN_UNITS'])  # Convert to string if it's an integer
            layers_value = str(self.params['LAYERS'])  # Convert to string if it's an integer

            hidden_units_list = [int(h) for h in hidden_units_value.split(',')]  # Parse hidden units
            layers_list = [int(l) for l in layers_value.split(',')]  # Parse layers

            # **Get input and output feature sizes**
            feature_columns = self.params.get("FEATURE_COLUMNS", [])
            target_column = self.params.get("TARGET_COLUMN", "")
            model_type = self.params.get("MODEL_TYPE", "")

            if not feature_columns or not target_column:
                raise ValueError("Feature columns or target column not set in hyperparameters.")

            input_size = len(feature_columns)  # Number of input features
            output_size = 1  # Assuming a single target variable for regression

            self.logger.info(f"Building {model_type} models with INPUT_SIZE={input_size}, OUTPUT_SIZE={output_size}")

            # Iterate over all combinations of hidden_units and layers
            for hidden_units in hidden_units_list:
                for layers in layers_list:
                    self.logger.info(f"Creating model with hidden_units: {hidden_units}, layers: {layers}")

                    # Create model directory
                    model_dir = os.path.join(
                        self.job_manager.get_job_folder(),
                        'models',
                        f'model_{model_type}_hu_{hidden_units}_layers_{layers}'
                    )
                    os.makedirs(model_dir, exist_ok=True)

                    model_name = "untrained_model_template.pth" # Changed filename
                    model_path = os.path.join(model_dir, model_name)

                    # Model parameters
                    model_params = {
                        "INPUT_SIZE": input_size,  # Dynamically set input size
                        "OUTPUT_SIZE": output_size,  # Single target output
                        "HIDDEN_UNITS": hidden_units,
                        "LAYERS": layers
                    }

                    # Create and save the LSTM model
                    model = self.create_selected_model(model_type, model_params, model_path)

                    # Store model information
                    self.models.append({
                        'model': model,
                        'model_type': model_type,
                        'model_dir': model_dir,
                        "FEATURE_COLUMNS": feature_columns,
                        "TARGET_COLUMN": target_column,
                        'hyperparams': {
                            'INPUT_SIZE': input_size,
                            'OUTPUT_SIZE': output_size,
                            'LAYERS': layers,
                            'HIDDEN_UNITS': hidden_units,
                            'model_path': model_path
                        }
                    })
            
            self.logger.info("Model building complete.")

        except Exception as e:
            self.logger.error(f"Error during model building: {e}")
            raise

    def create_training_tasks(self):
        """Create training tasks based on hyperparameters."""
        task_list = []
        job_normalization_metadata = {} # To store data from job_metadata.json once
        job_folder = self.job_manager.get_job_folder()
        metadata_file_path = os.path.join(job_folder, "job_metadata.json")

        if os.path.exists(metadata_file_path):
            try:
                with open(metadata_file_path, 'r') as f_meta:
                    job_normalization_metadata = json.load(f_meta)
                self.logger.info(f"Loaded job_metadata.json for task creation: {job_normalization_metadata}")
            except Exception as e:
                self.logger.error(f"Error loading job_metadata.json in create_training_tasks: {e}")
                # Proceed without normalization info if file is corrupt or unreadable
        else:
            self.logger.info("job_metadata.json not found. Tasks will not include normalization metadata.")

        try:
            # Parse all comma-separated values
            max_epochs_list = [int(e.strip()) for e in str(self.current_hyper_params['MAX_EPOCHS']).split(',')]
            learning_rates = [float(lr.strip()) for lr in str(self.current_hyper_params['INITIAL_LR']).split(',')]
            train_val_splits = [float(self.current_hyper_params['TRAIN_VAL_SPLIT'])]
            lookbacks = [int(lb.strip()) for lb in str(self.current_hyper_params['LOOKBACK']).split(',')]
            batch_sizes = [int(bs.strip()) for bs in str(self.current_hyper_params['BATCH_SIZE']).split(',')]
            valid_patience = [int(vp.strip()) for vp in str(self.current_hyper_params['VALID_PATIENCE']).split(',')]
            valid_frequency = int(self.current_hyper_params.get('VALID_FREQUENCY', '3'))
            repetitions = int(self.current_hyper_params.get('REPETITIONS', '1'))
            
            # Robustly get MAX_TRAINING_TIME_SECONDS
            raw_max_time = self.current_hyper_params.get('MAX_TRAINING_TIME_SECONDS')
            if isinstance(raw_max_time, (int, float)):
                max_training_time_seconds = int(raw_max_time)
            elif isinstance(raw_max_time, str) and raw_max_time.isdigit():
                max_training_time_seconds = int(raw_max_time)
            else:
                max_training_time_seconds = 0 # Default if not found, None, or not a valid number string
                if raw_max_time is not None:
                    self.logger.warning(f"Invalid value for MAX_TRAINING_TIME_SECONDS in TrainingSetupManager: '{raw_max_time}'. Defaulting to 0.")
            self.logger.info(f"TrainingSetupManager using MAX_TRAINING_TIME_SECONDS: {max_training_time_seconds}")

            # Get scheduler type
            scheduler_type = self.current_hyper_params.get('SCHEDULER_TYPE', 'StepLR')

            # Parse scheduler-specific parameters
            if scheduler_type == 'StepLR':
                lr_periods = [int(p.strip()) for p in str(self.current_hyper_params['LR_PERIOD']).split(',')]
                lr_factors = [float(f.strip()) for f in str(self.current_hyper_params['LR_PARAM']).split(',')]
            else:  # ReduceLROnPlateau
                plateau_patience = [int(p.strip()) for p in str(self.current_hyper_params['PLATEAU_PATIENCE']).split(',')]
                plateau_factors = [float(f.strip()) for f in str(self.current_hyper_params['PLATEAU_FACTOR']).split(',')]

            # Create tasks for each model and combination of hyperparameters
            for model_task in self.models:
                for max_epochs in max_epochs_list:  # Add iteration over max_epochs
                    for lr in learning_rates:
                        for train_val_split in train_val_splits:
                            for lookback in lookbacks:
                                for batch_size in batch_sizes:
                                    for vp in valid_patience:
                                        if scheduler_type == 'StepLR':
                                            for period in lr_periods:
                                                for factor in lr_factors:
                                                    for rep in range(1, repetitions + 1):
                                                        task_info = self._create_task_info(
                                                            model_task=model_task,
                                                            hyperparams={
                                                                'INITIAL_LR': lr,
                                                                'TRAIN_VAL_SPLIT': train_val_split,
                                                                'LOOKBACK': lookback,
                                                                'BATCH_SIZE': batch_size,
                                                                'VALID_PATIENCE': vp,
                                                                'MAX_EPOCHS': max_epochs,  # Use the current max_epochs value
                                                                'SCHEDULER_TYPE': 'StepLR',
                                                                'LR_PERIOD': period,
                                                                'LR_PARAM': factor,
                                                                'REPETITIONS': rep,
                                                                'ValidFrequency': valid_frequency,
                                                                'MAX_TRAINING_TIME_SECONDS': max_training_time_seconds, # Add here
                                                            },
                                                            repetition=rep,
                                                            job_normalization_metadata=job_normalization_metadata,
                                                            max_training_time_seconds_arg=max_training_time_seconds # Pass as separate arg for clarity
                                                        )
                                                        task_list.append(task_info)
                                        else:  # ReduceLROnPlateau
                                            for p_patience in plateau_patience:
                                                for p_factor in plateau_factors:
                                                    for rep in range(1, repetitions + 1):
                                                        task_info = self._create_task_info(
                                                            model_task=model_task,
                                                            hyperparams={
                                                                'INITIAL_LR': lr,
                                                                'TRAIN_VAL_SPLIT': train_val_split,
                                                                'LOOKBACK': lookback,
                                                                'BATCH_SIZE': batch_size,
                                                                'VALID_PATIENCE': vp,
                                                                'MAX_EPOCHS': max_epochs,  # Use the current max_epochs value
                                                                'SCHEDULER_TYPE': 'ReduceLROnPlateau',
                                                                'PLATEAU_PATIENCE': p_patience,
                                                                'PLATEAU_FACTOR': p_factor,
                                                                'REPETITIONS': rep,
                                                                'ValidFrequency': valid_frequency,
                                                                'MAX_TRAINING_TIME_SECONDS': max_training_time_seconds, # Add here
                                                            },
                                                            repetition=rep,
                                                            job_normalization_metadata=job_normalization_metadata,
                                                            max_training_time_seconds_arg=max_training_time_seconds # Pass as separate arg for clarity
                                                        )
                                                        task_list.append(task_info)

            # Save the task list and return
            self.training_tasks = task_list
            
            # Save task info for each task
            for task_info in task_list:
                task_dir = task_info['model_dir']
                task_info_file = os.path.join(task_dir, 'task_info.json')
                serializable_info = {k: v for k, v in task_info.items() if k != 'model'}
                with open(task_info_file, 'w') as f:
                    json.dump(serializable_info, f, indent=4)

            # Save tasks summary
            tasks_summary_file = os.path.join(self.job_manager.get_job_folder(), 'training_tasks_summary.json')
            serializable_tasks = [{k: v for k, v in task.items() if k != 'model'} for task in task_list]
            with open(tasks_summary_file, 'w') as f:
                json.dump(serializable_tasks, f, indent=4)

            return task_list

        except Exception as e:
            self.logger.error(f"Error creating training tasks: {e}")
            raise

    def _create_task_info(self, model_task, hyperparams, repetition, job_normalization_metadata=None, max_training_time_seconds_arg=0): # Added new arg
        """Helper method to create a task info dictionary."""
        if job_normalization_metadata is None:
            job_normalization_metadata = {} # Default to empty dict if not provided
        timestamp = time.strftime("%Y%m%d%H%M%S")
        task_counter = getattr(self, '_task_counter', 0) + 1
        self._task_counter = task_counter
        # Create unique task ID
        task_id = f"task_{timestamp}_{task_counter}_rep_{repetition}"
        
        # Create task directory with relevant parameters and repetition number
        scheduler_type = hyperparams['SCHEDULER_TYPE']
        if scheduler_type == 'StepLR':
            task_dir_name = f'lr_{hyperparams["INITIAL_LR"]}_period_{hyperparams["LR_PERIOD"]}_factor_{hyperparams["LR_PARAM"]}'
        else:
            task_dir_name = f'lr_{hyperparams["INITIAL_LR"]}_plat_pat_{hyperparams["PLATEAU_PATIENCE"]}_factor_{hyperparams["PLATEAU_FACTOR"]}'
        
        task_dir = os.path.join(
            model_task['model_dir'],
            f'{task_id}'
        )
        os.makedirs(task_dir, exist_ok=True)

        # Create logs directory within task directory
        logs_dir = os.path.join(task_dir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)

        
        # Get model architecture parameters
        hidden_units = model_task['hyperparams']['HIDDEN_UNITS']
        layers = model_task['hyperparams']['LAYERS']
        input_size = model_task['hyperparams']['INPUT_SIZE']
        output_size = model_task['hyperparams']['OUTPUT_SIZE']

        # Calculate num_learnable_params
        num_learnable_params = self.calculate_learnable_parameters(
            layers,
            input_size,
            hidden_units
        )

        return {
            'task_id': task_id,
            'model': model_task['model'],
            'model_dir': task_dir,
            'task_dir': task_dir,
            'model_path': os.path.join(task_dir, 'model.pth'),
            'logs_dir': logs_dir,
            'model_metadata': {
                'model_type': model_task.get('model_type', 'LSTM'),
                'num_learnable_params': num_learnable_params,
                'hidden_units': hidden_units,
                'num_layers': layers,
                'input_size': input_size,
                'output_size': output_size
            },
            'hyperparams': {
                'LAYERS': layers,
                'HIDDEN_UNITS': hidden_units,
                'INPUT_SIZE': input_size,
                'OUTPUT_SIZE': output_size,
                'BATCH_TRAINING': self.current_hyper_params.get('BATCH_TRAINING', True), # Propagate BATCH_TRAINING
                'BATCH_SIZE': hyperparams['BATCH_SIZE'],
                'MAX_EPOCHS': hyperparams['MAX_EPOCHS'],
                'INITIAL_LR': hyperparams['INITIAL_LR'],
                'VALID_PATIENCE': hyperparams['VALID_PATIENCE'],
                'ValidFrequency': hyperparams['ValidFrequency'],
                'LOOKBACK': hyperparams['LOOKBACK'],
                'SCHEDULER_TYPE': hyperparams['SCHEDULER_TYPE'],
                'LR_PERIOD': hyperparams.get('LR_PERIOD'),
                'LR_PARAM': hyperparams.get('LR_PARAM'),
                'PLATEAU_PATIENCE': hyperparams.get('PLATEAU_PATIENCE'),
                'PLATEAU_FACTOR': hyperparams.get('PLATEAU_FACTOR'),
                'REPETITIONS': hyperparams['REPETITIONS'],
                'NUM_LEARNABLE_PARAMS': num_learnable_params,
            },
            'data_loader_params': {
                'lookback': hyperparams['LOOKBACK'],
                'batch_size': hyperparams['BATCH_SIZE'],
                'feature_columns': model_task['FEATURE_COLUMNS'],
                'target_column': model_task['TARGET_COLUMN'],
                'train_val_split': hyperparams['TRAIN_VAL_SPLIT'],
                'num_workers': 4
            },
            'training_params': {
                'early_stopping': True,
                'early_stopping_patience': hyperparams['VALID_PATIENCE'],
                'save_best_model': True,
                'checkpoint_dir': os.path.join(logs_dir, 'checkpoints'),
                'best_model_path': os.path.join(task_dir, 'best_model.pth'),
                'max_training_time_seconds': max_training_time_seconds_arg # Add to training_params
            },
            'results': {
                'best_val_loss': float('inf'),
                'best_epoch': 0,
                'training_time': 0,
                'early_stopped': False,
                'completed': False
            },
            'csv_log_file': os.path.join(logs_dir, 'training_progress.csv'),
            'db_log_file': os.path.join(logs_dir, f'{task_id}_training.db'),
            'job_metadata': job_normalization_metadata, # Embed normalization metadata from job_metadata.json
            'job_folder_augmented_from': self.job_manager.get_job_folder() # Add path to job folder for scaler path resolution
        }

    def calculate_learnable_parameters(self, layers, input_size, hidden_units):
        """
        Calculate the number of learnable parameters for an LSTM model.

        :param layers: Number of layers (an integer representing the number of LSTM layers)
        :param input_size: The size of the input features (e.g., 3 for [SOC, Current, Temp])
        :param hidden_units: An integer representing the number of hidden units in each layer
        :return: Total number of learnable parameters
        """

        # Initialize the number of parameters
        learnable_params = 0

        # Input-to-hidden weights for the first layer (4 * hidden_units * (input_size + hidden_units))
        # We account for 4 gates (input, forget, output, and candidate gates)
        input_layer_params = 4 * (input_size + hidden_units) * hidden_units

        # Add bias terms for each gate in the first layer
        input_layer_bias = 4 * hidden_units

        learnable_params += input_layer_params + input_layer_bias

        # For each additional LSTM layer, it's hidden_units -> hidden_units
        for i in range(1, layers):
            hidden_layer_params = 4 * (hidden_units + hidden_units) * hidden_units
            hidden_layer_bias = 4 * hidden_units
            learnable_params += hidden_layer_params + hidden_layer_bias

        # Output layer (assuming 1 output)
        output_size = 1
        output_layer_params = hidden_units * output_size
        output_layer_bias = output_size  # 1 bias for the output layer
        learnable_params += output_layer_params + output_layer_bias

        return learnable_params

    def update_task(self, task_id, db_log_file=None, csv_log_file=None):
        """Update a specific task in the manager."""
        for task in self.training_tasks:
            if task['task_id'] == task_id:
                if db_log_file:
                    task['db_log_file'] = db_log_file
                if csv_log_file:
                    task['csv_log_file'] = csv_log_file
                # Additional fields can be updated similarly
                break

    def get_task_list(self):
        """Returns the list of training tasks."""
        return self.training_tasks

    def validate_parameters(self, params):
        """Validate and convert parameters to appropriate types."""
        try:
            validated = {
                # Integer conversions
                'LAYERS': int(params['LAYERS']),
                'HIDDEN_UNITS': int(params['HIDDEN_UNITS']),
                'BATCH_SIZE': int(params['BATCH_SIZE']),
                
                # Float conversions
                'TRAIN_VAL_SPLIT': float(params['TRAIN_VAL_SPLIT']),
                
                # String parameters (for potential comma-separated values)
                'INITIAL_LR': str(params['INITIAL_LR']),
                'LR_PARAM': str(params['LR_PARAM']),
                'MAX_EPOCHS': str(params.get('MAX_EPOCHS', '100')),
                # ... other parameters ...
            }
            return validated
        except (ValueError, KeyError) as e:
            raise ValueError(f"Parameter validation failed: {str(e)}")
