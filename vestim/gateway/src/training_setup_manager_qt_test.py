import os, uuid, time
import json
from vestim.gateway.src.hyper_param_manager_qt_test import VEstimHyperParamManager
from vestim.services.model_training.src.LSTM_model_service_test import LSTMModelService
from vestim.gateway.src.job_manager_qt import JobManager
import logging

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
            return model_map[model_type](model_params, model_path)
        
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

                    model_name = f"model_lstm_hu_{hidden_units}_layers_{layers}.pth"
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
        """Create training tasks based on hyperparameters and selected scheduler."""
        self.logger.info("Creating training tasks...")
        task_list = []

        def parse_param_list(param_value, convert_func=float):
            """Safely parse parameter that might be comma-separated."""
            if isinstance(param_value, (int, float)):
                return [param_value]
            try:
                values = [v.strip() for v in str(param_value).replace(',', ' ').split() if v]
                if not values:
                    raise ValueError(f"Empty parameter value")
                return [convert_func(v) for v in values]
            except ValueError as e:
                self.logger.error(f"Error parsing parameter: {param_value}")
                raise ValueError(f"Invalid value in list: {param_value}. Expected {convert_func.__name__} values.")

        try:
            # Common parameters for both schedulers
            learning_rates = parse_param_list(self.current_hyper_params['INITIAL_LR'], float)
            train_val_splits = [float(self.current_hyper_params['TRAIN_VAL_SPLIT'])]
            lookbacks = parse_param_list(self.current_hyper_params['LOOKBACK'], int)
            batch_sizes = parse_param_list(self.current_hyper_params['BATCH_SIZE'], int)
            max_epochs = parse_param_list(self.current_hyper_params.get('MAX_EPOCHS', '100'), int)
            valid_patience = parse_param_list(self.current_hyper_params['VALID_PATIENCE'], int)
            valid_frequency = int(self.current_hyper_params.get('VALID_FREQUENCY', '3'))
            repetitions = int(self.current_hyper_params.get('REPETITIONS', '1'))
            
            # Get scheduler type
            scheduler_type = self.current_hyper_params.get('SCHEDULER_TYPE', 'StepLR')

            # Parse scheduler-specific parameters
            if scheduler_type == 'StepLR':
                lr_periods = parse_param_list(self.current_hyper_params['LR_PERIOD'], int)
                lr_factors = parse_param_list(self.current_hyper_params['LR_PARAM'], float)
            else:  # ReduceLROnPlateau
                plateau_patience = parse_param_list(self.current_hyper_params['PLATEAU_PATIENCE'], int)
                plateau_factors = parse_param_list(self.current_hyper_params['PLATEAU_FACTOR'], float)

            # Create tasks for each model
            for model_task in self.models:
                feature_columns = model_task['FEATURE_COLUMNS']
                target_column = model_task['TARGET_COLUMN']
                model = model_task['model']
                
                # Base nested loops for common parameters
                for lr in learning_rates:
                    for train_val_split in train_val_splits:
                        for lookback in lookbacks:
                            for batch_size in batch_sizes:
                                for vp in valid_patience:
                                    # Branch based on scheduler type
                                    if scheduler_type == 'StepLR':
                                        for period in lr_periods:
                                            for factor in lr_factors:
                                                # Add repetitions as innermost loop
                                                for rep in range(1, repetitions + 1):
                                                    task_info = self._create_task_info(
                                                        model_task=model_task,
                                                        hyperparams={
                                                            'INITIAL_LR': lr,
                                                            'TRAIN_VAL_SPLIT': train_val_split,
                                                            'LOOKBACK': lookback,
                                                            'BATCH_SIZE': batch_size,
                                                            'VALID_PATIENCE': vp,
                                                            'MAX_EPOCHS': max_epochs[0],
                                                            'SCHEDULER_TYPE': 'StepLR',
                                                            'LR_PERIOD': period,
                                                            'LR_PARAM': factor,
                                                            'REPETITIONS': rep,
                                                            'ValidFrequency': valid_frequency,
                                                        },
                                                        repetition=rep
                                                    )
                                                    task_list.append(task_info)
                                    else:
                                        for p_patience in plateau_patience:
                                            for p_factor in plateau_factors:
                                                # Add repetitions as innermost loop
                                                for rep in range(1, repetitions + 1):
                                                    task_info = self._create_task_info(
                                                        model_task=model_task,
                                                        hyperparams={
                                                            'INITIAL_LR': lr,
                                                            'TRAIN_VAL_SPLIT': train_val_split,
                                                            'LOOKBACK': lookback,
                                                            'BATCH_SIZE': batch_size,
                                                            'VALID_PATIENCE': vp,
                                                            'MAX_EPOCHS': max_epochs[0],
                                                            'SCHEDULER_TYPE': 'ReduceLROnPlateau',
                                                            'PLATEAU_PATIENCE': p_patience,
                                                            'PLATEAU_FACTOR': p_factor,
                                                            'REPETITIONS': rep,
                                                            'ValidFrequency': valid_frequency,
                                                        },
                                                        repetition=rep
                                                    )
                                                    task_list.append(task_info)

            # Save the task list
            self.training_tasks = task_list
            
            # Save task info for each task
            for task_info in task_list:
                task_dir = task_info['model_dir']
                task_info_file = os.path.join(task_dir, 'task_info.json')
                
                # Create a copy of task_info without the model object (which can't be serialized)
                serializable_info = {k: v for k, v in task_info.items() if k != 'model'}
                
                with open(task_info_file, 'w') as f:
                    json.dump(serializable_info, f, indent=4)

            # Save the entire task list summary at the job level
            tasks_summary_file = os.path.join(self.job_manager.get_job_folder(), 'training_tasks_summary.json')
            serializable_tasks = [{k: v for k, v in task.items() if k != 'model'} for task in task_list]
            
            with open(tasks_summary_file, 'w') as f:
                json.dump(serializable_tasks, f, indent=4)

            self.logger.info(f"Created {len(task_list)} training tasks and saved task information to disk.")
            return task_list

        except Exception as e:
            self.logger.error(f"Error creating training tasks: {e}")
            raise

    def _create_task_info(self, model_task, hyperparams, repetition):
        """Helper method to create a task info dictionary."""
        timestamp = time.strftime("%Y%m%d%H%M%S")
        task_counter = getattr(self, '_task_counter', 0) + 1
        self._task_counter = task_counter
        
        # Create task directory with relevant parameters and repetition number
        scheduler_type = hyperparams['SCHEDULER_TYPE']
        if scheduler_type == 'StepLR':
            task_dir_name = f'lr_{hyperparams["INITIAL_LR"]}_period_{hyperparams["LR_PERIOD"]}_factor_{hyperparams["LR_PARAM"]}'
        else:
            task_dir_name = f'lr_{hyperparams["INITIAL_LR"]}_plat_pat_{hyperparams["PLATEAU_PATIENCE"]}_factor_{hyperparams["PLATEAU_FACTOR"]}'
        
        task_dir = os.path.join(
            model_task['model_dir'],
            f'{task_dir_name}_vp_{hyperparams["VALID_PATIENCE"]}_lb_{hyperparams["LOOKBACK"]}_bs_{hyperparams["BATCH_SIZE"]}_rep_{repetition}'
        )
        os.makedirs(task_dir, exist_ok=True)

        # Create logs directory within task directory
        logs_dir = os.path.join(task_dir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)

        # Create unique task ID
        task_id = f"task_{timestamp}_{task_counter}_rep_{repetition}"

        # Calculate num_learnable_params once to use in both places
        num_learnable_params = self.calculate_learnable_parameters(
            model_task['hyperparams']['LAYERS'],
            model_task['hyperparams']['INPUT_SIZE'],
            model_task['hyperparams']['HIDDEN_UNITS']
        )

        return {
            'task_id': task_id,
            'model': model_task['model'],
            'model_dir': task_dir,
            'task_dir': task_dir,
            'model_path': os.path.join(task_dir, 'model.pth'),
            'logs_dir': logs_dir,
            'hyperparams': {
                # Ensure all required hyperparameters are explicitly included
                'LAYERS': model_task['hyperparams']['LAYERS'],
                'HIDDEN_UNITS': model_task['hyperparams']['HIDDEN_UNITS'],
                'INPUT_SIZE': model_task['hyperparams']['INPUT_SIZE'],
                'OUTPUT_SIZE': model_task['hyperparams']['OUTPUT_SIZE'],
                'BATCH_SIZE': hyperparams['BATCH_SIZE'],
                'MAX_EPOCHS': hyperparams['MAX_EPOCHS'],
                'INITIAL_LR': hyperparams['INITIAL_LR'],
                'VALID_PATIENCE': hyperparams['VALID_PATIENCE'],
                'ValidFrequency': hyperparams['ValidFrequency'],
                'LOOKBACK': hyperparams['LOOKBACK'],
                'SCHEDULER_TYPE': hyperparams['SCHEDULER_TYPE'],
                # Scheduler-specific parameters
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
                'num_workers': 4  # Make this configurable if needed
            },
            'csv_log_file': os.path.join(logs_dir, 'training_progress.csv'),
            'db_log_file': os.path.join(logs_dir, f'{task_id}_training.db'),
            'model_metadata': {  # Add metadata for easier task management
                'model_type': model_task.get('model_type', 'LSTM'),
                'num_learnable_params': num_learnable_params
            }
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


