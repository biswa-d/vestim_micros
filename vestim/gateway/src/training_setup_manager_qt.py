
import os, uuid
import json
from vestim.gateway.src.hyper_param_manager_qt import VEstimHyperParamManager
from vestim.services.model_training.src.LSTM_model_service import LSTMModelService
from vestim.services.model_training.src.FNN_model_service import FNNModelService
from vestim.services.model_training.src.GRU_model_service import GRUModelService
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
            self.lstm_model_service = LSTMModelService()
            self.fnn_model_service = FNNModelService()
            self.gru_model_service = GRUModelService()
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

    def build_models(self):
        """Build and store models based on hyperparameters."""
        self.models = [] # Clear previous models
        
        # self.params is from hyper_param_manager.get_hyper_params() and should be clean for the current MODEL_TYPE
        # due to changes in VEstimHyperParamManager.update_params
        current_model_params = self.params
        self.logger.info(f"Starting build_models with current_model_params: {current_model_params}")

        model_type_str = current_model_params.get('MODEL_TYPE', 'LSTM').upper()
        self.logger.info(f"Determined MODEL_TYPE for build: {model_type_str}")

        feature_cols_str = current_model_params.get('FEATURE_COLS', 'SOC,Current,Temp')
        try:
            feature_cols_list = [col.strip() for col in feature_cols_str.split(',')]
            input_size = len(feature_cols_list)
            if input_size == 0:
                raise ValueError("FEATURE_COLS cannot be empty.")
        except Exception as e:
            self.logger.error(f"Error parsing FEATURE_COLS '{feature_cols_str}': {e}. Using default input_size=3.")
            input_size = 3 # Fallback

        if model_type_str == 'LSTM':
            # Directly get LSTM specific params from current_model_params, expecting them to be present and correct
            hidden_units_list_str = current_model_params.get('HIDDEN_UNITS', '64') # Should be present if MODEL_TYPE is LSTM
            layers_list_str = current_model_params.get('LAYERS', '1')
            dropout_prob_list_str = current_model_params.get('DROPOUT_PROB', '0.0')

            hidden_units_list = [int(h.strip()) for h in hidden_units_list_str.split(',')]
            layers_list = [int(l.strip()) for l in layers_list_str.split(',')]
            dropout_prob_list = [float(d.strip()) for d in dropout_prob_list_str.split(',')]

            for hidden_units in hidden_units_list:
                for layers in layers_list:
                    for dropout_prob in dropout_prob_list:
                        model_identifier = f"model_lstm_hu_{hidden_units}_layers_{layers}_dropout_{dropout_prob}"
                        model_dir = os.path.join(self.job_manager.get_job_folder(), 'models', model_identifier)
                        os.makedirs(model_dir, exist_ok=True)
                        model_path = os.path.join(model_dir, "model.pth")
                        model_build_params = {
                            "INPUT_SIZE": input_size, "HIDDEN_UNITS": hidden_units,
                            "LAYERS": layers, "DROPOUT_PROB": dropout_prob
                        }
                        model = self.lstm_model_service.create_and_save_lstm_model(model_build_params, model_path)
                        self.models.append({'model': model, 'model_type': 'LSTM', 'model_dir': model_dir,
                                            'model_path': model_path, 'build_params': model_build_params})
        
        elif model_type_str == 'FNN':
            # Directly get FNN specific params from current_model_params
            fnn_hidden_layers_configs_str = current_model_params.get('FNN_HIDDEN_LAYERS', '128,64') # Should be present
            fnn_dropout_list_str = current_model_params.get('FNN_DROPOUT_PROB', '0.0') # Should be present

            fnn_hidden_layers_configs = fnn_hidden_layers_configs_str.split(';')
            fnn_dropout_list = [float(d.strip()) for d in fnn_dropout_list_str.split(',')]

            for config_str in fnn_hidden_layers_configs:
                hidden_layer_sizes = [int(s.strip()) for s in config_str.split(',')]
                for dropout_prob in fnn_dropout_list:
                    layers_str = "_".join(map(str, hidden_layer_sizes))
                    model_identifier = f"model_fnn_layers_{layers_str}_dropout_{dropout_prob}"
                    model_dir = os.path.join(self.job_manager.get_job_folder(), 'models', model_identifier)
                    os.makedirs(model_dir, exist_ok=True)
                    model_path = os.path.join(model_dir, "model.pth")
                    model_build_params = {
                        "INPUT_SIZE": input_size, "HIDDEN_LAYER_SIZES": hidden_layer_sizes,
                        "OUTPUT_SIZE": 1, "DROPOUT_PROB": dropout_prob
                    }
                    if hasattr(self, 'fnn_model_service') and self.fnn_model_service is not None:
                        model = self.fnn_model_service.create_and_save_fnn_model(model_build_params, model_path)
                        self.models.append({'model': model, 'model_type': 'FNN', 'model_dir': model_dir,
                                            'model_path': model_path, 'build_params': model_build_params})
                    else:
                        self.logger.warning("FNNModelService not available. Skipping FNN model creation.")
                        self.models.append({'model': None, 'model_type': 'FNN', 'model_dir': model_dir,
                                            'model_path': model_path, 'build_params': model_build_params})

        elif model_type_str == 'GRU':
            # Directly get GRU specific params
            hidden_units_list_str = current_model_params.get('GRU_HIDDEN_UNITS', current_model_params.get('HIDDEN_UNITS', '64'))
            layers_list_str = current_model_params.get('GRU_LAYERS', current_model_params.get('LAYERS', '1'))
            dropout_prob_list_str = current_model_params.get('GRU_DROPOUT_PROB', current_model_params.get('DROPOUT_PROB', '0.0'))

            hidden_units_list = [int(h.strip()) for h in hidden_units_list_str.split(',')]
            layers_list = [int(l.strip()) for l in layers_list_str.split(',')]
            dropout_prob_list = [float(d.strip()) for d in dropout_prob_list_str.split(',')]

            for hidden_units in hidden_units_list:
                for layers in layers_list:
                    for dropout_prob in dropout_prob_list:
                        model_identifier = f"model_gru_hu_{hidden_units}_layers_{layers}_dropout_{dropout_prob}"
                        model_dir = os.path.join(self.job_manager.get_job_folder(), 'models', model_identifier)
                        os.makedirs(model_dir, exist_ok=True)
                        model_path = os.path.join(model_dir, "model.pth")
                        model_build_params = {
                            "INPUT_SIZE": input_size, "HIDDEN_UNITS": hidden_units,
                            "NUM_LAYERS": layers, "OUTPUT_SIZE": 1, "DROPOUT_PROB": dropout_prob
                        }
                        if hasattr(self, 'gru_model_service') and self.gru_model_service is not None:
                            model = self.gru_model_service.create_and_save_gru_model(model_build_params, model_path)
                            self.models.append({'model': model, 'model_type': 'GRU', 'model_dir': model_dir,
                                                'model_path': model_path, 'build_params': model_build_params})
                        else:
                            self.logger.warning("GRUModelService not available. Skipping GRU model creation.")
                            self.models.append({'model': None, 'model_type': 'GRU', 'model_dir': model_dir,
                                                'model_path': model_path, 'build_params': model_build_params})
        else:
            self.logger.error(f"Unsupported MODEL_TYPE: {model_type_str}")
            raise ValueError(f"Unsupported MODEL_TYPE: {model_type_str}")
        
        if not self.models:
            self.logger.warning("No models were built. Check hyperparameter configurations.")
        self.logger.info(f"Model building complete. {len(self.models)} model configurations prepared.")


    def create_training_tasks(self):
        """
        Create a list of training tasks by combining models and relevant training hyperparameters.
        This method will also save the task information to disk for future use.
        """
        self.logger.info("Creating training tasks...")
        task_list = []  # Initialize a list to store tasks

        # Retrieve relevant hyperparameters for training
        # Data Loader related params from self.current_hyper_params
        training_method = self.current_hyper_params.get('TRAINING_METHOD', 'SequenceRNN') # Default
        feature_cols_str = self.current_hyper_params.get('FEATURE_COLS', 'SOC,Current,Temp')
        feature_cols = [col.strip() for col in feature_cols_str.split(',')]
        target_col = self.current_hyper_params.get('TARGET_COL', 'Voltage')
        concatenate_raw_data = self.current_hyper_params.get('CONCATENATE_RAW_DATA', 'False').lower() == 'true'
        
        # Training loop related params
        learning_rates = [float(lr.strip()) for lr in self.current_hyper_params.get('INITIAL_LR', '0.001').split(',')]
        lr_drop_factors = [float(df.strip()) for df in self.current_hyper_params.get('LR_DROP_FACTOR', '0.1').split(',')]
        lr_drop_periods = [int(dp.strip()) for dp in self.current_hyper_params.get('LR_DROP_PERIOD', '10').split(',')]
        valid_patience_values = [int(vp.strip()) for vp in self.current_hyper_params.get('VALID_PATIENCE', '5').split(',')]
        repetitions = int(self.current_hyper_params.get('REPETITIONS', '1'))
        lookbacks = [int(lb.strip()) for lb in self.current_hyper_params.get('LOOKBACK', '50').split(',')] # Still relevant for SequenceRNN
        batch_sizes = [int(bs.strip()) for bs in self.current_hyper_params.get('BATCH_SIZE', '32').split(',')]
        max_epochs = int(self.current_hyper_params.get('MAX_EPOCHS', '100'))
        valid_frequency = int(self.current_hyper_params.get('ValidFrequency', '1')) # Renamed from ValidFrequency for consistency
        weight_decay_list = [float(wd.strip()) for wd in self.current_hyper_params.get('WEIGHT_DECAY', '0.0').split(',')]


        for model_config in self.models: # model_config from self.build_models()
            model_object = model_config['model'] # This can be None if FNN/GRU service is not ready
            model_type = model_config['model_type']
            model_build_params = model_config['build_params']

            # Prepare model_metadata based on actual model_type and its parameters
            model_metadata = {'model_type': model_type, **model_build_params}
            
            # Calculate learnable parameters (needs to be model-specific)
            num_learnable_params = 0
            if model_object: # Only if model was successfully built
                if model_type in ['LSTM', 'GRU', 'FNN']:
                    num_learnable_params = sum(p.numel() for p in model_object.parameters() if p.requires_grad)
                # Add specific calculation if needed, or rely on PyTorch's sum(p.numel())
            else:
                self.logger.warning(f"Model object is None for a {model_type} config. num_learnable_params set to 0.")


            # Iterate through training hyperparameters
            for lr in learning_rates:
                for wd in weight_decay_list:
                    for drop_factor in lr_drop_factors:
                        for drop_period in lr_drop_periods:
                            for patience in valid_patience_values:
                                # Lookback is only a varying param if training_method is SequenceRNN
                                current_lookbacks = lookbacks if training_method == 'SequenceRNN' else [None] # Use None or a default if not applicable
                                
                                for lookback_val in current_lookbacks:
                                    for batch_size in batch_sizes:
                                        for rep in range(1, repetitions + 1):
                                            task_id = str(uuid.uuid4())
                                            
                                            # Build task_dir name based on relevant varying params
                                            param_name_parts = [
                                                f"model_{model_type}",
                                                f"lr_{lr}", f"wd_{wd}", f"bs_{batch_size}"
                                            ]
                                            if training_method == 'SequenceRNN' and lookback_val is not None:
                                                param_name_parts.append(f"lb_{lookback_val}")
                                            # Add parts from model_build_params to make dir unique for model architecture
                                            for k,v in model_build_params.items():
                                                if isinstance(v, list): # e.g. FNN_HIDDEN_LAYERS
                                                    param_name_parts.append(f"{k}_{'_'.join(map(str,v))}")
                                                else:
                                                    param_name_parts.append(f"{k}_{v}")
                                            param_name_parts.append(f"rep_{rep}")

                                            task_specific_name = "_".join(param_name_parts)
                                            task_dir = os.path.join(model_config['model_dir'], task_specific_name)
                                            os.makedirs(task_dir, exist_ok=True)
                                            
                                            csv_log_file = os.path.join(task_dir, f"{task_id}_train_log.csv")
                                            db_log_file = os.path.join(task_dir, f"{task_id}_train_log.db")

                                            task_info = {
                                                'task_id': task_id,
                                                'model': model_object, # Can be None if FNN/GRU service not ready
                                                'model_metadata': model_metadata,
                                                'data_loader_params': {
                                                    'training_method': training_method,
                                                    'feature_cols': feature_cols,
                                                    'target_col': target_col,
                                                    'lookback': lookback_val if training_method == 'SequenceRNN' else None,
                                                    'batch_size': batch_size,
                                                    'concatenate_raw_data': concatenate_raw_data,
                                                    # Add other DataLoader params like num_workers, train_split, seed if they vary per task or are global
                                                    'num_workers': self.current_hyper_params.get('NUM_WORKERS', 4),
                                                    'train_split': float(self.current_hyper_params.get('TRAIN_SPLIT', 0.7)),
                                                    'seed': int(self.current_hyper_params.get('SEED', 42)) # Example seed
                                                },
                                                'model_dir': task_dir, # Specific task run dir
                                                'model_path': os.path.join(task_dir, 'model.pth'), # Model will be saved here per task run
                                                'hyperparams': { # These are for the training loop itself
                                                    **model_build_params, # Include model architecture params
                                                    'BATCH_SIZE': batch_size,
                                                    'LOOKBACK': lookback_val if training_method == 'SequenceRNN' else None,
                                                    'INITIAL_LR': lr,
                                                    'WEIGHT_DECAY': wd,
                                                    'LR_DROP_FACTOR': drop_factor,
                                                    'LR_DROP_PERIOD': drop_period,
                                                    'VALID_PATIENCE': patience,
                                                    'ValidFrequency': valid_frequency,
                                                    'REPETITIONS': rep,
                                                    'MAX_EPOCHS': max_epochs,
                                                    'NUM_LEARNABLE_PARAMS': num_learnable_params, # Calculated based on model
                                                    # Pass model_type for TrainingTaskService to adapt model call
                                                    'MODEL_TYPE': model_type
                                                },
                                                'csv_log_file': csv_log_file,
                                                'db_log_file': db_log_file
                                            }
                                            task_list.append(task_info)
                                            task_info_file = os.path.join(task_dir, 'task_info.json')
                                            # Create a serializable version of task_info (model object cannot be directly dumped)
                                            serializable_task_info = {k: v for k, v in task_info.items() if k != 'model'}
                                            serializable_task_info['model_path_from_build'] = model_config['model_path'] # Path where initial model (if any) was saved
                                            with open(task_info_file, 'w') as f:
                                                json.dump(serializable_task_info, f, indent=4)
        self.training_tasks = task_list

        # Optionally, save the entire task list for future reference at the root level
        tasks_summary_file = os.path.join(self.job_manager.get_job_folder(), 'training_tasks_summary.json')
        with open(tasks_summary_file, 'w') as f:
            json.dump([{k: v for k, v in task.items() if k != 'model'} for task in self.training_tasks], f, indent=4)

        # Emit progress signal to notify the GUI
        task_count = len(self.training_tasks)
        if self.progress_signal:
            self.logger.info(f"Created {len(self.training_tasks)} training tasks.")
            self.progress_signal.emit(f"Created {task_count} training tasks and saved to disk.", self.job_manager.get_job_folder(), task_count)

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

        # Input to the first hidden layer (input_size + 1 for bias)
        learnable_params += 4 * (input_size + 1) * hidden_units  # 4 comes from LSTM's gates

        # Hidden layers (recurrent part: previous hidden layer to the next hidden layer)
        for i in range(1, layers):
            learnable_params += 4 * (hidden_units + 1) * hidden_units  # No need to index, it's a single value

        # Last hidden layer to output (assuming 1 output)
        output_size = 1
        learnable_params += hidden_units * output_size

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


