import os, uuid, time
import json
from itertools import product
from vestim.gateway.src.hyper_param_manager_qt import VEstimHyperParamManager
from vestim.services.model_training.src.LSTM_model_service import LSTMModelService
from vestim.services.model_training.src.GRU_model_service import GRUModelService
from vestim.services.model_training.src.FNN_model_service import FNNModelService
from vestim.gateway.src.job_manager_qt import JobManager
import logging
import torch
try:
    import fireducks.pandas as pd
except ImportError:
    import pandas as pd

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
            self.gru_model_service = GRUModelService()  # Initialize GRU model service
            self.fnn_model_service = FNNModelService()  # Initialize FNN model service
            self.job_manager = job_manager  # JobManager should be passed in or initialized separately
            self.models = []  # Store model information
            self.training_tasks = []  # Store created tasks
            self.progress_signal = progress_signal  # Signal to communicate progress with the GUI
            self.initialized = True  # Mark as initialized

    def setup_training(self):
        """Set up the training process for grid search."""
        self.logger.info("Setting up grid search training...")
        try:
            self.params = self.hyper_param_manager.get_hyper_params()
            self.current_hyper_params = self.params
            self.logger.info(f"Params after updating: {self.current_hyper_params}")

            if self.progress_signal:
                self.progress_signal.emit("Building models...", "", 0)

            self.build_models()

            if self.progress_signal:
                self.progress_signal.emit("Creating training tasks...", "", 0)

            self.create_training_tasks()

            task_count = len(self.training_tasks)
            if self.progress_signal:
                self.progress_signal.emit(
                    f"Setup complete! {task_count} tasks created in {self.job_manager.get_job_folder()}.",
                    self.job_manager.get_job_folder(),
                    task_count
                )
        except Exception as e:
            self.logger.error(f"Error during grid search setup: {str(e)}")
            if self.progress_signal:
                self.progress_signal.emit(f"Error during setup: {str(e)}", "", 0)

    def setup_training_from_optuna(self, optuna_configs, base_params):
        """Set up the training process using configurations from Optuna."""
        self.logger.info("Setting up training from Optuna configurations...")
        try:
            if not optuna_configs:
                raise ValueError("Optuna configurations are missing.")
            if not base_params:
                raise ValueError("Base parameters are missing for Optuna setup.")

            # Set the main params from the base_params to make them available to helper methods.
            self.params = base_params
            self.current_hyper_params = self.params

            if self.progress_signal:
                self.progress_signal.emit("Creating training tasks from Optuna configs...")

            self.create_tasks_from_optuna(optuna_configs, base_params)

            task_count = len(self.training_tasks)
            if self.progress_signal:
                self.progress_signal.emit(f"Setup complete! {task_count} Optuna tasks created.")
        except Exception as e:
            self.logger.error(f"Error during Optuna setup: {str(e)}")
            if self.progress_signal:
                self.progress_signal.emit(f"Error during setup: {str(e)}")

    def create_selected_model(self, model_type, model_params, model_path):
        """Creates and saves the selected model based on the dropdown selection."""
        
        model_map = {
            "LSTM": self.lstm_model_service.create_and_save_lstm_model,
            "LSTM_EMA": self.lstm_model_service.create_and_save_lstm_model,
            "LSTM_LPF": self.lstm_model_service.create_and_save_lstm_model,
            "GRU": self.gru_model_service.create_and_save_gru_model,
            "FNN": self.fnn_model_service.create_and_save_fnn_model,
        }
        model_params['MODEL_TYPE'] = model_type

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
            # Pass the target_device to all model creation functions
            return model_map[model_type](model_params, model_path, target_device)
        
        raise ValueError(f"Unsupported model type: {model_type}")

    def build_models(self):
        """Build and store models based on hyperparameters and model type."""
        try:
            model_type = self.params.get("MODEL_TYPE", "")
            
            # Handle model-specific parameter extraction
            if model_type == "GRU":
                # For GRU models, check RNN_LAYER_SIZES first, then GRU-specific parameters
                rnn_layer_sizes = self.params.get('RNN_LAYER_SIZES')
                if rnn_layer_sizes:
                    hidden_units_value = str(rnn_layer_sizes)
                    # Count layers from comma-separated values
                    layers_value = str(len(rnn_layer_sizes.split(',')))
                else:
                    hidden_units_value = str(self.params.get('GRU_HIDDEN_UNITS', '10'))
                    layers_value = str(self.params.get('GRU_LAYERS', '1'))
            elif model_type == "FNN":
                # For FNN models, use FNN-specific parameters
                hidden_units_value = str(self.params.get('FNN_HIDDEN_LAYERS', '128,64'))
                layers_value = "1"  # FNN doesn't use layers like RNN, set to 1 for consistency
            else:
                # For LSTM and other models, check RNN_LAYER_SIZES first, then standard parameters
                rnn_layer_sizes = self.params.get('RNN_LAYER_SIZES')
                if rnn_layer_sizes:
                    hidden_units_value = str(rnn_layer_sizes)
                    # Count layers from comma-separated values
                    layers_value = str(len(rnn_layer_sizes.split(',')))
                else:
                    hidden_units_value = str(self.params.get('HIDDEN_UNITS', '10'))
                    layers_value = str(self.params.get('LAYERS', '1'))

            # **Get input and output feature sizes**
            feature_columns = self.params.get("FEATURE_COLUMNS", [])
            target_column = self.params.get("TARGET_COLUMN", "")

            if not feature_columns or not target_column:
                raise ValueError("Feature columns or target column not set in hyperparameters.")

            input_size = len(feature_columns)  # Number of input features
            output_size = 1  # Assuming a single target variable for regression

            self.logger.info(f"Building {model_type} models with INPUT_SIZE={input_size}, OUTPUT_SIZE={output_size}")

            if model_type == "FNN":
                # For FNN models, hidden_units_value contains layer configurations
                # Support both formats:
                # 1. Semicolon-separated: "128,64;100,50,25" 
                # 2. Bracket format: "[128,64,32], [100,50,25]"
                fnn_configs = []
                
                # Check if using bracket format
                if '[' in hidden_units_value and ']' in hidden_units_value:
                    # Parse bracket format: [128,64,32], [100,50,25]
                    import re
                    bracket_matches = re.findall(r'\[([^\]]+)\]', hidden_units_value)
                    for match in bracket_matches:
                        if match.strip():
                            fnn_configs.append(match.strip())
                else:
                    # Parse semicolon format: 128,64;100,50,25
                    for config in hidden_units_value.split(';'):
                        if config.strip():
                            fnn_configs.append(config.strip())
                
                # If no valid configs found, treat the whole string as a single config
                if not fnn_configs:
                    fnn_configs = [hidden_units_value.strip()]
                
                self.logger.info(f"Found {len(fnn_configs)} FNN configuration(s): {fnn_configs}")
                
                # Process each FNN configuration
                for config_idx, fnn_config in enumerate(fnn_configs):
                    hidden_units = fnn_config  # This is the config string like "128,64"
                    layers = 1  # Not applicable for FNN, but set for consistency
                    
                    self.logger.info(f"Creating FNN model {config_idx + 1}/{len(fnn_configs)} with layer configuration: {hidden_units}")
                    
                    # Create model directory
                    layer_config = str(hidden_units).replace(',', '_')
                    model_dir_name = self._generate_model_dir_name({'MODEL_TYPE': model_type, 'FNN_UNITS': hidden_units})
                    model_dir = os.path.join(self.job_manager.get_job_folder(), 'models', model_dir_name)
                    os.makedirs(model_dir, exist_ok=True)

                    model_name = "untrained_model_template.pth"
                    model_path = os.path.join(model_dir, model_name)

                    # FNN model uses different parameter structure
                    hidden_layer_sizes = [int(h.strip()) for h in str(hidden_units).split(',')]
                    dropout_prob = float(self.params.get('FNN_DROPOUT_PROB', '0.1'))
                    model_params = {
                        "INPUT_SIZE": input_size,
                        "OUTPUT_SIZE": output_size,
                        "HIDDEN_LAYER_SIZES": hidden_layer_sizes,
                        "DROPOUT_PROB": dropout_prob,
                        "normalization_applied": self.load_job_normalization_metadata().get('normalization_applied', False)
                    }

                    # Create and save the model
                    model = self.create_selected_model(model_type, model_params, model_path)

                    # Store model information for FNN
                    self.models.append({
                        'model': model,
                        'model_type': model_type,
                        'model_dir': model_dir,
                        "FEATURE_COLUMNS": feature_columns,
                        "TARGET_COLUMN": target_column,
                        'hyperparams': {
                            'INPUT_SIZE': input_size,
                            'OUTPUT_SIZE': output_size,
                            'HIDDEN_LAYER_SIZES': hidden_layer_sizes,
                            'DROPOUT_PROB': dropout_prob,
                            'model_path': model_path
                        }
                    })
            else:
                # For RNN models (LSTM/GRU)
                # Check if using RNN_LAYER_SIZES (new format: "64,32;128,64")
                rnn_layer_sizes = self.params.get('RNN_LAYER_SIZES')
                
                if rnn_layer_sizes:
                    # New format: Parse semicolon-separated architectures
                    architectures = [arch.strip() for arch in str(rnn_layer_sizes).split(';') if arch.strip()]
                    self.logger.info(f"Found {len(architectures)} RNN architecture(s): {architectures}")
                    
                    for arch_idx, arch_config in enumerate(architectures):
                        self.logger.info(f"Creating {model_type} model {arch_idx + 1}/{len(architectures)} with layer sizes: {arch_config}")
                        
                        # Create model directory
                        model_dir_name = self._generate_model_dir_name({'MODEL_TYPE': model_type, 'RNN_LAYER_SIZES': arch_config})
                        model_dir = os.path.join(self.job_manager.get_job_folder(), 'models', model_dir_name)
                        os.makedirs(model_dir, exist_ok=True)
                        
                        model_name = "untrained_model_template.pth"
                        model_path = os.path.join(model_dir, model_name)
                        
                        # Model parameters with RNN_LAYER_SIZES
                        model_params = {
                            "INPUT_SIZE": input_size,
                            "OUTPUT_SIZE": output_size,
                            "RNN_LAYER_SIZES": arch_config  # Pass the architecture string
                        }
                        
                        # Create the model
                        model = self.create_selected_model(model_type, model_params, model_path)
                        
                        # Parse layer sizes to extract hidden_units and layers for compatibility
                        layer_sizes = [int(x.strip()) for x in arch_config.split(',')]
                        hidden_units = layer_sizes[0]  # Use first layer size
                        num_layers = len(layer_sizes)
                        
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
                                'RNN_LAYER_SIZES': arch_config,
                                'HIDDEN_UNITS': hidden_units,  # For compatibility
                                'LAYERS': num_layers,  # For compatibility
                                'model_path': model_path
                            }
                        })
                else:
                    # Legacy format: Use hidden_units_value and layers_value
                    hidden_units_list = [int(h) for h in hidden_units_value.split(',')]  # Parse hidden units
                    layers_list = [int(l) for l in layers_value.split(',')]  # Parse layers

                    # Iterate over all combinations of hidden_units and layers
                    for hidden_units in hidden_units_list:
                        for layers in layers_list:
                            self.logger.info(f"Creating model with hidden_units: {hidden_units}, layers: {layers}")

                            # Create model directory for RNN models
                            model_dir_name = self._generate_model_dir_name({'MODEL_TYPE': model_type, 'LAYERS': layers, 'HIDDEN_UNITS': hidden_units})
                            model_dir = os.path.join(self.job_manager.get_job_folder(), 'models', model_dir_name)
                            os.makedirs(model_dir, exist_ok=True)

                            model_name = "untrained_model_template.pth"
                            model_path = os.path.join(model_dir, model_name)

                            # Model parameters - handle different model types
                            if model_type == "GRU":
                                # GRU model service expects LAYERS (same as LSTM)
                                model_params = {
                                    "INPUT_SIZE": input_size,
                                    "OUTPUT_SIZE": output_size,
                                    "HIDDEN_UNITS": hidden_units,
                                    "LAYERS": layers  # GRU uses LAYERS, not NUM_LAYERS
                                }
                                self.logger.info(f"GRU model params: {model_params}")
                            else:
                                # LSTM and other models use LAYERS
                                model_params = {
                                    "INPUT_SIZE": input_size,
                                    "OUTPUT_SIZE": output_size,
                                    "HIDDEN_UNITS": hidden_units,
                                "LAYERS": layers,
                                "normalization_applied": self.load_job_normalization_metadata().get('normalization_applied', False)
                            }

                        # Create and save the model
                        model = self.create_selected_model(model_type, model_params, model_path)

                        # Store model information for RNN models
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
        """Create training tasks from grid search."""
        self.create_tasks_from_grid_search()

    def create_tasks_from_optuna(self, best_configs, base_params):
        """Create training tasks from a list of Optuna best configurations."""
        task_list = []
        job_normalization_metadata = self.load_job_normalization_metadata()
        for i, config_data in enumerate(best_configs):
            # Create a complete hyperparameter set by starting with the base
            # and updating it with the optimized values.
            hyperparams = base_params.copy()
            hyperparams.update(config_data['params'])
            trial_number = config_data.get('trial_number', 'N/A')
            rank = i + 1

            # Create a single model instance for this task
            n_best = len(best_configs)
            model_task = self._build_single_model(hyperparams, trial_number, rank, n_best, job_normalization_metadata)
            
            # Create the task info using only the complete hyperparams for this task
            task_info = self._create_task_info(
                model_task=model_task,
                hyperparams=hyperparams,
                repetition=1,  # Each Optuna config is a single task
                job_normalization_metadata=job_normalization_metadata,
                max_training_time_seconds_arg=hyperparams.get('MAX_TRAINING_TIME_SECONDS', 0),
                is_optuna_task=True,
                rank=rank,
                n_best=n_best
            )
            task_list.append(task_info)

        self.training_tasks = task_list
        self.save_tasks_to_files(task_list)
        return task_list

    def create_tasks_from_grid_search(self):
        """Create training tasks from the models built during grid search."""
        task_list = []
        job_normalization_metadata = self.load_job_normalization_metadata()
        max_training_time_seconds_arg = self.params.get('MAX_TRAINING_TIME_SECONDS', 0)

        # Define parameters that can be grid-searched, excluding those handled by model building
        grid_keys = ['MAX_EPOCHS', 'INITIAL_LR', 'LR_PARAM', 'LR_PERIOD', 'PLATEAU_PATIENCE', 'PLATEAU_FACTOR', 'BATCH_SIZE']
        
        param_grid = {}
        for key in grid_keys:
            if key in self.params and isinstance(self.params[key], str) and ',' in self.params[key]:
                values = [v.strip() for v in self.params[key].split(',')]
                param_grid[key] = values
            else:
                if self.params.get(key) is not None:
                    param_grid[key] = [self.params.get(key)]

        grid_param_names = list(param_grid.keys())
        grid_param_values = list(param_grid.values())
        
        distinct_grid_keys = [key for key, values in param_grid.items() if len(values) > 1]

        for model_task in self.models:
            for param_combination_values in product(*grid_param_values):
                param_combination = dict(zip(grid_param_names, param_combination_values))
                
                repetitions = int(self.params.get('REPETITIONS', 1))
                for i in range(1, repetitions + 1):
                    # Combine hyperparameters: start with model-specific, then override with global GUI params, then grid
                    # This ensures GUI params (like DEVICE_SELECTION, NUM_WORKERS) take precedence over model defaults
                    task_hyperparams = {}
                    task_hyperparams.update(model_task['hyperparams'])  # Model-specific params (INPUT_SIZE, etc.)
                    task_hyperparams.update(self.params)  # GUI params override model defaults
                    task_hyperparams.update(param_combination)  # Grid combination overrides everything

                    task_info = self._create_task_info(
                        model_task=model_task,
                        hyperparams=task_hyperparams,
                        repetition=i,
                        job_normalization_metadata=job_normalization_metadata,
                        max_training_time_seconds_arg=max_training_time_seconds_arg,
                        grid_keys=distinct_grid_keys
                    )
                    task_list.append(task_info)

        self.training_tasks = task_list
        self.save_tasks_to_files(task_list)
        return task_list

    def _build_single_model(self, hyperparams, trial_number, rank, n_best, job_normalization_metadata=None):
        """Build a single model based on a given hyperparameter set."""
        model_type = hyperparams.get("MODEL_TYPE", "LSTM")
        input_size = len(hyperparams.get("FEATURE_COLUMNS", []))
        output_size = 1

        # For Optuna, the model directory name includes the rank and model architecture
        model_dir_name = self._generate_model_dir_name(hyperparams, rank=rank, n_best=n_best)
        model_dir = os.path.join(
            self.job_manager.get_job_folder(),
            'models',
            model_dir_name
        )
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "untrained_model_template.pth")

        model_params = {
            "INPUT_SIZE": input_size,
            "OUTPUT_SIZE": output_size,
            "normalization_applied": job_normalization_metadata.get('normalization_applied', False) if job_normalization_metadata else False
        }
        if model_type in ["LSTM", "GRU", "LSTM_EMA", "LSTM_LPF"]:
            # Check for RNN_LAYER_SIZES first (new format), then fall back to legacy
            rnn_layer_sizes = hyperparams.get("RNN_LAYER_SIZES")
            if rnn_layer_sizes:
                model_params["RNN_LAYER_SIZES"] = rnn_layer_sizes
            else:
                # Legacy format
                model_params["HIDDEN_UNITS"] = int(hyperparams.get("HIDDEN_UNITS", hyperparams.get("GRU_HIDDEN_UNITS", 10)))
                model_params["LAYERS"] = int(hyperparams.get("LAYERS", hyperparams.get("GRU_LAYERS", 1)))
        elif model_type == "FNN":
            # For Optuna, FNN_UNITS will be a list of ints. For grid search, FNN_HIDDEN_LAYERS is a string.
            fnn_units = hyperparams.get("FNN_UNITS", hyperparams.get("FNN_HIDDEN_LAYERS"))
            if isinstance(fnn_units, str):
                model_params["HIDDEN_LAYER_SIZES"] = [int(s.strip()) for s in fnn_units.split(',')]
            elif isinstance(fnn_units, int):
                model_params["HIDDEN_LAYER_SIZES"] = [fnn_units]
            else:
                model_params["HIDDEN_LAYER_SIZES"] = fnn_units # Should already be a list of ints
            model_params["DROPOUT_PROB"] = float(hyperparams.get("FNN_DROPOUT_PROB", 0.1))

        model = self.create_selected_model(model_type, model_params, model_path)
        
        return {
            'model': model,
            'model_type': model_type,
            'model_dir': model_dir,
            'task_dir': None, # This will be set in _create_task_info
            "FEATURE_COLUMNS": hyperparams.get("FEATURE_COLUMNS", []),
            "TARGET_COLUMN": hyperparams.get("TARGET_COLUMN", ""),
            'hyperparams': {**model_params, 'model_path': model_path}
        }

    def load_job_normalization_metadata(self):
        """Loads normalization metadata from the job folder."""
        job_folder = self.job_manager.get_job_folder()
        metadata_file_path = os.path.join(job_folder, "job_metadata.json")
        if os.path.exists(metadata_file_path):
            try:
                with open(metadata_file_path, 'r') as f_meta:
                    return json.load(f_meta)
            except Exception as e:
                self.logger.error(f"Error loading job_metadata.json: {e}")
        return {}

    def save_tasks_to_files(self, task_list):
        """Saves task information to JSON files."""
        for task_info in task_list:
            # Save task_info.json in the individual task directory, not the model architecture directory
            task_dir = task_info['task_dir']  # Use task_dir instead of model_dir
            task_info_file = os.path.join(task_dir, 'task_info.json')
            serializable_info = {k: v for k, v in task_info.items() if k != 'model'}
            
            # Ensure the task directory exists before saving
            os.makedirs(task_dir, exist_ok=True)
            
            with open(task_info_file, 'w') as f:
                json.dump(serializable_info, f, indent=4)

        tasks_summary_file = os.path.join(self.job_manager.get_job_folder(), 'training_tasks_summary.json')
        serializable_tasks = [{k: v for k, v in task.items() if k != 'model'} for task in task_list]
        with open(tasks_summary_file, 'w') as f:
            json.dump(serializable_tasks, f, indent=4)

    def _generate_model_dir_name(self, hyperparams, rank=None, n_best=None):
        """Generates a descriptive folder name based on model architecture."""
        try:
            model_type = hyperparams.get('MODEL_TYPE', 'MDL')
            name_parts = []
            if model_type in ['LSTM', 'GRU', 'LSTM_EMA', 'LSTM_LPF']:
                # Check for RNN_LAYER_SIZES first (new format: "64,32")
                rnn_layer_sizes = hyperparams.get('RNN_LAYER_SIZES')
                if rnn_layer_sizes:
                    # Use layer sizes string in folder name (replace commas with underscores)
                    layer_str = str(rnn_layer_sizes).replace(',', '_')
                    name_parts.extend([model_type, f"ARCH_{layer_str}"])
                else:
                    # Fall back to legacy LAYERS + HIDDEN_UNITS
                    layers = int(hyperparams.get('LAYERS', hyperparams.get('GRU_LAYERS', 0)))
                    hidden_units = int(hyperparams.get('HIDDEN_UNITS', hyperparams.get('GRU_HIDDEN_UNITS', 0)))
                    name_parts.extend([model_type, f"L{layers}", f"HU{hidden_units}"])
            elif model_type == 'FNN':
                fnn_units = hyperparams.get('FNN_UNITS', hyperparams.get('HIDDEN_LAYER_SIZES', []))
                hidden_layers_str = '_'.join(map(str, fnn_units)) if isinstance(fnn_units, list) else str(fnn_units).replace(',', '_')
                name_parts.append(f"FNN_{hidden_layers_str}")
            
            sanitized_name = '_'.join(name_parts).replace('.', 'p')
            if rank is not None and n_best is not None:
                return f"rank_{rank}_of_{n_best}_{sanitized_name}"
            return sanitized_name
        except Exception as e:
            self.logger.error(f"Could not generate model folder name: {e}. Falling back to UUID.")
            return f"model_{uuid.uuid4().hex[:8]}"

    def _generate_task_dir_name(self, hyperparams, grid_keys=None):
        """Generates a descriptive folder name based on training parameters."""
        try:
            name_parts = []
            if grid_keys and len(grid_keys) > 1:
                for key in sorted(grid_keys):
                    if key in hyperparams:
                        value = hyperparams[key]
                        prefix = ''.join([c for c in key if c.isupper()]) or key[:2]
                        name_parts.append(f"{prefix}{value}")
            else:
                bs = int(hyperparams.get('BATCH_SIZE', 0))
                scheduler_type = hyperparams.get('SCHEDULER_TYPE', 'StepLR')
                scheduler_map = {'StepLR': 'SLR', 'ReduceLROnPlateau': 'RLROP'}
                scheduler_name = scheduler_map.get(scheduler_type, scheduler_type)
                name_parts.extend([f"B{bs}", f"LR_{scheduler_name}"])
                vp = hyperparams.get('VALID_PATIENCE', 'N/A')
                name_parts.append(f"VP{vp}")
            
            return '_'.join(name_parts).replace('.', 'p')
        except Exception as e:
            self.logger.error(f"Could not generate task folder name: {e}. Falling back to UUID.")
            return f"task_{uuid.uuid4().hex[:8]}"

    def _create_task_info(self, model_task, hyperparams, repetition, job_normalization_metadata=None, max_training_time_seconds_arg=0, is_optuna_task=False, rank=None, n_best=None, grid_keys=None):
        """Helper method to create a task info dictionary."""
        if job_normalization_metadata is None:
            job_normalization_metadata = {}

        model_dir = model_task['model_dir']
        task_dir_name = self._generate_task_dir_name(hyperparams, grid_keys=grid_keys)
        if not is_optuna_task and int(hyperparams.get('REPETITIONS', 1)) > 1:
            task_dir_name += f"_rep_{repetition}"
        
        task_dir = os.path.join(model_dir, task_dir_name)
        os.makedirs(task_dir, exist_ok=True)

        if is_optuna_task:
            task_id = f"rank_{rank}_of_{n_best}"
            model_name = model_dir.split(os.sep)[-1]
        else:
            timestamp = time.strftime("%Y%m%d%H%M%S")
            task_counter = getattr(self, '_task_counter', 0) + 1
            self._task_counter = task_counter
            task_id = f"task_{timestamp}_{task_counter}_rep_{repetition}"
            model_name = None

        # Create logs directory within task directory
        logs_dir = os.path.join(task_dir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)

        
        # Get model architecture parameters - handle both RNN and FNN models
        input_size = model_task['hyperparams']['INPUT_SIZE']
        output_size = model_task['hyperparams']['OUTPUT_SIZE']
        model_type = model_task.get('model_type', 'LSTM')
        
        # For RNN models (LSTM/GRU), use HIDDEN_UNITS and LAYERS
        if model_type in ['LSTM', 'GRU', 'LSTM_EMA', 'LSTM_LPF']:
            # Try to get from model_task hyperparams (may have RNN_LAYER_SIZES or legacy params)
            hidden_units = model_task['hyperparams'].get('HIDDEN_UNITS')
            layers = model_task['hyperparams'].get('LAYERS')
            rnn_layer_sizes_str = model_task['hyperparams'].get('RNN_LAYER_SIZES')
            
            # Parse layer sizes if available for accurate parameter counting
            layer_sizes_list = None
            if rnn_layer_sizes_str:
                layer_sizes_list = [int(x.strip()) for x in str(rnn_layer_sizes_str).split(',')]
                if hidden_units is None:
                    hidden_units = layer_sizes_list[0]
                if layers is None:
                    layers = len(layer_sizes_list)
            
            # If still not present, use fallback defaults
            if hidden_units is None or layers is None:
                hidden_units = 10
                layers = 1
            
            # Calculate parameters with layer_sizes if available for accuracy
            num_learnable_params = self.calculate_learnable_parameters(
                layers,
                input_size,
                hidden_units,
                model_type,
                layer_sizes=layer_sizes_list  # Pass variable layer sizes if available
            )
        else:  # FNN model
            hidden_layer_sizes = model_task['hyperparams']['HIDDEN_LAYER_SIZES']
            dropout_prob = model_task['hyperparams']['DROPOUT_PROB']
            # For FNN, calculate parameters differently
            num_learnable_params = self.calculate_fnn_learnable_parameters(
                input_size,
                hidden_layer_sizes,
                output_size
            )
            # Set dummy values for compatibility with existing code
            hidden_units = len(hidden_layer_sizes)  # Number of layers as a proxy
            layers = 1  # FNN doesn't have "layers" in the RNN sense

        # Build a clean hyperparameter dictionary for the final task
        final_hyperparams = {
            'MODEL_TYPE': model_type,
            'TRAINING_METHOD': hyperparams.get('TRAINING_METHOD', 'Sequence-to-Sequence'),
            'INPUT_SIZE': input_size,
            'OUTPUT_SIZE': output_size,
            'BATCH_TRAINING': hyperparams.get('BATCH_TRAINING', True),
            'BATCH_SIZE': hyperparams['BATCH_SIZE'],
            'MAX_EPOCHS': hyperparams['MAX_EPOCHS'],
            'INITIAL_LR': hyperparams['INITIAL_LR'],
            'VALID_PATIENCE': hyperparams['VALID_PATIENCE'],
            'VALID_FREQUENCY': hyperparams['VALID_FREQUENCY'],
            'SCHEDULER_TYPE': hyperparams['SCHEDULER_TYPE'],
            'REPETITIONS': hyperparams['REPETITIONS'],
            'CURRENT_REPETITION': repetition,
            'NUM_LEARNABLE_PARAMS': num_learnable_params,
            'DEVICE_SELECTION': hyperparams.get('DEVICE_SELECTION', 'cuda:0'),  # Include device selection from GUI
            'NUM_WORKERS': hyperparams.get('NUM_WORKERS', 4),
        }
        final_hyperparams['normalization_applied'] = job_normalization_metadata.get('normalization_applied', False)
        final_hyperparams['FEATURE_COLUMNS'] = model_task['FEATURE_COLUMNS']
        final_hyperparams['TARGET_COLUMN'] = model_task['TARGET_COLUMN']

        # Debug log to confirm device selection is included
        self.logger.info(f"TrainingSetupManager._create_task_info: final_hyperparams DEVICE_SELECTION = {final_hyperparams.get('DEVICE_SELECTION')}")

        # Enhance device name with actual GPU model name for display in GUI
        device_selection = final_hyperparams['DEVICE_SELECTION']
        if device_selection and 'cuda' in device_selection.lower():
            try:
                if torch.cuda.is_available():
                    # Parse GPU index from device string
                    gpu_idx = int(device_selection.split(':')[1]) if ':' in device_selection else 0
                    gpu_name = torch.cuda.get_device_name(gpu_idx)
                    current_device = f"{device_selection.upper()} ({gpu_name})"
                    self.logger.info(f"TrainingSetupManager: Enhanced device name: {device_selection} -> {current_device}")
                else:
                    current_device = device_selection.upper()
                    self.logger.warning(f"TrainingSetupManager: CUDA device {device_selection} selected but CUDA not available")
            except Exception as e:
                current_device = device_selection.upper()
                self.logger.warning(f"TrainingSetupManager: Could not enhance device name {device_selection}: {e}")
        elif device_selection and device_selection.upper() == 'CPU':
            current_device = 'CPU'
        else:
            current_device = device_selection or 'Device'
        
        # Add the enhanced device name to hyperparams for GUI display
        final_hyperparams['CURRENT_DEVICE'] = current_device

        # Add scheduler-specific params
        if final_hyperparams['SCHEDULER_TYPE'] == 'StepLR':
            final_hyperparams['LR_PERIOD'] = hyperparams.get('LR_PERIOD')
            final_hyperparams['LR_PARAM'] = hyperparams.get('LR_PARAM')
        elif final_hyperparams['SCHEDULER_TYPE'] == 'ReduceLROnPlateau':
            final_hyperparams['PLATEAU_PATIENCE'] = hyperparams.get('PLATEAU_PATIENCE')
            final_hyperparams['PLATEAU_FACTOR'] = hyperparams.get('PLATEAU_FACTOR')
        elif final_hyperparams['SCHEDULER_TYPE'] == 'CosineAnnealingWarmRestarts':
            final_hyperparams['COSINE_T0'] = hyperparams.get('COSINE_T0')
            final_hyperparams['COSINE_T_MULT'] = hyperparams.get('COSINE_T_MULT')
            final_hyperparams['COSINE_ETA_MIN'] = hyperparams.get('COSINE_ETA_MIN')

        # Add exploit-phase parameters
        final_hyperparams['EXPLOIT_LR'] = hyperparams.get('EXPLOIT_LR')
        final_hyperparams['EXPLOIT_EPOCHS'] = hyperparams.get('EXPLOIT_EPOCHS')
        final_hyperparams['EXPLOIT_REPETITIONS'] = hyperparams.get('EXPLOIT_REPETITIONS')
        final_hyperparams['FINAL_LR'] = hyperparams.get('FINAL_LR')

        # Add inference filter parameters for all model types
        final_hyperparams['INFERENCE_FILTER_TYPE'] = hyperparams.get('INFERENCE_FILTER_TYPE', 'None')
        final_hyperparams['INFERENCE_FILTER_WINDOW_SIZE'] = hyperparams.get('INFERENCE_FILTER_WINDOW_SIZE', 101)
        final_hyperparams['INFERENCE_FILTER_ALPHA'] = hyperparams.get('INFERENCE_FILTER_ALPHA', 0.1)
        final_hyperparams['INFERENCE_FILTER_POLYORDER'] = hyperparams.get('INFERENCE_FILTER_POLYORDER', 2)

        # Add model-specific and method-specific parameters
        if model_type in ['LSTM', 'GRU', 'LSTM_EMA', 'LSTM_LPF']:
            final_hyperparams['HIDDEN_UNITS'] = hidden_units
            final_hyperparams['LAYERS'] = layers
            final_hyperparams['LOOKBACK'] = hyperparams['LOOKBACK']
        elif model_type == 'FNN':
            final_hyperparams['HIDDEN_LAYER_SIZES'] = model_task['hyperparams']['HIDDEN_LAYER_SIZES']
            final_hyperparams['DROPOUT_PROB'] = model_task['hyperparams']['DROPOUT_PROB']
            # If not a sequence-based FNN, lookback is not applicable
            if final_hyperparams['TRAINING_METHOD'] != 'Sequence-to-Sequence':
                 final_hyperparams['LOOKBACK'] = 'N/A'
            else:
                 final_hyperparams['LOOKBACK'] = hyperparams['LOOKBACK']
            
            # Determine lookback for data loader, defaulting to 0 if not applicable
        if final_hyperparams.get('LOOKBACK') == 'N/A':
            dataloader_lookback = 0
        else:
            dataloader_lookback = hyperparams.get('LOOKBACK', 0)

        task_info = {
            'task_id': task_id,
            'repetition': repetition,
            'model_name': model_name,
            'model': model_task['model'],
            'model_dir': model_task['model_dir'],
            'task_dir': task_dir,
            'model_path': model_task['hyperparams']['model_path'],
            'logs_dir': logs_dir,
            'model_metadata': {
                'model_type': model_task.get('model_type', 'LSTM'),
                'num_learnable_params': num_learnable_params,
                'input_size': input_size,
                'output_size': output_size
            },
            'hyperparams': final_hyperparams,
            'data_loader_params': {
                'lookback': dataloader_lookback,
                'batch_size': hyperparams['BATCH_SIZE'],
                'feature_columns': model_task['FEATURE_COLUMNS'],
                'target_column': model_task['TARGET_COLUMN'],
                'num_workers': hyperparams.get('NUM_WORKERS', 4),  # Use GUI value or default to 4
                'pin_memory': hyperparams.get('PIN_MEMORY', True),
                'prefetch_factor': hyperparams.get('PREFETCH_FACTOR', 2)
            },
            'training_params': {
                'early_stopping': True,
                'early_stopping_patience': hyperparams['VALID_PATIENCE'],
                'save_best_model': True,
                'checkpoint_dir': os.path.join(logs_dir, 'checkpoints'),
                'best_model_path': os.path.join(task_dir, 'best_model.pth'),
                'max_training_time_seconds': max_training_time_seconds_arg
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
            'job_metadata': job_normalization_metadata,
            'job_folder_augmented_from': self.job_manager.get_job_folder()
        }
        return task_info

    def calculate_fnn_learnable_parameters(self, input_size, hidden_layer_sizes, output_size):
        """
        Calculate the number of learnable parameters for FNN models.
        
        :param input_size: Size of input features
        :param hidden_layer_sizes: List of hidden layer sizes
        :param output_size: Size of output
        :return: Total number of learnable parameters
        """
        total_params = 0
        prev_size = input_size
        
        # Calculate parameters for each hidden layer
        for hidden_size in hidden_layer_sizes:
            # Weights: prev_size * hidden_size
            # Biases: hidden_size
            total_params += (prev_size * hidden_size) + hidden_size
            prev_size = hidden_size
        
        # Output layer
        # Weights: prev_size * output_size
        # Biases: output_size
        total_params += (prev_size * output_size) + output_size
        
        return total_params

    def calculate_learnable_parameters(self, layers, input_size, hidden_units, model_type="LSTM", layer_sizes=None):
        """
        Calculate the number of learnable parameters for RNN models (LSTM or GRU).

        :param layers: Number of layers (an integer representing the number of RNN layers)
        :param input_size: The size of the input features (e.g., 3 for [SOC, Current, Temp])
        :param hidden_units: An integer representing the number of hidden units (used if layer_sizes not provided)
        :param model_type: Type of model ("LSTM" or "GRU")
        :param layer_sizes: Optional list of hidden units per layer (e.g., [16, 12] for variable sizes)
        :return: Total number of learnable parameters
        """

        # Initialize the number of parameters
        learnable_params = 0

        if model_type == "GRU":
            # GRU has 3 gates: reset, update, and new gate (fewer than LSTM's 4)
            gates = 3
        else:  # LSTM or default
            # LSTM has 4 gates: input, forget, output, and candidate gates
            gates = 4

        # If layer_sizes provided, use variable layer sizes; otherwise use uniform hidden_units
        if layer_sizes is None:
            layer_sizes = [hidden_units] * layers
        
        # Input-to-hidden weights for the first layer
        # For LSTM: 4 * hidden_units * (input_size + hidden_units)
        # For GRU: 3 * hidden_units * (input_size + hidden_units)
        first_layer_hidden = layer_sizes[0]
        input_layer_params = gates * (input_size + first_layer_hidden) * first_layer_hidden
        input_layer_bias = gates * first_layer_hidden
        learnable_params += input_layer_params + input_layer_bias

        # For each additional layer, input is previous layer's hidden size
        for i in range(1, len(layer_sizes)):
            prev_hidden = layer_sizes[i - 1]
            curr_hidden = layer_sizes[i]
            hidden_layer_params = gates * (prev_hidden + curr_hidden) * curr_hidden
            hidden_layer_bias = gates * curr_hidden
            learnable_params += hidden_layer_params + hidden_layer_bias

        # Output layer (assuming 1 output) - uses last layer's hidden size
        output_size = 1
        last_layer_hidden = layer_sizes[-1]
        output_layer_params = last_layer_hidden * output_size
        output_layer_bias = output_size
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
                
                # Removed TRAIN_VAL_SPLIT as we now use separate train/val/test folders
                
                # String parameters (for potential comma-separated values)
                'INITIAL_LR': str(params['INITIAL_LR']),
                'LR_PARAM': str(params['LR_PARAM']),
                'MAX_EPOCHS': str(params.get('MAX_EPOCHS', '100')),
                # ... other parameters ...
            }
            return validated
        except (ValueError, KeyError) as e:
            raise ValueError(f"Parameter validation failed: {str(e)}")
