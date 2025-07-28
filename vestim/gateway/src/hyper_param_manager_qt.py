import os
import json
from vestim.gateway.src.job_manager_qt import JobManager
from vestim.config_manager import update_last_used_hyperparams
import logging

class VEstimHyperParamManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(VEstimHyperParamManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, job_manager=None):
        if not hasattr(self, 'initialized'):
            self.logger = logging.getLogger(__name__)  # Set up logger for this class
            self.job_manager = job_manager if job_manager else JobManager()
            self.current_params = {}  # Initialize current_params as None
            # self.param_sets = []  # Initialize param_sets as an empty list
            self.initialized = True
            self.logger.info("VEstimHyperParamManager initialized.")
    
    def validate_for_optuna(self, params):
        """
        Performs strict validation for Optuna search.
        - Enforces [min,max] boundary format for core tunable parameters.
        - Allows single values for non-core or specified flexible parameters.
        - Rejects comma-separated lists for any tunable parameter.
        Returns (bool, str): (is_valid, error_message)
        """
        self.logger.info("Performing strict validation for Optuna mode.")
        
        model_type = params.get('MODEL_TYPE')
        scheduler_type = params.get('SCHEDULER_TYPE')
        invalid_params = []

        # Define keys that are always required
        always_required = ['BATCH_SIZE', 'MAX_EPOCHS', 'INITIAL_LR']

        # Define model-specific requirements
        model_specific_requirements = {
            'LSTM': ['LAYERS', 'HIDDEN_UNITS', 'LOOKBACK'],
            'GRU': ['GRU_LAYERS', 'GRU_HIDDEN_UNITS', 'LOOKBACK'],
            'FNN': ['FNN_HIDDEN_LAYERS', 'FNN_DROPOUT_PROB']
        }

        # Define scheduler-specific requirements
        scheduler_specific_requirements = {
            'StepLR': ['LR_PARAM', 'LR_PERIOD'],
            'ReduceLROnPlateau': ['PLATEAU_FACTOR', 'PLATEAU_PATIENCE']
        }

        # Combine all keys that need to be validated for boundary format
        strictly_bounded_keys = (
            always_required +
            model_specific_requirements.get(model_type, []) +
            scheduler_specific_requirements.get(scheduler_type, [])
        )
        
        flexible_keys = ['VALID_PATIENCE', 'VALID_FREQUENCY']
        
        keys_to_validate = strictly_bounded_keys + flexible_keys

        for key in keys_to_validate:
            value_str = str(params.get(key) or "").strip()

            # Rule 1: Strictly bounded keys must not be empty and must be in boundary format
            if key in strictly_bounded_keys:
                if not value_str:
                    invalid_params.append(f"'{key}' cannot be empty.")
                    continue
                if not (value_str.startswith('[') and value_str.endswith(']')):
                    invalid_params.append(f"'{key}' must be in [min,max] format for Auto Search.")
                    continue
            
            # Rule 2: FNN_HIDDEN_LAYERS has its own special validation handled by validate_hyperparameters_for_gui
            # We just ensure it's not empty here.
            if key == 'FNN_HIDDEN_LAYERS' and not value_str:
                invalid_params.append(f"'{key}' cannot be empty.")
                continue

            # Rule 3: For all tunable keys, comma-separated lists are invalid
            if ',' in value_str and not (value_str.startswith('[') and value_str.endswith(']')):
                # This check is now more specific. FNN_HIDDEN_LAYERS for grid search can have commas, but not for Optuna.
                # The GUI validation should handle the distinction.
                # For Optuna, we enforce boundaries.
                msg = f"Invalid format for '{key}' in Auto Search mode. Use [min,max] for ranges or a single value, not a comma-separated list."
                self.logger.error(msg)
                return False, msg

            # Rule 3: Flexible keys, if not a boundary, must be a single number
            if key in flexible_keys and value_str and not (value_str.startswith('[') and value_str.endswith(']')):
                try:
                    float(value_str)
                except ValueError:
                    msg = f"Invalid format for '{key}': '{value_str}'. It must be a single number or a range like [min,max]."
                    self.logger.error(msg)
                    return False, msg

        if invalid_params:
            error_message = "The following parameters have errors for Auto Search:\n\n" + "\n".join(invalid_params) + "\n\nPlease correct them to proceed."
            self.logger.error(error_message)
            return False, error_message

        if 'OPTIMIZER_TYPE' in params and isinstance(params['OPTIMIZER_TYPE'], str) and ',' in params['OPTIMIZER_TYPE']:
            return False, "Auto Search (Optuna) does not support multiple optimizers. Please select only one."

        return True, ""

    def validate_for_grid_search(self, params):
        """
        Performs strict validation for Grid Search.
        - Allows comma-separated lists for tunable parameters.
        - Allows single values.
        - Rejects [min,max] boundary format.
        Returns (bool, str): (is_valid, error_message)
        """
        self.logger.info("Performing strict validation for Grid Search mode.")
        model_type = params.get('MODEL_TYPE')
        scheduler_type = params.get('SCHEDULER_TYPE')

        # Define common tunable keys
        common_tunable_keys = ['BATCH_SIZE', 'MAX_EPOCHS', 'INITIAL_LR', 'VALID_PATIENCE']

        # Define model-specific tunable keys
        model_specific_tunable_keys = {
            'LSTM': ['LAYERS', 'HIDDEN_UNITS', 'LOOKBACK'],
            'GRU': ['GRU_LAYERS', 'GRU_HIDDEN_UNITS', 'LOOKBACK'],
            'FNN': ['FNN_HIDDEN_LAYERS', 'FNN_DROPOUT_PROB']
        }

        # Define scheduler-specific tunable keys
        scheduler_specific_tunable_keys = {
            'StepLR': ['LR_PARAM', 'LR_PERIOD'],
            'ReduceLROnPlateau': ['PLATEAU_PATIENCE', 'PLATEAU_FACTOR']
        }

        # Combine keys based on model and scheduler type
        tunable_keys = (
            common_tunable_keys +
            model_specific_tunable_keys.get(model_type, []) +
            scheduler_specific_tunable_keys.get(scheduler_type, [])
        )

        for key in tunable_keys:
            value = params.get(key)
            if isinstance(value, str):
                is_boundary = value.strip().startswith('[') and value.strip().endswith(']')
                if is_boundary:
                    # FNN_HIDDEN_LAYERS in Optuna format is an exception, but this is for Grid Search
                    if key == 'FNN_HIDDEN_LAYERS' and value.count('[') == 2 and value.count(']') == 2:
                         msg = f"Invalid format for '{key}' in Exhaustive Search. For FNN layer ranges, use semicolon-separated lists of units, not Optuna's double-bracket format."
                         self.logger.error(msg)
                         return False, msg
                    elif key != 'FNN_HIDDEN_LAYERS':
                        msg = f"Invalid format for '{key}' in Exhaustive Search mode. Use comma-separated values for lists, not [min,max]."
                        self.logger.error(msg)
                        return False, msg
        return True, ""
    def validate_hyperparameters_for_gui(self, params, search_mode):
        """
        Performs validation specific to the GUI inputs before proceeding.
        This is particularly for complex fields like FNN_HIDDEN_LAYERS.
        Returns (bool, str): (is_valid, error_message)
        """
        model_type = params.get('MODEL_TYPE')

        if model_type == 'FNN':
            fnn_hidden_layers_str = params.get('FNN_HIDDEN_LAYERS', '').strip()
            is_optuna_format = fnn_hidden_layers_str.count('[') == 2 and fnn_hidden_layers_str.count(']') == 2
            is_grid_format = ';' in fnn_hidden_layers_str or ',' in fnn_hidden_layers_str and not is_optuna_format

            if search_mode == 'optuna':
                if not is_optuna_format:
                    return False, "For Auto Search, 'FNN_HIDDEN_LAYERS' must be in the format [min1,min2],[max1,max2]."
                try:
                    import re
                    matches = re.findall(r'\[(.*?)\]', fnn_hidden_layers_str)
                    if len(matches) != 2:
                        return False, "Invalid FNN_HIDDEN_LAYERS format. Expected two lists for min and max bounds."
                    min_bounds_str, max_bounds_str = matches
                    min_bounds = [int(x.strip()) for x in min_bounds_str.split(',') if x.strip()]
                    max_bounds = [int(x.strip()) for x in max_bounds_str.split(',') if x.strip()]
                    if len(min_bounds) != len(max_bounds):
                        return False, "'FNN_HIDDEN_LAYERS' dimension mismatch between min and max bounds."
                    if not min_bounds:
                        return False, "'FNN_HIDDEN_LAYERS' bounds cannot be empty."
                    for i in range(len(min_bounds)):
                        if min_bounds[i] > max_bounds[i]:
                            return False, f"Invalid range in 'FNN_HIDDEN_LAYERS': min value {min_bounds[i]} is greater than max value {max_bounds[i]} for layer {i+1}."
                except (ValueError, IndexError):
                    return False, "Invalid number format in 'FNN_HIDDEN_LAYERS'. Ensure all bounds are integers."

            elif search_mode == 'grid':
                if is_optuna_format:
                    return False, "For Grid Search, 'FNN_HIDDEN_LAYERS' should be a semicolon-separated list (e.g., '128,64;64,32'), not the [min,max] format."
                if fnn_hidden_layers_str:
                    try:
                        architectures = [arch.strip() for arch in fnn_hidden_layers_str.split(';')]
                        for arch in architectures:
                            if not arch: continue
                            # Validate that each part of the architecture is an integer
                            [int(unit.strip()) for unit in arch.split(',')]
                    except ValueError:
                        return False, f"Invalid architecture in 'FNN_HIDDEN_LAYERS' for Grid Search: '{arch}'. Each layer size must be an integer."

        return True, ""

    def load_params(self, filepath):
        """Load and validate parameters from a JSON file."""
        self.logger.info(f"Loading parameters from {filepath}")
        with open(filepath, 'r') as file:
            params = json.load(file)
            validated_params = self.validate_and_normalize_params(params)
            # self.param_sets.append(validated_params)
            self.current_params = validated_params  # Set the current_params to the loaded params
            self.logger.info("Parameters successfully loaded and validated.")
        return validated_params

    def validate_and_normalize_params(self, params):
        """Validate and normalize the parameter values while ensuring type consistency."""
        validated_params = {}
        
        # Get model type for validation context
        model_type = params.get('MODEL_TYPE', 'LSTM')

        for key, value in params.items():
            if isinstance(value, str):
                # Model-type aware parameter validation
                integer_params = ['BATCH_SIZE', 'MAX_EPOCHS', 'LR_DROP_PERIOD',
                                'VALID_PATIENCE', 'VALID_FREQUENCY', 'REPETITIONS',
                                'MAX_TRAIN_HOURS', 'MAX_TRAIN_MINUTES', 'MAX_TRAIN_SECONDS']
                
                # Add model-specific integer parameters
                if model_type in ['LSTM', 'GRU']:
                    integer_params.extend(['LAYERS', 'HIDDEN_UNITS', 'LOOKBACK'])
                elif model_type == 'FNN':
                    # FNN does not use LOOKBACK, so it's not added to integer_params
                    pass
                
                # Keep the original string for parameters that might be comma-separated or boundary format
                if key in integer_params:
                    # Check if it's boundary format [min,max] first
                    if value.strip().startswith('[') and value.strip().endswith(']'):
                        # Boundary format validation for Optuna
                        try:
                            inner = value.strip()[1:-1]
                            parts = [part.strip() for part in inner.split(',')]
                            if len(parts) == 2:
                                int(parts[0])  # Validate min as integer
                                int(parts[1])  # Validate max as integer
                                validated_params[key] = value  # Keep boundary format as string
                            else:
                                raise ValueError("Boundary format must have exactly 2 values")
                        except ValueError:
                            self.logger.error(f"Invalid boundary format for {key}: {value}")
                            raise ValueError(f"Invalid value for {key}: Expected boundary format [min,max] or integers, got {value}")
                    else:
                        # Regular comma-separated validation for grid search
                        value_list = [v.strip() for v in value.replace(',', ' ').split() if v]
                        try:
                            [int(v) for v in value_list]  # Just validate, don't convert
                            validated_params[key] = value  # Keep as string
                        except ValueError:
                            self.logger.error(f"Invalid integer value for {key}: {value}")
                            raise ValueError(f"Invalid value for {key}: Expected integers, got {value}")

                elif key in ['INITIAL_LR', 'LR_DROP_FACTOR', 'DROPOUT_PROB', 'LR_PARAM', 'PLATEAU_FACTOR', 'FNN_DROPOUT_PROB']:
                    # Check if it's boundary format [min,max] first
                    if value.strip().startswith('[') and value.strip().endswith(']'):
                        # Boundary format validation for Optuna
                        try:
                            inner = value.strip()[1:-1]
                            parts = [part.strip() for part in inner.split(',')]
                            if len(parts) == 2:
                                float(parts[0])  # Validate min as float
                                float(parts[1])  # Validate max as float
                                validated_params[key] = value  # Keep boundary format as string
                            else:
                                raise ValueError("Boundary format must have exactly 2 values")
                        except ValueError:
                            self.logger.error(f"Invalid boundary format for {key}: {value}")
                            raise ValueError(f"Invalid value for {key}: Expected boundary format [min,max] or floats, got {value}")
                    else:
                        # Regular comma-separated validation for grid search
                        value_list = [v.strip() for v in value.replace(',', ' ').split() if v]
                        try:
                            [float(v) for v in value_list]  # Just validate, don't convert
                            validated_params[key] = value  # Keep as string
                        except ValueError:
                            self.logger.error(f"Invalid float value for {key}: {value}")
                            raise ValueError(f"Invalid value for {key}: Expected floats, got {value}")

                # ✅ Ensure boolean conversion for checkboxes (if applicable)
                elif key in ['BATCH_TRAINING']:
                    validated_params[key] = value.lower() in ['true', '1', 'yes']

                else:
                    validated_params[key] = value

            elif isinstance(value, list):
                # ✅ Ensure lists retain proper types
                validated_params[key] = value

            else:
                validated_params[key] = value  # Keep as-is for other data types

        # ✅ Feature & Target Columns (No validation needed, comes from UI dropdowns)
        validated_params["FEATURE_COLUMNS"] = params.get("FEATURE_COLUMNS", [])
        validated_params["TARGET_COLUMN"] = params.get("TARGET_COLUMN", "")
        validated_params["MODEL_TYPE"] = params.get("MODEL_TYPE", "")

        self.logger.info("Parameter validation and normalization completed successfully.")
        return validated_params


    def save_params(self):
        """Save the current validated parameters to the job folder in a JSON file."""
        job_folder = self.job_manager.get_job_folder()

        if not job_folder:
            self.logger.error("Job folder is not set. Cannot save parameters.")
            raise ValueError("Job folder is not set or current parameters are unavailable.")

        if not self.current_params:
            self.logger.error("No parameters available to save.")
            raise ValueError("No parameters available for saving.")

        params_file = os.path.join(job_folder, 'hyperparams.json')

        try:
            # Create a copy to modify for saving, especially for max_training_time_seconds
            params_to_save = self.current_params.copy()

            # Calculate max_training_time_seconds
            try:
                hours = int(params_to_save.get("MAX_TRAIN_HOURS", 0) or 0)
                minutes = int(params_to_save.get("MAX_TRAIN_MINUTES", 0) or 0)
                seconds = int(params_to_save.get("MAX_TRAIN_SECONDS", 0) or 0)
                max_training_time_seconds = (hours * 3600) + (minutes * 60) + seconds
                params_to_save["MAX_TRAINING_TIME_SECONDS"] = max_training_time_seconds
                self.current_params["MAX_TRAINING_TIME_SECONDS"] = max_training_time_seconds # Update in-memory params
                self.logger.info(f"Calculated and stored MAX_TRAINING_TIME_SECONDS: {max_training_time_seconds}")
            except ValueError:
                self.logger.warning("Could not parse MAX_TRAIN_HOURS/MINUTES/SECONDS to integers. MAX_TRAINING_TIME_SECONDS will not be saved or default to 0 if not already present.")
                if "MAX_TRAINING_TIME_SECONDS" not in params_to_save: # Only add if not already there from a previous load
                    params_to_save["MAX_TRAINING_TIME_SECONDS"] = 0


            # Remove individual H, M, S from the dict to be saved to avoid redundancy,
            # as they are primarily GUI input fields.
            params_to_save.pop("MAX_TRAIN_HOURS", None)
            params_to_save.pop("MAX_TRAIN_MINUTES", None)
            params_to_save.pop("MAX_TRAIN_SECONDS", None)

            # ✅ Validate before saving to avoid corrupt JSON
            # Note: validate_and_normalize_params might need adjustment if it expects H/M/S and they are removed
            # For now, we validate self.current_params which still has H/M/S, then save the modified params_to_save
            _ = self.validate_and_normalize_params(self.current_params) # Validate original structure

            with open(params_file, 'w') as file:
                json.dump(params_to_save, file, indent=4) # Save the modified dict

            self.logger.info(f"Hyperparameters successfully saved to file: {params_file}")
            self.logger.info(f"Saved content: {params_to_save}")

            # Also save to defaults config for next time
            try:
                # Use the current_params (which includes GUI fields) for defaults
                update_last_used_hyperparams(self.current_params)
                self.logger.info("Successfully saved hyperparameters as defaults for future use")
            except Exception as e:
                self.logger.warning(f"Failed to save hyperparameters as defaults: {e}")


        except Exception as e:
            self.logger.error(f"Failed to save parameters: {e}")
            raise ValueError(f"Error saving hyperparameters: {e}")

    def save_params_to_file(self, new_params, filepath):
        """Save new parameters to a specified file."""
        with open(filepath, 'w') as file:
            json.dump(new_params, file, indent=4)
        self.logger.info("New parameters successfully saved.")

    def update_params(self, new_params):
        """Update the current parameters with new values."""
        validated_params = self.validate_and_normalize_params(new_params)
        self.current_params.update(validated_params)
        # self.param_sets.append(self.current_params)
        self.logger.info("Parameters successfully updated.")

    def get_current_params(self):
        """Load the parameters from the saved JSON file in the job folder."""
        job_folder = self.job_manager.get_job_folder()
        params_file = os.path.join(job_folder, 'hyperparams.json')
        
        if os.path.exists(params_file):
            with open(params_file, 'r') as file:
                current_params = json.load(file)
                self.current_params = current_params  # Set the current_params to the loaded params
                return current_params
        else:
            self.logger.error(f"Hyperparameters file not found in {job_folder}")
            raise FileNotFoundError("Hyperparameters JSON file not found in the job folder.")

    def get_hyper_params(self):
        """Return the current hyperparameters stored in memory."""
        if self.current_params:
            self.logger.info("Returning current hyperparameters.")
            return self.current_params
        else:
            self.logger.error("No current parameters available in memory.")
            raise ValueError("No current parameters are available in memory.")
