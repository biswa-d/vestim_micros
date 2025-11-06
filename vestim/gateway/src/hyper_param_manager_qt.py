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
        # For LSTM/GRU: Either RNN_LAYER_SIZES OR (LAYERS + HIDDEN_UNITS) must be present
        model_specific_requirements = {
            'LSTM': ['LOOKBACK'],  # RNN_LAYER_SIZES or (LAYERS + HIDDEN_UNITS) checked separately
            'GRU': ['LOOKBACK'],   # RNN_LAYER_SIZES or (LAYERS + HIDDEN_UNITS) checked separately
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
        
        # Check RNN architecture parameters for LSTM/GRU
        if model_type in ['LSTM', 'GRU']:
            rnn_layer_sizes = params.get('RNN_LAYER_SIZES')
            has_legacy = params.get('LAYERS') or params.get('HIDDEN_UNITS')
            
            if not rnn_layer_sizes and not has_legacy:
                invalid_params.append("For RNN models, either 'RNN_LAYER_SIZES' or both 'LAYERS' and 'HIDDEN_UNITS' must be provided.")
            elif rnn_layer_sizes:
                # If using RNN_LAYER_SIZES, add it to validation
                strictly_bounded_keys.append('RNN_LAYER_SIZES')
            else:
                # If using legacy params, add them to validation
                strictly_bounded_keys.extend(['LAYERS', 'HIDDEN_UNITS'])
        
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
            
            # Rule 2: FNN_HIDDEN_LAYERS and RNN_LAYER_SIZES have special validation
            # Support both single boundary [min,max] and double bracket format [[min1,min2],[max1,max2]]
            if key in ['FNN_HIDDEN_LAYERS', 'RNN_LAYER_SIZES']:
                if not value_str:
                    invalid_params.append(f"'{key}' cannot be empty.")
                    continue
                # Check if it's the dynamic architecture format [[...],[...]]
                if value_str.count('[') == 2 and value_str.count(']') == 2:
                    # Valid dynamic format, skip further validation
                    continue
                # Otherwise should be boundary format [min,max]
                if not (value_str.startswith('[') and value_str.endswith(']')):
                    invalid_params.append(f"'{key}' must be in [min,max] or [[min1,min2],[max1,max2]] format for Auto Search.")
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
            'LSTM': ['RNN_LAYER_SIZES', 'LAYERS', 'HIDDEN_UNITS', 'LOOKBACK'],  # RNN_LAYER_SIZES takes precedence
            'GRU': ['RNN_LAYER_SIZES', 'GRU_LAYERS', 'GRU_HIDDEN_UNITS', 'LOOKBACK'],  # RNN_LAYER_SIZES takes precedence
            'FNN': ['FNN_HIDDEN_LAYERS', 'FNN_DROPOUT_PROB']  # FNN does not use LOOKBACK
        }

        # Define scheduler-specific tunable keys
        scheduler_specific_tunable_keys = {
            'StepLR': ['LR_PARAM', 'LR_PERIOD'],
            'ReduceLROnPlateau': ['PLATEAU_PATIENCE', 'PLATEAU_FACTOR'],
            'CosineAnnealingWarmRestarts': ['COSINE_T0', 'COSINE_T_MULT', 'COSINE_ETA_MIN']
        }

        # Combine keys based on model and scheduler type
        tunable_keys = (
            common_tunable_keys +
            model_specific_tunable_keys.get(model_type, []) +
            scheduler_specific_tunable_keys.get(scheduler_type, [])
        )
        
        # Also validate ALL scheduler parameters regardless of current scheduler selection
        # because grid search code checks all params with commas
        all_scheduler_params = ['LR_PARAM', 'LR_PERIOD', 'PLATEAU_PATIENCE', 'PLATEAU_FACTOR', 
                                'COSINE_T0', 'COSINE_T_MULT', 'COSINE_ETA_MIN']
        tunable_keys = tunable_keys + all_scheduler_params

        for key in tunable_keys:
            value = params.get(key)
            if isinstance(value, str):
                value_stripped = value.strip()
                # Reject any bracket notation in grid search
                if '[' in value_stripped or ']' in value_stripped:
                    # FNN_HIDDEN_LAYERS with semicolons is allowed (e.g., "64,32;128,64")
                    # But reject Optuna format [min],[max] even for FNN_HIDDEN_LAYERS
                    if key == 'FNN_HIDDEN_LAYERS' and ';' in value_stripped:
                        # Grid search format for FNN - allow it
                        pass
                    else:
                        msg = f"Invalid format for '{key}' in Exhaustive Search mode. Brackets [] are not allowed. Use comma-separated values (e.g., '5,10,15') or semicolon-separated architectures for FNN_HIDDEN_LAYERS (e.g., '64,32;128,64')."
                        self.logger.error(msg)
                        return False, msg
        
        # Additional check: FNN should not have LOOKBACK parameter
        if model_type == 'FNN' and 'LOOKBACK' in params:
            lookback_val = params.get('LOOKBACK', '').strip()
            if lookback_val:  # Only reject if not empty
                msg = "Invalid parameter 'LOOKBACK' for FNN model. LOOKBACK is only applicable to RNN models (LSTM/GRU)."
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
                    # Check if min_bounds[i] <= max_bounds[i] for all i
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
                    integer_params.extend(['LAYERS', 'HIDDEN_UNITS', 'LOOKBACK', 'GRU_LAYERS', 'GRU_HIDDEN_UNITS'])
                    # Note: RNN_LAYER_SIZES is a string parameter (comma-separated) and NOT treated as integer
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

                # Ensure boolean conversion for checkboxes (if applicable)
                elif key in ['BATCH_TRAINING']:
                    validated_params[key] = value.lower() in ['true', '1', 'yes']

                else:
                    validated_params[key] = value

            elif isinstance(value, list):
                # Ensure lists retain proper types
                validated_params[key] = value

            else:
                validated_params[key] = value  # Keep as-is for other data types

        # Feature and Target Columns (no validation needed, comes from UI dropdowns)
        validated_params["FEATURE_COLUMNS"] = params.get("FEATURE_COLUMNS", [])
        validated_params["TARGET_COLUMN"] = params.get("TARGET_COLUMN", "")
        validated_params["MODEL_TYPE"] = params.get("MODEL_TYPE", "")
        
        # Add inference filter parameters
        validated_params["INFERENCE_FILTER_TYPE"] = params.get("INFERENCE_FILTER_TYPE", "None")
        validated_params["INFERENCE_FILTER_WINDOW_SIZE"] = params.get("INFERENCE_FILTER_WINDOW_SIZE", "101")
        validated_params["INFERENCE_FILTER_ALPHA"] = params.get("INFERENCE_FILTER_ALPHA", "0.1")
        validated_params["INFERENCE_FILTER_POLYORDER"] = params.get("INFERENCE_FILTER_POLYORDER", "2")

        # === Critical validation for learning rates ===
        initial_lr_str = validated_params.get('INITIAL_LR', '')
        if initial_lr_str:
            # Check if it's boundary format or actual value
            if not (initial_lr_str.strip().startswith('[') and initial_lr_str.strip().endswith(']')):
                try:
                    initial_lr_val = float(initial_lr_str)
                    if initial_lr_val <= 0:
                        error_msg = f"INITIAL_LR must be positive, got {initial_lr_val}. Please enter a valid learning rate (e.g., 0.001, 0.0001)."
                        self.logger.error(error_msg)
                        raise ValueError(error_msg)
                except ValueError as e:
                    if "could not convert" in str(e):
                        error_msg = f"INITIAL_LR must be a valid number, got '{initial_lr_str}'"
                        self.logger.error(error_msg)
                        raise ValueError(error_msg)
                    else:
                        raise
        
        # === Validation for exploit parameters ===
        exploit_epochs_str = validated_params.get('EXPLOIT_EPOCHS', '')
        exploit_lr_str = validated_params.get('EXPLOIT_LR', '')
        
        # Determine if exploitation is enabled
        exploit_enabled = False
        if exploit_epochs_str and not (exploit_epochs_str.strip().startswith('[') and exploit_epochs_str.strip().endswith(']')):
            try:
                exploit_epochs_val = int(exploit_epochs_str)
                if exploit_epochs_val > 0:
                    exploit_enabled = True
            except (ValueError, TypeError):
                pass
        
        # If exploit is enabled, validate EXPLOIT_LR
        if exploit_enabled:
            if not exploit_lr_str or exploit_lr_str.strip() == '':
                error_msg = "EXPLOIT_LR is required when EXPLOIT_EPOCHS > 0. Please enter a learning rate for the exploitation phase (e.g., 1e-5)."
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Validate EXPLOIT_LR is positive
            if not (exploit_lr_str.strip().startswith('[') and exploit_lr_str.strip().endswith(']')):
                try:
                    exploit_lr_val = float(exploit_lr_str)
                    if exploit_lr_val <= 0:
                        error_msg = f"EXPLOIT_LR must be positive, got {exploit_lr_val}. Please enter a valid learning rate."
                        self.logger.error(error_msg)
                        raise ValueError(error_msg)
                except ValueError as e:
                    if "could not convert" in str(e):
                        error_msg = f"EXPLOIT_LR must be a valid number, got '{exploit_lr_str}'"
                        self.logger.error(error_msg)
                        raise ValueError(error_msg)
                    else:
                        raise
        
        # Validate FINAL_LR if provided (optional parameter)
        final_lr_str = validated_params.get('FINAL_LR', '')
        if final_lr_str and final_lr_str.strip() != '':
            if not (final_lr_str.strip().startswith('[') and final_lr_str.strip().endswith(']')):
                try:
                    final_lr_val = float(final_lr_str)
                    if final_lr_val < 0:
                        error_msg = f"FINAL_LR must be non-negative, got {final_lr_val}."
                        self.logger.error(error_msg)
                        raise ValueError(error_msg)
                except ValueError as e:
                    if "could not convert" in str(e):
                        error_msg = f"FINAL_LR must be a valid number, got '{final_lr_str}'"
                        self.logger.error(error_msg)
                        raise ValueError(error_msg)
                    else:
                        raise

        # === General numeric constraints (catch accidental zeros/negatives) ===
        # Skip fields using Optuna boundary format; enforce on single or comma-separated values
        rules = {
            # Core training
            'MAX_EPOCHS': {'type': 'int', 'min': 1},
            'BATCH_SIZE': {'type': 'int', 'min': 1},
            'VALID_PATIENCE': {'type': 'int', 'min': 1},
            'VALID_FREQUENCY': {'type': 'int', 'min': 1},
            'REPETITIONS': {'type': 'int', 'min': 1},
            # Sequence & batching
            'LOOKBACK': {'type': 'int', 'min': 1},
            # Scheduler specifics
            'LR_PERIOD': {'type': 'int', 'min': 1},
            'LR_PARAM': {'type': 'float', 'min': 1e-12, 'max': 1 - 1e-12},
            'PLATEAU_PATIENCE': {'type': 'int', 'min': 1},
            'PLATEAU_FACTOR': {'type': 'float', 'min': 1e-12, 'max': 1 - 1e-12},
            'COSINE_T0': {'type': 'int', 'min': 1},
            'COSINE_T_MULT': {'type': 'int', 'min': 1},
            'COSINE_ETA_MIN': {'type': 'float', 'min': 0.0},
            # Device and data loading
            'NUM_WORKERS': {'type': 'int', 'min': 0},
            'PREFETCH_FACTOR': {'type': 'int', 'min': 1},
            # Regularization / dropout
            'WEIGHT_DECAY': {'type': 'float', 'min': 0.0},
            'FNN_DROPOUT_PROB': {'type': 'float', 'min': 0.0, 'max': 1.0},
            # Inference filter params
            'INFERENCE_FILTER_WINDOW_SIZE': {'type': 'int', 'min': 1},
            'INFERENCE_FILTER_ALPHA': {'type': 'float', 'min': 0.0, 'max': 1.0},
            'INFERENCE_FILTER_POLYORDER': {'type': 'int', 'min': 1},
            # Time fields (HH:MM:SS)
            'MAX_TRAIN_HOURS': {'type': 'int', 'min': 0},
            'MAX_TRAIN_MINUTES': {'type': 'int', 'min': 0, 'max': 59},
            'MAX_TRAIN_SECONDS': {'type': 'int', 'min': 0, 'max': 59},
        }

        numeric_errors = []
        for key, rule in rules.items():
            raw = validated_params.get(key)
            if raw is None:
                continue
            if not isinstance(raw, str):
                # Non-string (already normalized) values: skip here
                continue
            value_str = raw.strip()
            if value_str == '':
                continue
            if value_str.startswith('[') and value_str.endswith(']'):
                # Boundary format: defer to search-time parsing
                continue
            # Validate each token (comma or semicolon separated)
            tokens = [t.strip() for t in value_str.replace(';', ',').split(',') if t.strip()]
            want_int = (rule.get('type') == 'int')
            min_v = rule.get('min')
            max_v = rule.get('max')
            for tok in tokens:
                try:
                    val = int(tok) if want_int else float(tok)
                except ValueError:
                    numeric_errors.append(f"'{key}' contains an invalid {'integer' if want_int else 'number'}: '{tok}'")
                    continue
                if min_v is not None and val < min_v:
                    numeric_errors.append(f"'{key}' must be >= {min_v}, got {val}")
                if max_v is not None and val > max_v:
                    numeric_errors.append(f"'{key}' must be <= {max_v}, got {val}")

        if numeric_errors:
            msg = "Input validation error(s):\n\n" + "\n".join(numeric_errors)
            self.logger.error(msg)
            raise ValueError(msg)

        self.logger.info("Parameter validation and normalization completed successfully.")
        return validated_params


    def _filter_hyperparams_for_saving(self, hyperparams):
        """Filters and reorders the hyperparameters dictionary for saving."""
        params_to_filter = hyperparams.copy()
        
        # --- Filtering ---
        scheduler_type = params_to_filter.get('SCHEDULER_TYPE')
        if scheduler_type == 'StepLR':
            if 'LR_PARAM' in params_to_filter:
                params_to_filter['LR_DROP_FACTOR'] = params_to_filter.pop('LR_PARAM')
            if 'LR_PERIOD' in params_to_filter:
                params_to_filter['LR_DROP_PERIOD'] = params_to_filter.pop('LR_PERIOD')
            params_to_remove = ['PLATEAU_PATIENCE', 'PLATEAU_FACTOR', 'COSINE_T0', 'COSINE_T_MULT', 'COSINE_ETA_MIN']
        elif scheduler_type == 'ReduceLROnPlateau':
            params_to_remove = ['LR_DROP_PERIOD', 'LR_DROP_FACTOR', 'LR_PERIOD', 'LR_PARAM', 'COSINE_T0', 'COSINE_T_MULT', 'COSINE_ETA_MIN']
        elif scheduler_type == 'CosineAnnealingWarmRestarts':
            params_to_remove = ['LR_DROP_PERIOD', 'LR_DROP_FACTOR', 'LR_PERIOD', 'LR_PARAM', 'PLATEAU_PATIENCE', 'PLATEAU_FACTOR']
        else:
            params_to_remove = ['LR_DROP_PERIOD', 'LR_DROP_FACTOR', 'LR_PERIOD', 'LR_PARAM', 'PLATEAU_PATIENCE', 'PLATEAU_FACTOR', 'COSINE_T0', 'COSINE_T_MULT', 'COSINE_ETA_MIN']
        for param in params_to_remove:
            params_to_filter.pop(param, None)

        if params_to_filter.get('INFERENCE_FILTER_TYPE') == 'None':
            for param in ['INFERENCE_FILTER_WINDOW_SIZE', 'INFERENCE_FILTER_ALPHA', 'INFERENCE_FILTER_POLYORDER']:
                params_to_filter.pop(param, None)

        if int(params_to_filter.get('EXPLOIT_REPETITIONS', 0)) == 0:
            for param in ['EXPLOIT_EPOCHS', 'EXPLOIT_LR', 'FINAL_LR']:
                params_to_filter.pop(param, None)

        model_type = params_to_filter.get('MODEL_TYPE')
        training_method = params_to_filter.get('TRAINING_METHOD')
        if model_type == 'FNN':
            params_to_filter.pop('LAYERS', None)
            params_to_filter.pop('HIDDEN_UNITS', None)
            if training_method == 'WholeSequenceFNN':
                params_to_filter.pop('LOOKBACK', None)
        elif model_type in ['LSTM', 'GRU']:
            params_to_filter.pop('FNN_HIDDEN_LAYERS', None)
            params_to_filter.pop('FNN_ACTIVATION', None)
            params_to_filter.pop('FNN_DROPOUT_PROB', None)

        if int(params_to_filter.get('MAX_TRAINING_TIME_SECONDS', 0)) == 0:
            params_to_filter.pop('MAX_TRAINING_TIME_SECONDS', None)

        # --- Reordering ---
        ordered_params = {}
        order = [
            # Data and Device
            "FEATURE_COLUMNS", "TARGET_COLUMN", "DEVICE_SELECTION", "USE_MIXED_PRECISION",
            # Model
            "MODEL_TYPE", "TRAINING_METHOD", "LOOKBACK",
            # FNN Specific
            "FNN_HIDDEN_LAYERS", "FNN_ACTIVATION", "FNN_DROPOUT_PROB",
            # RNN Specific (NEW: RNN_LAYER_SIZES takes precedence over legacy LAYERS/HIDDEN_UNITS)
            "RNN_LAYER_SIZES", "LAYERS", "HIDDEN_UNITS", "GRU_LAYERS", "GRU_HIDDEN_UNITS",
            # Training Core
            "BATCH_TRAINING", "BATCH_SIZE", "VALID_PATIENCE", "VALID_FREQUENCY", "MAX_EPOCHS", "REPETITIONS",
            # Optimizer
            "OPTIMIZER_TYPE", "WEIGHT_DECAY",
            # Scheduler
            "SCHEDULER_TYPE", "INITIAL_LR", "LR_DROP_FACTOR", "LR_DROP_PERIOD",
            "PLATEAU_PATIENCE", "PLATEAU_FACTOR", "COSINE_T0", "COSINE_T_MULT", "COSINE_ETA_MIN",
            # Exploit Phase
            "EXPLOIT_REPETITIONS", "EXPLOIT_EPOCHS", "EXPLOIT_LR", "FINAL_LR",
            # Inference Filter
            "INFERENCE_FILTER_TYPE", "INFERENCE_FILTER_WINDOW_SIZE", "INFERENCE_FILTER_ALPHA", "INFERENCE_FILTER_POLYORDER",
            # Performance
            "NUM_WORKERS", "PIN_MEMORY", "PREFETCH_FACTOR", "MAX_TRAINING_TIME_SECONDS"
        ]
        
        for key in order:
            if key in params_to_filter:
                ordered_params[key] = params_to_filter.pop(key)
        
        # Add any remaining params that were not in the predefined order
        ordered_params.update(params_to_filter)

        return ordered_params

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
            params_to_process = self.current_params.copy()

            # Calculate max_training_time_seconds
            try:
                hours = int(params_to_process.get("MAX_TRAIN_HOURS", 0) or 0)
                minutes = int(params_to_process.get("MAX_TRAIN_MINUTES", 0) or 0)
                seconds = int(params_to_process.get("MAX_TRAIN_SECONDS", 0) or 0)
                max_training_time_seconds = (hours * 3600) + (minutes * 60) + seconds
                params_to_process["MAX_TRAINING_TIME_SECONDS"] = max_training_time_seconds
                self.current_params["MAX_TRAINING_TIME_SECONDS"] = max_training_time_seconds
                self.logger.info(f"Calculated and stored MAX_TRAINING_TIME_SECONDS: {max_training_time_seconds}")
            except (ValueError, TypeError):
                self.logger.warning("Could not parse MAX_TRAIN_HOURS/MINUTES/SECONDS. MAX_TRAINING_TIME_SECONDS will default to 0.")
                params_to_process["MAX_TRAINING_TIME_SECONDS"] = 0

            params_to_process.pop("MAX_TRAIN_HOURS", None)
            params_to_process.pop("MAX_TRAIN_MINUTES", None)
            params_to_process.pop("MAX_TRAIN_SECONDS", None)

            # Filter the parameters before saving
            params_to_save = self._filter_hyperparams_for_saving(params_to_process)

            with open(params_file, 'w') as file:
                json.dump(params_to_save, file, indent=4)

            self.logger.info(f"Filtered hyperparameters successfully saved to file: {params_file}")
            self.logger.info(f"Saved content: {params_to_save}")

            try:
                update_last_used_hyperparams(self.current_params)
                self.logger.info("Successfully saved current hyperparameters (unfiltered) as defaults for future use")
            except Exception as e:
                self.logger.warning(f"Failed to save hyperparameters as defaults: {e}")

        except Exception as e:
            self.logger.error(f"Failed to save parameters: {e}", exc_info=True)
            raise ValueError(f"Error saving hyperparameters: {e}")

    def save_params_to_file(self, new_params, filepath):
        """Save new parameters to a specified file."""
        with open(filepath, 'w') as file:
            json.dump(new_params, file, indent=4)
        self.logger.info("New parameters successfully saved.")

    def update_params(self, new_params):
        """Update the current parameters with new values - REPLACES old params completely."""
        validated_params = self.validate_and_normalize_params(new_params)
        # Replace entirely instead of merging to prevent stale parameters from persisting
        self.current_params = validated_params.copy()
        # self.param_sets.append(self.current_params)
        self.logger.info("Parameters successfully updated (replaced completely).")

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
