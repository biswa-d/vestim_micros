import os
import json
from vestim.gateway.src.job_manager_qt import JobManager
import logging

class VEstimHyperParamManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(VEstimHyperParamManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.logger = logging.getLogger(__name__)  # Set up logger for this class
            self.job_manager = JobManager()
            self.current_params = {}  # Initialize current_params as None
            # self.param_sets = []  # Initialize param_sets as an empty list
            self.initialized = True
            self.logger.info("VEstimHyperParamManager initialized.")

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
        """Validate the parameter values. Strings are generally kept as is, to be parsed by consuming components."""
        validated_params = {}
        
        # Define categories of parameters for easier handling
        # These are params that are expected to be comma-separated strings of numbers by VEstimTrainingSetupManager
        numeric_string_list_params_int = [
            'LAYERS', 'HIDDEN_UNITS', 'BATCH_SIZE', 'MAX_EPOCHS',
            'LR_DROP_PERIOD', 'VALID_PATIENCE', 'ValidFrequency', 'LOOKBACK', 'REPETITIONS',
            'GRU_LAYERS', 'GRU_HIDDEN_UNITS' # Added GRU params
        ]
        # These are params that are expected to be comma-separated strings of floats
        numeric_string_list_params_float = [
            'INITIAL_LR', 'LR_DROP_FACTOR', 'DROPOUT_PROB',
            'FNN_DROPOUT_PROB', 'GRU_DROPOUT_PROB', # Added FNN/GRU dropout
            'WEIGHT_DECAY' # Added WEIGHT_DECAY
        ]
        
        # String parameters (some might be comma-separated lists of strings, like FEATURE_COLS or FNN_HIDDEN_LAYERS)
        # FNN_HIDDEN_LAYERS can also be "128,64;100,50" - so it's a complex string.
        string_params = [
            'MODEL_TYPE', 'TRAINING_METHOD', 'FEATURE_COLS', 'TARGET_COL',
            'CONCATENATE_RAW_DATA', # Expected "True" or "False"
            'FNN_HIDDEN_LAYERS', # e.g., "128,64" or "100;50,20"
            # Add any other new string-based params here
        ]

        for key, value in params.items():
            if not isinstance(value, str): # All params from GUI QLineEdit/QComboBox come as strings
                self.logger.warning(f"Parameter '{key}' has value '{value}' of type {type(value)}, expected string. Will attempt to convert.")
                # Attempt to convert to string, or handle as error if critical
                try:
                    value = str(value)
                except Exception as e:
                    self.logger.error(f"Could not convert parameter '{key}' to string: {e}")
                    raise ValueError(f"Parameter '{key}' could not be converted to string.")

            validated_params[key] = value.strip() # Store stripped string value

            # Optional: Basic validation for specific string formats if desired here,
            # but detailed parsing is deferred to VEstimTrainingSetupManager.
            if key == 'CONCATENATE_RAW_DATA' and value.lower() not in ['true', 'false', '']:
                self.logger.warning(f"Parameter 'CONCATENATE_RAW_DATA' has value '{value}'. Expected 'True' or 'False'.")
            
            if key in numeric_string_list_params_int or key in numeric_string_list_params_float:
                # For these, VEstimTrainingSetupManager will split by comma and convert.
                # We can do a basic check here if they are empty or contain invalid characters
                # if not value: # Allow empty strings, setup manager can use defaults
                #     continue
                # For example, check if they only contain numbers, commas, spaces, semicolons (for FNN_HIDDEN_LAYERS)
                # This validation can be made more robust if needed.
                pass


        self.logger.info("Parameter validation/normalization (keeping as strings) complete.")
        return validated_params

    def save_params(self):
        """Save the current parameters to the job folder."""
        job_folder = self.job_manager.get_job_folder()
        if job_folder and self.current_params:
            params_file = os.path.join(job_folder, 'hyperparams.json')
            with open(params_file, 'w') as file:
                json.dump(self.current_params, file, indent=4)
                self.logger.info("Parameters successfully saved.")
        else:
            self.logger.error("Failed to save parameters: Job folder or current parameters are not set.")
            raise ValueError("Job folder is not set or current parameters are not available.")

    def save_params_to_file(self, new_params, filepath):
        """Save new parameters to a specified file."""
        with open(filepath, 'w') as file:
            json.dump(new_params, file, indent=4)
        self.logger.info("New parameters successfully saved.")

    def update_params(self, new_params):
        """Update the current parameters with new values."""
        validated_params = self.validate_and_normalize_params(new_params)
        # Set current_params to only the latest validated parameters from the GUI.
        # This ensures that stale keys from previous model type selections are cleared.
        self.current_params = validated_params
        # self.param_sets.append(self.current_params) # If param_sets were for history, this logic might change
        self.logger.info(f"Parameters successfully set to: {self.current_params}")

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
