import os
import json, ast
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
        """Validate the parameter values and ensure they are correctly formatted."""
        validated_params = {}

        for key, value in params.items():
            if isinstance(value, str):
                # Handle arrays like [1e-4, 1e-5],[1e-5, 1e-8]
                if key in ['EXPLORATION_LR_RANGE', 'EXPLOITATION_LR_RANGE']:
                    try:
                        # Split the entire string by '],[' to separate individual ranges
                        ranges = value.replace('],[', ']||[').split('||')
                        # Parse each range string to a list using ast.literal_eval
                        lr_values = [ast.literal_eval(r.strip()) for r in ranges]
                        # Ensure each parsed range contains exactly two float values
                        for lr_range in lr_values:
                            if len(lr_range) != 2 or not all(isinstance(v, float) for v in lr_range):
                                raise ValueError
                        validated_params[key] = lr_values  # Store as list of lists
                    except (ValueError, SyntaxError):
                        self.logger.error(f"Invalid value for {key}: {value} (expected float range arrays)")
                        raise ValueError(f"Invalid value for {key}: Expected float range arrays, got {value}")

                # Handle scalar integers for specific keys (consolidated logic)
                elif key in ['LAYERS', 'HIDDEN_UNITS', 'BATCH_SIZE', 'MAX_EPOCHS', 'VALID_PATIENCE',
                            'ValidFrequency', 'LOOKBACK', 'N_PARTICLES', 'EXPLORATION_EPOCHS',
                            'EXPLORATION_LR_UPDATE_INTERVAL', 'EXPLOITATION_LR_UPDATE_INTERVAL']:
                    try:
                        # Just validate if the value can be converted to an integer
                        int(value)
                        validated_params[key] = value  # Keep as string
                    except ValueError:
                        self.logger.error(f"Invalid value for {key}: {value} (expected integer)")
                        raise ValueError(f"Invalid value for {key}: Expected integer, got {value}")

            elif isinstance(value, list):
                # Directly validate the list inputs without alteration
                validated_params[key] = value

            else:
                validated_params[key] = value

        self.logger.info("Parameter validation complete.")
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
