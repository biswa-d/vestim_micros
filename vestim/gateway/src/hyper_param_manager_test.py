import os
import json
from vestim.gateway.src.job_manager import JobManager

class VEstimHyperParamManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(VEstimHyperParamManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.job_manager = JobManager()
            self.current_params = {}  # Initialize current_params as None
            # self.param_sets = []  # Initialize param_sets as an empty list
            self.initialized = True

    def load_params(self, filepath):
        """Load and validate parameters from a JSON file."""
        with open(filepath, 'r') as file:
            params = json.load(file)
            validated_params = self.validate_and_normalize_params(params)
            # self.param_sets.append(validated_params)
            self.current_params = validated_params  # Set the current_params to the loaded params
        return validated_params

    def validate_and_normalize_params(self, params):
        """Validate the parameter values without altering them."""
        validated_params = {}

        for key, value in params.items():
            if isinstance(value, str):
                # Split the string into a list, allowing for comma or space separation
                value_list = [v.strip() for v in value.replace(',', ' ').split() if v]
                # Check data types without converting
                if key in ['LAYERS', 'HIDDEN_UNITS', 'BATCH_SIZE', 'MAX_EPOCHS', 'LR_DROP_PERIOD', 'VALID_PATIENCE', 'ValidFrequency', 'LOOKBACK', 'REPETITIONS']:
                    # Ensure values are integers
                    if not all(v.isdigit() for v in value_list):
                        raise ValueError(f"Invalid value for {key}: Expected integers, got {value_list}")
                    validated_params[key] = value  # Keep the original string

                elif key in ['INITIAL_LR', 'LR_DROP_FACTOR']:
                    # Ensure values are floats
                    try:
                        [float(v) for v in value_list]
                    except ValueError:
                        raise ValueError(f"Invalid value for {key}: Expected floats, got {value_list}")
                    validated_params[key] = value  # Keep the original string

                else:
                    validated_params[key] = value

            elif isinstance(value, list):
                # Check list elements without altering them
                validated_params[key] = value
            else:
                validated_params[key] = value

        return validated_params


    def save_params(self):
        """Save the current parameters to the job folder."""
        job_folder = self.job_manager.get_job_folder()
        if job_folder and self.current_params:
            params_file = os.path.join(job_folder, 'hyperparams.json')
            with open(params_file, 'w') as file:
                json.dump(self.current_params, file, indent=4)
        else:
            raise ValueError("Job folder is not set or current parameters are not available.")

    def save_params_to_file(self, new_params, filepath):
        """Save new parameters to a specified file."""
        with open(filepath, 'w') as file:
            json.dump(new_params, file, indent=4)

    def update_params(self, new_params):
        """Update the current parameters with new values."""
        validated_params = self.validate_and_normalize_params(new_params)
        self.current_params.update(validated_params)
        # self.param_sets.append(self.current_params)

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
            raise FileNotFoundError("Hyperparameters JSON file not found in the job folder.")

    def get_hyper_params(self):
        """Return the current hyperparameters stored in memory."""
        if self.current_params:
            return self.current_params
        else:
            raise ValueError("No current parameters are available in memory.")
