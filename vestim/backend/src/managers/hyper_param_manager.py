import json
import os
import logging

class VEstimHyperParamManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.params = {}
        self.logger.info("VEstimHyperParamManager initialized.")

    def load_params(self, filepath):
        """Load and validate parameters from a JSON file."""
        self.logger.info(f"Loading parameters from {filepath}")
        with open(filepath, 'r') as file:
            params = json.load(file)
            validated_params = self.validate_and_normalize_params(params)
            self.params = validated_params
            self.logger.info("Parameters successfully loaded and validated.")
        return validated_params

    def validate_and_normalize_params(self, params):
        """Validate and normalize the parameter values while ensuring type consistency."""
        validated_params = {}

        for key, value in params.items():
            if isinstance(value, str):
                if key in ['LAYERS', 'HIDDEN_UNITS', 'BATCH_SIZE', 'MAX_EPOCHS', 'LR_DROP_PERIOD',
                        'VALID_PATIENCE', 'ValidFrequency', 'LOOKBACK', 'REPETITIONS']:
                    value_list = [v.strip() for v in value.replace(',', ' ').split() if v]
                    try:
                        [int(v) for v in value_list]
                        validated_params[key] = value
                    except ValueError:
                        self.logger.error(f"Invalid integer value for {key}: {value}")
                        raise ValueError(f"Invalid value for {key}: Expected integers, got {value}")
                
                elif key == 'MAX_TRAINING_TIME':
                    try:
                        h, m, s = map(int, value.split(':'))
                        if not (0 <= h <= 99 and 0 <= m <= 59 and 0 <= s <= 59):
                            raise ValueError("Time components out of range.")
                        validated_params[key] = value
                    except ValueError:
                        self.logger.error(f"Invalid time format for MAX_TRAINING_TIME: {value}. Expected hh:mm:ss")
                        raise ValueError(f"Invalid format for MAX_TRAINING_TIME: Expected hh:mm:ss, got {value}")

                elif key in ['INITIAL_LR', 'LR_DROP_FACTOR', 'DROPOUT_PROB', 'TRAIN_VAL_SPLIT']:
                    value_list = [v.strip() for v in value.replace(',', ' ').split() if v]
                    try:
                        [float(v) for v in value_list]
                        validated_params[key] = value
                    except ValueError:
                        self.logger.error(f"Invalid float value for {key}: {value}")
                        raise ValueError(f"Invalid value for {key}: Expected floats, got {value}")

                elif key in ['BATCH_TRAINING']:
                    validated_params[key] = value.lower() in ['true', '1', 'yes']

                else:
                    validated_params[key] = value

            elif isinstance(value, list):
                validated_params[key] = value

            else:
                validated_params[key] = value
        validated_params["FEATURE_COLUMNS"] = params.get("FEATURE_COLUMNS", [])
        validated_params["TARGET_COLUMN"] = params.get("TARGET_COLUMN", "")
        validated_params["MODEL_TYPE"] = params.get("MODEL_TYPE", "")

        self.logger.info("Parameter validation and normalization completed successfully.")
        return validated_params

    def save_params(self, job_folder: str):
        """Save the current validated parameters to the job folder in a JSON file."""
        if not job_folder:
            self.logger.error("Job folder is not provided. Cannot save parameters.")
            raise ValueError("Job folder must be provided to save parameters.")

        if not self.params:
            self.logger.error("No parameters available to save.")
            raise ValueError("No parameters available for saving.")

        params_file = os.path.join(job_folder, 'hyperparams.json')

        try:
            params_to_save = self.params.copy()

            if "MAX_TRAINING_TIME" in params_to_save:
                try:
                    time_str = params_to_save["MAX_TRAINING_TIME"]
                    h, m, s = map(int, time_str.split(':'))
                    max_training_time_seconds = (h * 3600) + (m * 60) + s
                    params_to_save["MAX_TRAINING_TIME_SECONDS"] = max_training_time_seconds
                    self.params["MAX_TRAINING_TIME_SECONDS"] = max_training_time_seconds
                    self.logger.info(f"Calculated and stored MAX_TRAINING_TIME_SECONDS: {max_training_time_seconds} from {time_str}")
                except ValueError:
                    self.logger.warning(f"Could not parse MAX_TRAINING_TIME '{params_to_save.get('MAX_TRAINING_TIME')}' to seconds. MAX_TRAINING_TIME_SECONDS will default to 0 if not already present.")
                    if "MAX_TRAINING_TIME_SECONDS" not in params_to_save:
                        params_to_save["MAX_TRAINING_TIME_SECONDS"] = 0
            elif "MAX_TRAINING_TIME_SECONDS" not in params_to_save:
                 params_to_save["MAX_TRAINING_TIME_SECONDS"] = 0

            _ = self.validate_and_normalize_params(self.params)

            with open(params_file, 'w') as file:
                json.dump(params_to_save, file, indent=4)
            self.logger.info(f"Hyperparameters successfully saved to file: {params_file}")
            self.logger.info(f"Saved content: {json.dumps(params_to_save, indent=4)}")

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
        self.params.update(validated_params)
        self.logger.info("Parameters successfully updated.")
    
    def get_hyper_params(self):
        """Return the current hyperparameters stored in memory."""
        if self.params:
            self.logger.info("Returning current hyperparameters.")
            return self.params
        else:
            self.logger.error("No current parameters available in memory.")
            raise ValueError("No current parameters are available in memory.")