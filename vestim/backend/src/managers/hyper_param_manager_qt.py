import os
import json
import logging

class VEstimHyperParamManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.current_params = {}
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
        """Validate and normalize the parameter values while ensuring type consistency."""
        validated_params = {}

        for key, value in params.items():
            if isinstance(value, str):
                # Keep the original string for parameters that might be comma-separated
                if key in ['LAYERS', 'HIDDEN_UNITS', 'BATCH_SIZE', 'MAX_EPOCHS', 'LR_DROP_PERIOD',
                        'VALID_PATIENCE', 'ValidFrequency', 'LOOKBACK', 'REPETITIONS']: # Removed old time fields
                    # Validate that all values are valid integers
                    value_list = [v.strip() for v in value.replace(',', ' ').split() if v]
                    try:
                        [int(v) for v in value_list]  # Just validate, don't convert
                        validated_params[key] = value  # Keep as string
                    except ValueError:
                        self.logger.error(f"Invalid integer value for {key}: {value}")
                        raise ValueError(f"Invalid value for {key}: Expected integers, got {value}")
                
                elif key == 'MAX_TRAINING_TIME': # New key for hh:mm:ss string
                    try:
                        # Validate hh:mm:ss format
                        h, m, s = map(int, value.split(':'))
                        if not (0 <= h <= 99 and 0 <= m <= 59 and 0 <= s <= 59): # Allow more than 23 hours
                            raise ValueError("Time components out of range.")
                        validated_params[key] = value # Store as string
                    except ValueError:
                        self.logger.error(f"Invalid time format for MAX_TRAINING_TIME: {value}. Expected hh:mm:ss")
                        raise ValueError(f"Invalid format for MAX_TRAINING_TIME: Expected hh:mm:ss, got {value}")

                elif key in ['INITIAL_LR', 'LR_DROP_FACTOR', 'DROPOUT_PROB', 'TRAIN_VAL_SPLIT']:
                    # Validate that all values are valid floats
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
                validated_params[key] = value  # Keep as-is for other data types        # ✅ Feature & Target Columns (No validation needed, comes from UI dropdowns)
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

        if not self.current_params:
            self.logger.error("No parameters available to save.")
            raise ValueError("No parameters available for saving.")

        params_file = os.path.join(job_folder, 'hyperparams.json')

        try:
            # Create a copy to modify for saving, especially for max_training_time_seconds
            params_to_save = self.current_params.copy()

            # Calculate max_training_time_seconds from MAX_TRAINING_TIME (hh:mm:ss string)
            if "MAX_TRAINING_TIME" in params_to_save:
                try:
                    time_str = params_to_save["MAX_TRAINING_TIME"]
                    h, m, s = map(int, time_str.split(':'))
                    max_training_time_seconds = (h * 3600) + (m * 60) + s
                    params_to_save["MAX_TRAINING_TIME_SECONDS"] = max_training_time_seconds
                    self.current_params["MAX_TRAINING_TIME_SECONDS"] = max_training_time_seconds # Update in-memory params
                    self.logger.info(f"Calculated and stored MAX_TRAINING_TIME_SECONDS: {max_training_time_seconds} from {time_str}")
                except ValueError:
                    self.logger.warning(f"Could not parse MAX_TRAINING_TIME '{params_to_save.get('MAX_TRAINING_TIME')}' to seconds. MAX_TRAINING_TIME_SECONDS will default to 0 if not already present.")
                    if "MAX_TRAINING_TIME_SECONDS" not in params_to_save:
                        params_to_save["MAX_TRAINING_TIME_SECONDS"] = 0
            elif "MAX_TRAINING_TIME_SECONDS" not in params_to_save: # Ensure it exists if MAX_TRAINING_TIME was not provided
                 params_to_save["MAX_TRAINING_TIME_SECONDS"] = 0


            # Validate before saving. self.current_params should now contain MAX_TRAINING_TIME (string)
            # and potentially MAX_TRAINING_TIME_SECONDS (int) after the above block.
            # The validate_and_normalize_params method has been updated to expect MAX_TRAINING_TIME as hh:mm:ss.
            _ = self.validate_and_normalize_params(self.current_params)

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
        self.current_params.update(validated_params)
        # self.param_sets.append(self.current_params)
        self.logger.info("Parameters successfully updated.")
    
    def get_current_params(self, job_folder: str):
        """Load the parameters from the saved JSON file in the job folder."""
        params_file = os.path.join(job_folder, 'hyperparams.json')
        
        if os.path.exists(params_file):
            with open(params_file, 'r') as file:
                current_params = json.load(file)
                self.current_params = current_params
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