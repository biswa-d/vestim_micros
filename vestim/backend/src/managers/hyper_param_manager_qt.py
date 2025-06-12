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
                        'VALID_PATIENCE', 'ValidFrequency', 'LOOKBACK', 'REPETITIONS',
                        'MAX_TRAIN_HOURS', 'MAX_TRAIN_MINUTES', 'MAX_TRAIN_SECONDS']: # Added time fields
                    # Validate that all values are valid integers
                    value_list = [v.strip() for v in value.replace(',', ' ').split() if v]
                    try:
                        [int(v) for v in value_list]  # Just validate, don't convert
                        validated_params[key] = value  # Keep as string
                    except ValueError:
                        self.logger.error(f"Invalid integer value for {key}: {value}")
                        raise ValueError(f"Invalid value for {key}: Expected integers, got {value}")

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
                json.dump(params_to_save, file, indent=4) # Save the modified dict            self.logger.info(f"Hyperparameters successfully saved to file: {params_file}")
            self.logger.info(f"Saved content: {params_to_save}")

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