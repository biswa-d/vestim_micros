import os
import json
from src.gateway.src.job_manager import JobManager

class VEstimHyperParamManager:
    def __init__(self):
        self.params = {}
        self.job_manager = JobManager()  # Initialize JobManager
        self.param_entries = {}

    def load_params(self, filepath):
        """Loads parameters from a JSON file."""
        if os.path.exists(filepath):
            with open(filepath, 'r') as file:
                self.params = json.load(file)
            return self.params
        else:
            print(f"File {filepath} not found.")
            return {}

    def save_params(self):
        """Saves current parameters to the job-specific folder."""
        job_folder = self.job_manager.get_job_folder()  # Get the job folder from the manager
        if job_folder:
            return self.save_hyperparams(self.params, job_folder)
        else:
            raise ValueError("Job folder is not set.")

    def save_hyperparams(self, new_params, job_folder):
        """Saves the given parameters to a JSON file in the job folder."""
        self.params.update(new_params)
        param_file = os.path.join(job_folder, 'hyperparams.json')
        os.makedirs(os.path.dirname(param_file), exist_ok=True)
        with open(param_file, 'w') as file:
            json.dump(self.params, file, indent=4)
        return param_file

    def save_params_to_file(self, new_params, filepath):
        """Saves the given parameters to a specified file path."""
        self.params.update(new_params)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as file:
            json.dump(self.params, file, indent=4)
        return filepath

    def update_params(self, new_params):
        """Updates the current set of parameters."""
        self.params.update(new_params)

    def get_current_params(self):
        """Returns the current parameters."""
        return self.params
