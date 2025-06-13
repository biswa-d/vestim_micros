import json
import os
import logging

class HyperParamManager:
    """Manages hyperparameters for a specific job."""
    def __init__(self, params=None):
        self.logger = logging.getLogger(__name__)
        self.params = params or {}

    def update_params(self, new_params: dict):
        """Update the hyperparameters."""
        self.params.update(new_params)
        self.logger.info(f"Hyperparameters updated: {self.params}")

    def get_hyper_params(self):
        """Return the current hyperparameters."""
        return self.params

    def save_params(self, job_folder: str, filename: str = "hyperparameters.json"):
        """Save hyperparameters to a file."""
        path = os.path.join(job_folder, filename)
        with open(path, 'w') as f:
            json.dump(self.params, f, indent=4)
        self.logger.info(f"Hyperparameters saved to {path}")

    def load_params(self, job_folder: str, filename: str = "hyperparameters.json"):
        """Load hyperparameters from a file."""
        path = os.path.join(job_folder, filename)
        with open(path, 'r') as f:
            self.params = json.load(f)
        self.logger.info(f"Hyperparameters loaded from {path}")
        return self.params