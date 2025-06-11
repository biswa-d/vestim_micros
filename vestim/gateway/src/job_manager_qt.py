import requests
import logging
import os

class JobManager:
    _instance = None
    BACKEND_URL = "http://127.0.0.1:8001"

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(JobManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'job_id'):
            self.job_id = None
            self.job_folder = None
            self.logger = logging.getLogger(__name__)

    def create_new_job(self):
        """Creates a new job by sending a request to the backend."""
        try:
            response = requests.post(f"{self.BACKEND_URL}/jobs", timeout=5)
            response.raise_for_status()
            job_data = response.json()
            self.job_id = job_data.get("job_id")
            self.job_folder = job_data.get("job_folder")
            self.logger.info(f"Successfully created job {self.job_id}")
            return self.job_id, self.job_folder
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to create new job: {e}", exc_info=True)
            return None, None

    def get_job_id(self):
        """Returns the current job ID."""
        return self.job_id

    def start_training(self, hyperparams: dict):
        """Starts the training tasks for the current job."""
        if not self.job_id:
            self.logger.error("Job ID not set. Cannot start training.")
            return None, None
        try:
            response = requests.post(f"{self.BACKEND_URL}/jobs/{self.job_id}/start_training", json={"params": hyperparams})
            response.raise_for_status()
            job_data = response.json()
            self.logger.info(f"Successfully started training for job {self.job_id}")
            return job_data.get("status")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to start training for job {self.job_id}: {e}", exc_info=True)
            return None

    def get_job_folder(self):
        """Returns the path to the current job folder."""
        return self.job_folder

    def get_train_folder(self):
        """Returns the path to the train processed data folder."""
        if self.job_folder:
            return os.path.join(self.job_folder, 'train_data', 'processed_data')
        return None

    def get_test_folder(self):
        """Returns the path to the test processed data folder."""
        if self.job_folder:
            return os.path.join(self.job_folder, 'test_data', 'processed_data')
        return None

    def get_test_results_folder(self):
        """
        Returns the path to the test results folder.
        """
        if self.job_folder:
            results_folder = os.path.join(self.job_folder, 'test', 'results')
            os.makedirs(results_folder, exist_ok=True)
            return results_folder
        return None
