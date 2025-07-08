import os
from datetime import datetime
from vestim.config import OUTPUT_DIR
from vestim.logger_config import configure_job_specific_logging # Import the new function
import logging # Import logging to potentially log the action

class JobManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(JobManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'job_id'):  # Ensure the attributes are initialized once
            self.job_id = None

    def create_new_job(self):
        """Generates a new job ID based on the current timestamp and initializes job directories."""
        self.job_id = f"job_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        job_folder = os.path.join(OUTPUT_DIR, self.job_id)
        os.makedirs(job_folder, exist_ok=True)
        
        # Configure logging to use a file within this new job folder
        try:
            configure_job_specific_logging(job_folder)
            logging.info(f"Job-specific logging configured for job: {self.job_id} in folder: {job_folder}")
        except Exception as e:
            logging.error(f"Failed to configure job-specific logging for {self.job_id}: {e}", exc_info=True)
            # Continue without job-specific logging if setup fails, default logging will be used.

        return self.job_id, job_folder

    def get_job_id(self):
        """Returns the current job ID."""
        return self.job_id

    def get_job_folder(self):
        """Returns the path to the current job folder."""
        if self.job_id:
            return os.path.join(OUTPUT_DIR, self.job_id)
        return None
    
    def get_train_folder(self):
        """Returns the path to the train processed data folder."""
        if self.job_id:
            return os.path.join(self.get_job_folder(), 'train_data', 'processed_data')
        return None

    def get_val_folder(self):
        """Returns the path to the validation processed data folder."""
        if self.job_id:
            return os.path.join(self.get_job_folder(), 'val_data', 'processed_data')
        return None

    def get_test_folder(self):
        """Returns the path to the test processed data folder."""
        if self.job_id:
            return os.path.join(self.get_job_folder(), 'test_data', 'processed_data')
        return None
    
    #Folder where test data will be stored
    def get_test_results_folder(self):
        """
        Returns the path to the test results folder.
        :return: Path to the test results folder within the job directory.
        """
        if self.job_id:
            results_folder = os.path.join(self.get_job_folder(), 'test', 'results')
            os.makedirs(results_folder, exist_ok=True)
            return results_folder
        return None
