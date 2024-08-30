import os
from datetime import datetime
from src.config import OUTPUT_DIR

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
        return self.job_id, job_folder

    def get_job_id(self):
        """Returns the current job ID."""
        return self.job_id

    def get_job_folder(self):
        """Returns the path to the current job folder."""
        if self.job_id:
            return os.path.join(OUTPUT_DIR, self.job_id)
        return None