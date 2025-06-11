import os
import json
import itertools
from datetime import datetime
from vestim.config import OUTPUT_DIR
from vestim.logger_config import configure_job_specific_logging
import logging
import glob
import shutil

class JobService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(JobService, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.job_id = None
            self.logger = logging.getLogger(__name__)
            
            # Ensure output directory exists
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            
            # Initialize the job registry file if it doesn't exist
            self.job_registry_file = os.path.join(OUTPUT_DIR, 'job_registry.json')
            if not os.path.exists(self.job_registry_file):
                with open(self.job_registry_file, 'w') as f:
                    json.dump({"jobs": []}, f, indent=4)

    def create_new_job(self, selections: dict):
        """Generates a new job ID and creates the main job directory."""
        job_id = f"job_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        job_folder = os.path.join(OUTPUT_DIR, job_id)
        os.makedirs(job_folder, exist_ok=True)
        
        try:
            configure_job_specific_logging(job_folder)
            self.logger.info(f"Job-specific logging configured for job: {job_id}")
        except Exception as e:
            self.logger.error(f"Failed to configure job-specific logging for {job_id}: {e}", exc_info=True)
            
        # Store the selections and initial status in a status.json file
        status_file = os.path.join(job_folder, 'status.json')
        job_info = {
            "job_id": job_id,
            "status": "created",
            "selections": selections,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "job_folder": job_folder,
            "state": {}
        }
        
        with open(status_file, 'w') as f:
            json.dump(job_info, f, indent=4)
            
        # Register the job in the global registry
        self._register_job(job_info)
            
        self.logger.info(f"Created new job {job_id} with selections: {selections}")
        
        return job_id, job_folder
    
    def _register_job(self, job_info):
        """Add a job to the registry file."""
        try:
            with open(self.job_registry_file, 'r') as f:
                registry = json.load(f)
            
            # Add the new job
            registry["jobs"].append(job_info)
            
            # Write back to the registry file
            with open(self.job_registry_file, 'w') as f:
                json.dump(registry, f, indent=4)
                
            self.logger.info(f"Job {job_info['job_id']} registered in the registry")
        except Exception as e:
            self.logger.error(f"Failed to register job in registry: {e}", exc_info=True)

    def update_job_status(self, job_id, status, state_payload=None):
        """Update the status and state of a job."""
        job_folder = os.path.join(OUTPUT_DIR, job_id)
        status_file = os.path.join(job_folder, 'status.json')
        
        if not os.path.exists(status_file):
            self.logger.error(f"Status file for job {job_id} not found")
            return False
        
        try:
            # Update the job status file
            with open(status_file, 'r') as f:
                job_info = json.load(f)
            
            job_info["status"] = status
            job_info["state"] = state_payload if state_payload else {}
            job_info["updated_at"] = datetime.now().isoformat()
            
            with open(status_file, 'w') as f:
                json.dump(job_info, f, indent=4)
            
            # Update the job in the registry
            self._update_job_in_registry(job_info)
            
            self.logger.info(f"Job {job_id} status updated to: {status}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to update job status: {e}", exc_info=True)
            return False
    
    def _update_job_in_registry(self, job_info):
        """Update a job in the registry file."""
        try:
            with open(self.job_registry_file, 'r') as f:
                registry = json.load(f)
            
            # Find and update the job
            for i, job in enumerate(registry["jobs"]):
                if job["job_id"] == job_info["job_id"]:
                    registry["jobs"][i] = job_info
                    break
            
            # Write back to the registry file
            with open(self.job_registry_file, 'w') as f:
                json.dump(registry, f, indent=4)
        except Exception as e:
            self.logger.error(f"Failed to update job in registry: {e}", exc_info=True)

    def generate_task_configs(self, hyperparams: dict):
        """
        Generates individual task configurations from a dictionary of hyperparameters,
        some of which may contain comma-separated values for tuning.
        """
        self.logger.info("Generating task configurations from hyperparameters.")
        
        # Separate fixed params from params that need to be iterated over
        fixed_params = {}
        tuning_params = {}
        
        for key, value in hyperparams.items():
            if isinstance(value, str) and ',' in value:
                # Split string by comma, strip whitespace, and filter out empty strings
                options = [v.strip() for v in value.split(',') if v.strip()]
                if options:
                    tuning_params[key] = options
                else:
                    # Handle case where a key has a comma but no valid values (e.g., " , ")
                    self.logger.warning(f"Hyperparameter '{key}' contained commas but no valid values. Ignoring.")
            else:
                fixed_params[key] = value

        if not tuning_params:
            # If no parameters are being tuned, create a single task config
            self.logger.info("No hyperparameter tuning detected. Creating a single task.")
            return [hyperparams]

        # Generate the Cartesian product of all tuning parameter values
        keys, values = zip(*tuning_params.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        self.logger.info(f"Generated {len(param_combinations)} unique parameter combinations for tuning.")

        # Create a full task configuration for each combination
        task_configs = []
        for combo in param_combinations:
            config = fixed_params.copy()
            config.update(combo)
            task_configs.append(config)
            
        return task_configs

    def get_job_id(self):
        return self.job_id
    
    def set_job_id(self, job_id):
        """Set the current job ID."""
        self.job_id = job_id

    def get_job_folder(self, job_id=None):
        """Get the folder path for a job, using current job ID if none provided."""
        id_to_use = job_id if job_id else self.job_id
        if id_to_use:
            return os.path.join(OUTPUT_DIR, id_to_use)
        return None
    
    def get_train_folder(self, job_id=None):
        id_to_use = job_id if job_id else self.job_id
        if id_to_use:
            folder = os.path.join(self.get_job_folder(id_to_use), 'train_data', 'processed_data')
            os.makedirs(folder, exist_ok=True)
            return folder
        return None

    def get_test_folder(self, job_id=None):
        id_to_use = job_id if job_id else self.job_id
        if id_to_use:
            folder = os.path.join(self.get_job_folder(id_to_use), 'test_data', 'processed_data')
            os.makedirs(folder, exist_ok=True)
            return folder
        return None
    
    def get_all_jobs(self):
        """Get all jobs from the registry."""
        try:
            if os.path.exists(self.job_registry_file):
                with open(self.job_registry_file, 'r') as f:
                    registry = json.load(f)
                return registry["jobs"]
            return []
        except Exception as e:
            self.logger.error(f"Failed to get all jobs: {e}", exc_info=True)
            return []
    
    def get_job_by_id(self, job_id):
        """Get a specific job by ID."""
        try:
            jobs = self.get_all_jobs()
            for job in jobs:
                if job["job_id"] == job_id:
                    return job
            return None
        except Exception as e:
            self.logger.error(f"Failed to get job by ID: {e}", exc_info=True)
            return None
    
    def clear_all_jobs(self):
        """Delete all jobs and their data."""
        try:
            # Delete job directories
            job_paths = glob.glob(os.path.join(OUTPUT_DIR, "job_*"))
            for path in job_paths:
                shutil.rmtree(path, ignore_errors=True)
            
            # Reset the job registry
            with open(self.job_registry_file, 'w') as f:
                json.dump({"jobs": []}, f, indent=4)
                
            self.logger.info("All jobs have been cleared")
            return True
        except Exception as e:
            self.logger.error(f"Failed to clear all jobs: {e}", exc_info=True)
            return False
    
    def get_test_results_folder(self):
        if self.job_id:
            results_folder = os.path.join(self.get_job_folder(), 'test', 'results')
            os.makedirs(results_folder, exist_ok=True)
            return results_folder
        return None