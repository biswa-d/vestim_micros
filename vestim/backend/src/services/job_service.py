import os
import itertools
from datetime import datetime
from vestim.config import OUTPUT_DIR
from vestim.logger_config import configure_job_specific_logging
import logging

class JobService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(JobService, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'job_id'):
            self.job_id = None
            self.logger = logging.getLogger(__name__)

    def create_new_job(self):
        """Generates a new job ID and creates the main job directory."""
        self.job_id = f"job_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        job_folder = os.path.join(OUTPUT_DIR, self.job_id)
        os.makedirs(job_folder, exist_ok=True)
        
        try:
            configure_job_specific_logging(job_folder)
            self.logger.info(f"Job-specific logging configured for job: {self.job_id}")
        except Exception as e:
            self.logger.error(f"Failed to configure job-specific logging for {self.job_id}: {e}", exc_info=True)
            
        return self.job_id, job_folder

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

    def get_job_folder(self):
        if self.job_id:
            return os.path.join(OUTPUT_DIR, self.job_id)
        return None
    
    def get_train_folder(self):
        if self.job_id:
            return os.path.join(self.get_job_folder(), 'train_data', 'processed_data')
        return None

    def get_test_folder(self):
        if self.job_id:
            return os.path.join(self.get_job_folder(), 'test_data', 'processed_data')
        return None
    
    def get_test_results_folder(self):
        if self.job_id:
            results_folder = os.path.join(self.get_job_folder(), 'test', 'results')
            os.makedirs(results_folder, exist_ok=True)
            return results_folder
        return None