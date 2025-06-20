import itertools
import logging

class VEstimTrainingSetupManager:
    """Manages the setup of training tasks for a specific job."""
    def __init__(self, job_id, job_folder, params=None):
        self.logger = logging.getLogger(__name__)
        self.job_id = job_id
        self.job_folder = job_folder
        self.params = params or {}
        self.task_list = []

    def setup_training(self):
        """Generate training tasks based on hyperparameters."""
        self.task_list = self._generate_task_configs(self.params)
        self.logger.info(f"Generated {len(self.task_list)} training tasks.")

    def get_task_list(self):
        """Return the list of generated training tasks."""
        return self.task_list

    def _generate_task_configs(self, hyperparams: dict):
        """
        Generates individual task configurations from a dictionary of hyperparameters.
        """
        fixed_params = {}
        tuning_params = {}

        for key, value in hyperparams.items():
            if isinstance(value, str) and ',' in value:
                options = [v.strip() for v in value.split(',') if v.strip()]
                if options:
                    tuning_params[key] = options
            else:
                fixed_params[key] = value

        if not tuning_params:
            self.logger.info("No hyperparameter tuning detected. Creating a single task.")
            config = fixed_params.copy()
            config['task_id'] = 'task_0'
            return [config]

        keys, values = zip(*tuning_params.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        self.logger.info(f"Generated {len(param_combinations)} unique parameter combinations for tuning.")

        task_configs = []
        for i, combo in enumerate(param_combinations):
            config = fixed_params.copy()
            config.update(combo)
            config['task_id'] = f'task_{i}'  # Add unique task ID
            task_configs.append(config)
            
        return task_configs