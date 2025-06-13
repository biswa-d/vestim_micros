import itertools
import logging

class TrainingSetupManager:
    """Manages the setup of training tasks for a specific job."""
    def __init__(self, params=None):
        self.logger = logging.getLogger(__name__)
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
            return [hyperparams]

        keys, values = zip(*tuning_params.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        task_configs = []
        for combo in param_combinations:
            config = fixed_params.copy()
            config.update(combo)
            task_configs.append(config)
            
        return task_configs