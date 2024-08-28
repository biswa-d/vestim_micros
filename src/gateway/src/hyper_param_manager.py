# src/gateway/hyper_param_manager.py

from src.services.hyper_param_selection.src.hyper_param_service import VEstimHyperParamService
from src.gateway.src.job_manager import JobManager


class VEstimHyperParamManager:
    def __init__(self):
        self.service = VEstimHyperParamService()
        self.job_manager = JobManager()  # Initialize JobManager
        self.param_sets = []

    def load_params(self, filepath):
        params = self.service.load_params_from_json(filepath)
        if params:
            self.param_sets.append(params)
        return params

    def save_params(self):
        job_folder = self.job_manager.get_job_folder()  # Get the job folder from the manager
        if job_folder:
            return self.service.save_hyperparams(self.service.get_current_params(), job_folder)
        else:
            raise ValueError("Job folder is not set.")

    def save_params_to_file(self, new_params, filepath):
        return self.service.save_params_to_file(new_params, filepath)

    def update_params(self, new_params):
        self.service.update_params(new_params)
        self.param_sets.append(self.service.get_current_params())

    def get_current_params(self):
        return self.service.get_current_params()

