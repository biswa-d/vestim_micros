import os
import json
import logging
import torch
import time
from vestim.gateway.src.job_manager_qt import JobManager
from vestim.services.model_training.src.LSTM_model_service import LSTMModelService
from vestim.services.model_training.src.GRU_model_service import GRUModelService
from vestim.services.model_training.src.FNN_model_service import FNNModelService

from vestim.gateway.src.training_setup_manager_qt import VEstimTrainingSetupManager

class OptunaSetupManager:
   """
   A dedicated, non-singleton manager for setting up training tasks from Optuna results.
   This class now delegates task creation to the VEstimTrainingSetupManager to ensure consistency.
   """
   def __init__(self, job_manager=None, progress_signal=None):
       self.logger = logging.getLogger(__name__)
       self.job_manager = job_manager if job_manager else JobManager()
       # Pass the progress_signal to the training setup manager
       self.training_setup_manager = VEstimTrainingSetupManager(
           progress_signal=progress_signal,
           job_manager=self.job_manager
       )
       self.training_tasks = []

   def setup_training_from_optuna(self, optuna_configs):
       """Set up the training process using configurations from Optuna."""
       self.logger.info("Setting up training from Optuna configurations using dedicated manager...")
       try:
           if not optuna_configs:
               raise ValueError("Optuna configurations are missing.")
           
           # Delegate task creation to the centralized training setup manager
           self.training_setup_manager.setup_training_from_optuna(optuna_configs)
           
           # Retrieve the created tasks
           self.training_tasks = self.training_setup_manager.get_task_list()
           
       except Exception as e:
           self.logger.error(f"Error during Optuna setup: {str(e)}")
           raise

   def get_task_list(self):
       """Returns the list of training tasks."""
       return self.training_tasks