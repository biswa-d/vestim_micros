# ---------------------------------------------------------------------------------
# Author: Biswanath Dehury
# Date: 2025-06-10
# Version: 1.0.0
# Description: 
# Service for handling the training setup process. This includes building
# models based on hyperparameter combinations and creating the initial
# training task configurations.
# ---------------------------------------------------------------------------------

import os
import json
import logging
import torch

from vestim.backend.src.services.job_service import JobService
# The following imports will need to be resolved within the backend context
from vestim.backend.src.managers.hyper_param_manager_qt import VEstimHyperParamManager
from vestim.backend.src.services.model_training.src.LSTM_model_service import LSTMModelService

class SetupService:
    """Handles the creation of models and training tasks based on hyperparameters."""

    def __init__(self, job_service: JobService, hyper_params: dict):
        """
        Initializes the SetupService.
        
        Args:
            job_service: An instance of JobService for managing job context.
            hyper_params: A dictionary of hyperparameters for the setup.
        """
        self.logger = logging.getLogger(__name__)
        self.job_service = job_service
        self.hyper_params = hyper_params
        self.hyper_param_manager = VEstimHyperParamManager() # This will be refactored later
        self.hyper_param_manager.params = self.hyper_params # Manually set params
        self.lstm_model_service = LSTMModelService()
        self.models = []
        self.training_tasks = []

    def run_setup(self):
        """
        Executes the full training setup process.
        """
        self.logger.info("Starting training setup process...")
        try:
            self.logger.info("Building models...")
            self.build_models()

            self.logger.info("Creating training tasks...")
            self.create_training_tasks()

            self.logger.info("Setup complete!")
            # In the future, we can write a status file here
            
        except Exception as e:
            self.logger.error(f"Error during setup process: {e}", exc_info=True)
            # In the future, we can write an error status file here
            raise

    def build_models(self):
        """Build and store the models based on hyperparameters."""
        # This is a simplified version of the original build_models method.
        # We will need to refactor this further to remove dependencies on the old manager.
        # For now, we will replicate the logic.
        
        try:
            hidden_units_value = str(self.hyper_params['HIDDEN_UNITS'])
            layers_value = str(self.hyper_params['LAYERS'])
            hidden_units_list = [int(h) for h in hidden_units_value.split(',')]
            layers_list = [int(l) for l in layers_value.split(',')]

            feature_columns = self.hyper_params.get("FEATURE_COLUMNS", [])
            target_column = self.hyper_params.get("TARGET_COLUMN", "")
            model_type = self.hyper_params.get("MODEL_TYPE", "LSTM")

            if not feature_columns or not target_column:
                raise ValueError("Feature columns or target column not set in hyperparameters.")

            input_size = len(feature_columns)
            output_size = 1

            self.logger.info(f"Building {model_type} models with INPUT_SIZE={input_size}, OUTPUT_SIZE={output_size}")

            for hidden_units in hidden_units_list:
                for layers in layers_list:
                    self.logger.info(f"Creating model with hidden_units: {hidden_units}, layers: {layers}")
                    model_dir = os.path.join(
                        self.job_service.get_job_folder(),
                        'models',
                        f'model_{model_type}_hu_{hidden_units}_layers_{layers}'
                    )
                    os.makedirs(model_dir, exist_ok=True)
                    model_path = os.path.join(model_dir, "untrained_model_template.pth")

                    model_params = {
                        "INPUT_SIZE": input_size,
                        "OUTPUT_SIZE": output_size,
                        "HIDDEN_UNITS": hidden_units,
                        "LAYERS": layers
                    }
                    
                    # This part will need further refactoring to select the model service dynamically
                    # and handle device placement correctly on the server.
                    target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    model = self.lstm_model_service.create_and_save_lstm_model_with_LN(model_params, model_path, target_device)

                    self.models.append({
                        'model': model,
                        'model_type': model_type,
                        'model_dir': model_dir,
                        "FEATURE_COLUMNS": feature_columns,
                        "TARGET_COLUMN": target_column,
                        'hyperparams': {
                            'INPUT_SIZE': input_size,
                            'OUTPUT_SIZE': output_size,
                            'LAYERS': layers,
                            'HIDDEN_UNITS': hidden_units,
                            'model_path': model_path
                        }
                    })
            self.logger.info("Model building complete.")
        except Exception as e:
            self.logger.error(f"Error during model building: {e}", exc_info=True)
            raise

    def create_training_tasks(self):
        """Create training tasks based on hyperparameters."""
        # This method is highly dependent on the hyperparameter structure and will be
        # refactored to be more robust. For now, we replicate the core logic.
        # The original method is very long, so this is a simplified placeholder.
        
        tasks_summary_file = os.path.join(self.job_service.get_job_folder(), 'training_tasks_summary.json')
        
        # This is where the complex loop from the original manager would go.
        # For now, we'll just write a placeholder file.
        placeholder_tasks = [{"task_id": "placeholder_task", "status": "created"}]
        
        with open(tasks_summary_file, 'w') as f:
            json.dump(placeholder_tasks, f, indent=4)
            
        self.training_tasks = placeholder_tasks
        self.logger.info(f"Training task summary saved to {tasks_summary_file}")