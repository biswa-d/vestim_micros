import os
import json
import logging
import torch
import time
from vestim.gateway.src.job_manager_qt import JobManager
from vestim.services.model_training.src.LSTM_model_service import LSTMModelService
from vestim.services.model_training.src.GRU_model_service import GRUModelService
from vestim.services.model_training.src.FNN_model_service import FNNModelService

class OptunaSetupManager:
    """
    A dedicated, non-singleton manager for setting up training tasks from Optuna results.
    Each instance of this class is clean and isolated.
    """
    def __init__(self, job_manager=None):
        self.logger = logging.getLogger(__name__)
        self.job_manager = job_manager if job_manager else JobManager()
        self.lstm_model_service = LSTMModelService()
        self.gru_model_service = GRUModelService()
        self.fnn_model_service = FNNModelService()
        self.training_tasks = []

    def setup_training_from_optuna(self, optuna_configs):
        """Set up the training process using configurations from Optuna."""
        self.logger.info("Setting up training from Optuna configurations using dedicated manager...")
        try:
            if not optuna_configs:
                raise ValueError("Optuna configurations are missing.")
            
            self.create_tasks_from_optuna(optuna_configs)
            
        except Exception as e:
            self.logger.error(f"Error during Optuna setup: {str(e)}")
            raise

    def create_tasks_from_optuna(self, best_configs):
        """Create training tasks from a list of Optuna best configurations."""
        task_list = []
        for i, config_data in enumerate(best_configs):
            hyperparams = config_data['params']
            trial_number = config_data.get('trial_number', 'N/A')
            rank = i + 1

            model_task = self._build_single_model(hyperparams, trial_number, rank)
            
            task_info = self._create_task_info(
                model_task=model_task,
                hyperparams=hyperparams,
                repetition=1,
                job_normalization_metadata=self.load_job_normalization_metadata(),
                max_training_time_seconds_arg=hyperparams.get('MAX_TRAINING_TIME_SECONDS', 0)
            )
            task_list.append(task_info)

        self.training_tasks = task_list
        self.save_tasks_to_files(task_list)
        return task_list

    def _build_single_model(self, hyperparams, trial_number, rank):
        """Build a single model based on a given hyperparameter set."""
        model_type = hyperparams.get("MODEL_TYPE", "LSTM")
        input_size = len(hyperparams.get("FEATURE_COLUMNS", []))
        output_size = 1

        model_dir = os.path.join(
            self.job_manager.get_job_folder(),
            'models',
            f'trial_{trial_number}_best_{rank}'
        )
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "untrained_model_template.pth")

        model_params = {
            "INPUT_SIZE": input_size,
            "OUTPUT_SIZE": output_size,
        }
        if model_type in ["LSTM", "GRU"]:
            model_params["HIDDEN_UNITS"] = int(hyperparams.get("HIDDEN_UNITS", 10))
            model_params["LAYERS"] = int(hyperparams.get("LAYERS", 1))
        elif model_type == "FNN":
            model_params["HIDDEN_LAYER_SIZES"] = [int(s) for s in hyperparams.get("FNN_HIDDEN_LAYERS", "128,64").split(',')]
            model_params["DROPOUT_PROB"] = float(hyperparams.get("FNN_DROPOUT_PROB", 0.1))

        model = self.create_selected_model(model_type, model_params, model_path)
        
        return {
            'model': model,
            'model_type': model_type,
            'model_dir': model_dir,
            "FEATURE_COLUMNS": hyperparams.get("FEATURE_COLUMNS", []),
            "TARGET_COLUMN": hyperparams.get("TARGET_COLUMN", ""),
            'hyperparams': model_params
        }

    def create_selected_model(self, model_type, model_params, model_path):
        """Creates and saves the selected model based on the dropdown selection."""
        model_map = {
            "LSTM": self.lstm_model_service.create_and_save_lstm_model,
            "GRU": self.gru_model_service.create_and_save_gru_model,
            "FNN": self.fnn_model_service.create_and_save_fnn_model,
        }
        if model_type in model_map:
            return model_map[model_type](model_params, model_path)
        raise ValueError(f"Unsupported model type: {model_type}")

    def load_job_normalization_metadata(self):
        """Loads normalization metadata from the job folder."""
        job_folder = self.job_manager.get_job_folder()
        metadata_file_path = os.path.join(job_folder, "job_metadata.json")
        if os.path.exists(metadata_file_path):
            try:
                with open(metadata_file_path, 'r') as f_meta:
                    return json.load(f_meta)
            except Exception as e:
                self.logger.error(f"Error loading job_metadata.json: {e}")
        return {}

    def save_tasks_to_files(self, task_list):
        """Saves task information to JSON files."""
        for task_info in task_list:
            task_dir = task_info['model_dir']
            task_info_file = os.path.join(task_dir, 'task_info.json')
            serializable_info = {k: v for k, v in task_info.items() if k != 'model'}
            with open(task_info_file, 'w') as f:
                json.dump(serializable_info, f, indent=4)

    def _create_task_info(self, model_task, hyperparams, repetition, job_normalization_metadata, max_training_time_seconds_arg):
        """Helper method to create a task info dictionary."""
        task_id = f"task_{time.strftime('%Y%m%d%H%M%S')}_{model_task['model_type']}_rep_{repetition}"
        task_dir = model_task['model_dir']
        logs_dir = os.path.join(task_dir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)

        return {
            'task_id': task_id,
            'model': model_task['model'],
            'model_dir': task_dir,
            'model_path': os.path.join(task_dir, 'model.pth'),
            'logs_dir': logs_dir,
            'hyperparams': hyperparams,
            'data_loader_params': {
                'lookback': hyperparams['LOOKBACK'],
                'batch_size': hyperparams['BATCH_SIZE'],
                'feature_columns': model_task['FEATURE_COLUMNS'],
                'target_column': model_task['TARGET_COLUMN'],
                'num_workers': 4
            },
            'training_params': {
                'early_stopping_patience': hyperparams['VALID_PATIENCE'],
                'best_model_path': os.path.join(task_dir, 'best_model.pth'),
                'max_training_time_seconds': max_training_time_seconds_arg
            },
            'results': {},
            'csv_log_file': os.path.join(logs_dir, 'training_progress.csv'),
            'db_log_file': os.path.join(logs_dir, f'{task_id}_training.db'),
            'job_metadata': job_normalization_metadata,
        }

    def get_task_list(self):
        """Returns the list of training tasks."""
        return self.training_tasks