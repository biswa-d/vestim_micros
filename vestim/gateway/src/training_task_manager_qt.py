import time, os, sys, math, json, requests
import csv
import sqlite3
import torch
from PyQt5.QtCore import QThread, pyqtSignal
from vestim.gateway.src.job_manager_qt import JobManager
from vestim.gateway.src.training_setup_manager_qt import VEstimTrainingSetupManager
import logging, wandb

def format_time(seconds):
    """Convert seconds to mm:ss format."""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

class TrainingTaskManager:
    BACKEND_URL = "http://127.0.0.1:8000"

    def __init__(self, global_params=None):
        self.logger = logging.getLogger(__name__)
        self.job_manager = JobManager()
        self.training_setup_manager = VEstimTrainingSetupManager()
        self.current_task = None
        self.stop_requested = False
        self.global_params = global_params if global_params else {}
        
        # Determine device based on global_params or fallback
        selected_device_str = self.global_params.get('DEVICE_SELECTION', 'cuda:0')
        try:
            if selected_device_str.startswith("cuda") and not torch.cuda.is_available():
                self.logger.warning(f"CUDA device {selected_device_str} selected, but CUDA is not available. Falling back to CPU.")
                self.device = torch.device("cpu")
            elif selected_device_str.startswith("cuda"):
                self.device = torch.device(selected_device_str)
            elif selected_device_str == "CPU":
                self.device = torch.device("cpu")
            else: # Default fallback if string is unrecognized
                self.logger.warning(f"Unrecognized device selection '{selected_device_str}'. Falling back to cuda:0 if available, else CPU.")
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        except Exception as e:
            self.logger.error(f"Error setting device to '{selected_device_str}': {e}. Falling back to CPU.")
            self.device = torch.device("cpu")
        
        self.logger.info(f"TrainingTaskManager initialized with device: {self.device}")

        self.training_thread = None  # Initialize the training thread here for PyQt
       
        # WandB setup (optional)
        self.use_wandb = False  # Set to False to disable WandB
        self.wandb_enabled = False
        if self.use_wandb:
            try:
                import wandb
                wandb.init(project="VEstim", config={"task_name": "LSTM Model Training"})
                self.wandb_enabled = True
                self.logger.info("WandB initialized successfully.")
            except Exception as e:
                self.wandb_enabled = False
                self.logger.error(f"Failed to initialize WandB: {e}")

    def process_task(self, task, update_progress_callback):
        """Process a single training task by making an API call to the backend."""
        try:
            self.logger.info(f"Starting training task: {task.get('task_id', 'N/A')}")
            payload = {
                "task_info": task,
                "global_params": self.global_params
            }
            response = requests.post(f"{self.BACKEND_URL}/tasks/start", json=payload)
            response.raise_for_status()
            
            # Start monitoring the task
            self.monitor_task(task['task_id'], update_progress_callback)

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error starting task {task.get('task_id', 'N/A')}: {e}", exc_info=True)
            update_progress_callback.emit({'task_error': str(e)})
        except Exception as e:
            self.logger.error(f"Error during task processing: {str(e)}")
            update_progress_callback.emit({'task_error': str(e)})

    def monitor_task(self, task_id, update_progress_callback):
        """Periodically polls the backend for the status of a task."""
        while not self.stop_requested:
            try:
                response = requests.get(f"{self.BACKEND_URL}/tasks/{task_id}/status")
                response.raise_for_status()
                status_data = response.json()
                update_progress_callback.emit(status_data)
                
                if status_data.get("status") in ["complete", "error", "stopped"]:
                    break
                
                time.sleep(5)  # Poll every 5 seconds
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Error getting status for task {task_id}: {e}", exc_info=True)
                update_progress_callback.emit({'task_error': str(e)})
                break
            except Exception as e:
                self.logger.error(f"Error during task monitoring: {str(e)}")
                update_progress_callback.emit({'task_error': str(e)})
                break

    def stop_task(self, task_id):
        """Sends a request to the backend to stop a running task."""
        self.stop_requested = True
        try:
            response = requests.post(f"{self.BACKEND_URL}/tasks/{task_id}/stop")
            response.raise_for_status()
            self.logger.info(f"Stop signal sent for task {task_id}")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to send stop signal for task {task_id}: {e}", exc_info=True)
