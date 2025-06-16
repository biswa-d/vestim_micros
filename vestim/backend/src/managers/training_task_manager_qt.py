import time, os, sys, math, json
import torch
import logging
import traceback
from vestim.backend.src.services.training_service import TrainingService
from vestim.backend.src.managers.training_setup_manager_qt import VEstimTrainingSetupManager

class TrainingTaskManager:
    """
    Manages the execution of a training task in a background process.
    This class is UI-agnostic and designed to be called by the JobManager.
    """
    def __init__(self):
        """
        Initializes the TrainingTaskManager.
        Note: Heavy initialization should be done inside the process_task_in_background
        method to ensure it runs in the separate process.
        """
        self.logger = logging.getLogger(__name__)

    def process_task_in_background(self, status_queue, task_info):
        """
        This method is the entry point for the background process started by the JobManager.
        It handles the entire lifecycle of a training task.
        """
        job_id = task_info.get('job_id')
        stop_flag = task_info.get('stop_flag')  # Event to check for stop signals
        
        try:
            self.logger.info(f"[{job_id}] Background training process started.")
            status_queue.put((job_id, 'initializing', {"message": "Setting up training environment..."}))

            # --- Initialization within the process ---
            training_setup_manager = VEstimTrainingSetupManager()
            
            # Determine device
            selected_device_str = task_info.get('DEVICE_SELECTION', 'cuda:0')
            if selected_device_str.startswith("cuda") and torch.cuda.is_available():
                device = torch.device(selected_device_str)
            else:
                device = torch.device("cpu")
            self.logger.info(f"[{job_id}] Using device: {device}")

            # --- Data and Model Setup ---
            status_queue.put((job_id, 'setup', {"message": "Loading data and model..."}))
            
            # Check for stop signal before expensive operations
            if stop_flag and stop_flag.is_set():
                self.logger.info(f"[{job_id}] Stop signal received during setup.")
                status_queue.put((job_id, 'stopped', {"message": "Job stopped during setup."}))
                return
            
            # Initialize the training service with the stop flag
            training_service = TrainingService(task_info=task_info, global_params=task_info, status_queue=status_queue)
            
            self.logger.info(f"[{job_id}] Setup complete. Starting training.")
            
            # Check for stop signal again before starting training
            if stop_flag and stop_flag.is_set():
                self.logger.info(f"[{job_id}] Stop signal received before training start.")
                status_queue.put((job_id, 'stopped', {"message": "Job stopped before training started."}))
                return
            
            # The run_task method contains the full training loop
            # We'll simulate it here for demonstration, but the real version would use training_service.run_task()
            
            # --- Training Loop ---
            num_epochs = task_info.get('epochs', 10)
            history = []
            
            for epoch in range(1, num_epochs + 1):
                # Check for stop signal before each epoch
                if stop_flag and stop_flag.is_set():
                    self.logger.info(f"[{job_id}] Stop signal received during training (epoch {epoch}).")
                    status_queue.put((job_id, 'stopped', {
                        "message": f"Training stopped at epoch {epoch}/{num_epochs}",
                        "history": history
                    }))
                    return
                
                status_queue.put((job_id, 'training', {
                    "message": f"Epoch {epoch}/{num_epochs}",
                    "progress": (epoch / num_epochs) * 100,
                    "history": history
                }))

                # Simulate training for one epoch
                # In a real implementation, this would be actual training code
                time.sleep(2) # Simulating work
                train_loss = math.exp(-epoch / 5)
                val_loss = math.exp(-epoch / 4)

                epoch_data = {
                    "epoch": epoch,
                    "total_epochs": num_epochs,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "status": "training",
                    "message": f"Epoch {epoch} complete."
                }
                history.append(epoch_data)

                # Send detailed update after each epoch
                status_queue.put((job_id, 'training', {
                    "message": f"Epoch {epoch} complete.",
                    "progress": (epoch / num_epochs) * 100,
                    "history": history
                }))

            # --- Completion ---
            self.logger.info(f"[{job_id}] Training completed successfully.")
            status_queue.put((job_id, 'complete', {
                "message": "Training finished.",
                "history": history
            }))

        except Exception as e:
            error_msg = f"[{job_id}] Error during training task: {e}"
            self.logger.error(error_msg, exc_info=True)
            # Include the traceback in the error details for better debugging
            status_queue.put((job_id, 'error', {
                "message": str(e),
                "error": error_msg,
                "traceback": traceback.format_exc()
            }))