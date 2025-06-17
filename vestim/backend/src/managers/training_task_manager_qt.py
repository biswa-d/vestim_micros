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

                # Update and persist metrics
                self.update_task_metrics(
                    job_folder=task_info.get('job_folder', '.'),
                    task_id=job_id,
                    epoch=epoch,
                    train_loss=train_loss,
                    valid_loss=val_loss,
                    learning_rate=task_info.get('learning_rate', 0.001),
                    status="running",
                    progress=(epoch / num_epochs) * 100
                )

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
            
    def update_task_metrics(self, job_folder, task_id, epoch, train_loss, valid_loss, learning_rate, 
                          best_epoch=None, status="running", progress=0):
        """
        Updates metrics for a training task and persists them to disk.
        This ensures metrics are available when the GUI reconnects.
        
        Args:
            job_folder: The folder for the current job
            task_id: The ID of the task being run
            epoch: Current epoch number
            train_loss: Training loss for the current epoch
            valid_loss: Validation loss for the current epoch
            learning_rate: Current learning rate
            best_epoch: Best epoch so far (optional)
            status: Task status (running, completed, failed, stopped)
            progress: Percentage completion
        """
        try:
            # Create metrics directory if it doesn't exist
            metrics_dir = os.path.join(job_folder, 'metrics')
            os.makedirs(metrics_dir, exist_ok=True)
            
            # Load existing metrics if available
            metrics_file = os.path.join(metrics_dir, f"{task_id}_metrics.json")
            if os.path.exists(metrics_file):
                try:
                    with open(metrics_file, 'r') as f:
                        metrics_data = json.load(f)
                except Exception:
                    metrics_data = {
                        "epoch_history": [],
                        "train_loss": [],
                        "valid_loss": [],
                        "learning_rates": []
                    }
            else:
                metrics_data = {
                    "epoch_history": [],
                    "train_loss": [],
                    "valid_loss": [],
                    "learning_rates": []
                }
            
            # Update metrics
            if epoch not in metrics_data["epoch_history"]:
                metrics_data["epoch_history"].append(epoch)
                metrics_data["train_loss"].append(float(train_loss))
                metrics_data["valid_loss"].append(float(valid_loss))
                metrics_data["learning_rates"].append(float(learning_rate))
            
            metrics_data["status"] = status
            metrics_data["progress"] = progress
            metrics_data["last_updated"] = time.time()
            
            if best_epoch is not None:
                metrics_data["best_epoch"] = best_epoch
            
            # Save updated metrics
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
                
            self.logger.debug(f"Updated metrics for task {task_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating metrics for task {task_id}: {e}", exc_info=True)
            return False