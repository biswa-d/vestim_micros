import time, os, sys, math, json
import torch
import logging
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
            
            # This is a simplified representation of what should happen.
            # In a real scenario, you would use task_info to load the correct data,
            # configure the model, optimizer, etc.
            # For example:
            # data_loader = training_setup_manager.get_data_loader(task_info['data_params'])
            # model = training_setup_manager.get_model(task_info['model_params']).to(device)
            # optimizer = torch.optim.Adam(model.parameters(), lr=task_info.get('learning_rate', 0.001))
            
            # The TrainingService will handle its own setup, including model and optimizer
            training_service = TrainingService(task_info=task_info, global_params=task_info, status_queue=status_queue)
            
            self.logger.info(f"[{job_id}] Setup complete. Starting training.")
            
            # The run_task method contains the full training loop
            training_service.run_task()
            
            # The status updates will be handled by the TrainingService, but we can send a final one
            self.logger.info(f"[{job_id}] Training process finished.")
            
            # --- Training Loop ---
            num_epochs = task_info.get('epochs', 10)
            history = []
            for epoch in range(1, num_epochs + 1):
                # In a real implementation, you would check a stop signal here
                # For now, the process is terminated by the JobManager
                
                status_queue.put((job_id, 'training', {
                    "message": f"Epoch {epoch}/{num_epochs}",
                    "progress": (epoch / num_epochs) * 100,
                    "history": history
                }))

                # Simulate training for one epoch
                # train_loss = training_service.train_epoch(data_loader)
                # val_loss = training_service.evaluate(val_loader)
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
            self.logger.error(f"[{job_id}] Error during training task: {e}", exc_info=True)
            status_queue.put((job_id, 'error', {"message": str(e)}))