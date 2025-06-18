import os
import json
import time
import logging
import torch
import sqlite3
import csv
from torch.cuda.amp import GradScaler, autocast
from vestim.backend.src.services.job_service import JobService
from vestim.backend.src.services.model_training.src.data_loader_service import DataLoaderService
from vestim.backend.src.services.model_training.src.training_task_service import TrainingTaskService as ModelTrainingService

class TrainingService:
    """Manages the execution of a single model training task."""

    def __init__(self, task_info: dict, global_params: dict, status_queue=None):
        self.logger = logging.getLogger(__name__)
        self.task_info = task_info
        self.global_params = global_params
        self.status_queue = status_queue
        self.data_loader_service = DataLoaderService()
        self.model_training_service = ModelTrainingService()
        self.stop_requested = False
        self.train_loss_history = []
        self.val_loss_history = []
        self.log_history = []
        
        self.device = torch.device(self.global_params.get('DEVICE_SELECTION', 'cpu'))
        self.use_amp = self.global_params.get('USE_MIXED_PRECISION', False) and self.device.type == 'cuda'
        
        self.logger.info(f"TrainingService initialized for task {self.task_info.get('task_id')} on device: {self.device} with AMP: {self.use_amp}")

    def run_task(self):
        """The main entry point to start processing the training task."""
        self.logger.info(f"Starting training task: {self.task_info.get('task_id')}")
        try:
            self.update_status("started", "Initializing...")
            
            self.setup_job_logging()
            train_loader, val_loader = self.create_data_loaders()
            
            self.run_training(train_loader, val_loader)

            if not self.stop_requested:
                self.update_status("complete", "Training finished successfully.")
                self.logger.info(f"Training task {self.task_info.get('task_id')} completed successfully.")
            else:
                self.update_status("stopped", "Training was stopped by user.")
                self.logger.info(f"Training task {self.task_info.get('task_id')} was stopped.")

        except Exception as e:
            self.logger.error(f"Error during training task {self.task_info.get('task_id')}: {e}", exc_info=True)
            self.update_status("error", str(e))
            raise

    def run_training(self, train_loader, val_loader):
        """The main training loop, including model setup, training, and validation."""
        hyperparams = self.task_info['hyperparams']
        
        # Initialize model, optimizer, etc.
        model = self.model_training_service.build_model(hyperparams).to(self.device)
        optimizer = self.model_training_service.get_optimizer(model, hyperparams)
        scheduler = self.model_training_service.get_scheduler(optimizer, hyperparams)
        criterion = torch.nn.MSELoss()
        scaler = GradScaler(enabled=self.use_amp)
        
        best_val_loss = float('inf')
        max_epochs = int(hyperparams.get('MAX_EPOCHS', 1))

        for epoch in range(1, max_epochs + 1):
            if self.stop_requested:
                break

            # Training phase
            model.train()
            total_train_loss = 0
            for batch in train_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                
                with autocast(enabled=self.use_amp):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            self.train_loss_history.append(avg_train_loss)

            # Validation phase
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    inputs, targets = batch
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    total_val_loss += loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            self.val_loss_history.append(avg_val_loss)
            
            log_message = f"Epoch {epoch}/{max_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
            self.logger.info(log_message)
            self.log_history.append(log_message)
            status_payload = {
                "epoch": epoch,
                "total_epochs": max_epochs,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "train_loss_history": self.train_loss_history,
                "val_loss_history": self.val_loss_history,
                "log_history": self.log_history,            }
            self.update_status("training", f"Epoch {epoch} complete", **status_payload)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_checkpoint(model, epoch, best_val_loss)

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

    def save_checkpoint(self, model, epoch, val_loss):
        """Saves the model checkpoint."""
        checkpoint_path = os.path.join(self.task_info['task_dir'], 'best_model.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
        }, checkpoint_path)
        self.logger.info(f"Saved new best model to {checkpoint_path} at epoch {epoch} with validation loss {val_loss:.4f}")

    def update_status(self, status: str, message: str, **kwargs):
        """Puts the current task status onto the status queue for the JobManager."""
        if not self.status_queue:
            return

        payload = {
            "message": message,
            "timestamp": time.time(),
            "task_id": self.task_info.get('task_id')
        }
        payload.update(kwargs)
        
        # Also add task-specific progress to the payload
        task_progress_data = {
            f"task_progress.{self.task_info.get('task_id')}": {
                "status": status,
                "message": message,
                "timestamp": time.time(),
                **kwargs
            }
        }
        payload.update(task_progress_data)
        
        try:
            self.status_queue.put((self.task_info['job_id'], status, payload))
        except Exception as e:
            self.logger.error(f"Could not put status on queue: {e}")

    def setup_job_logging(self):
        """Sets up the database and logging for the specific task."""
        self.logger.info("Setting up job-specific logging for the task.")
        db_log_file = self.task_info.get('db_log_file')
        if db_log_file:
            self.create_sql_tables(db_log_file)

    def create_sql_tables(self, db_log_file):
        """Creates the necessary SQL tables if they don't exist."""
        try:
            conn = sqlite3.connect(db_log_file)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS task_logs (
                    task_id TEXT, epoch INTEGER, train_loss REAL, val_loss REAL, 
                    elapsed_time REAL, avg_batch_time REAL, learning_rate REAL, 
                    best_val_loss REAL, num_learnable_params INTEGER, batch_size INTEGER, 
                    lookback INTEGER, max_epochs INTEGER, early_stopping INTEGER, 
                    model_memory_usage REAL, device TEXT, PRIMARY KEY(task_id, epoch)
                )
            ''')
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            self.logger.error(f"SQLite error in create_sql_tables: {e}")
            raise

    def create_data_loaders(self):
        """Creates and returns the data loaders for the task."""
        self.logger.info("Creating data loaders.")
        params = self.task_info['hyperparams']
        
        train_loader, val_loader = self.data_loader_service.create_data_loaders(
            folder_path=os.path.join(self.task_info['job_folder'], 'train_data', 'processed_data'),
            training_method=params.get('TRAINING_METHOD', 'Sequence-to-Sequence'),
            lookback=int(params.get('LOOKBACK', 50)),
            feature_cols=self.global_params['FEATURE_COLUMNS'],
            target_col=self.global_params['TARGET_COLUMN'],
            batch_size=int(params.get('BATCH_SIZE', 32)),
            num_workers=int(self.global_params.get('NUM_WORKERS', 4)),
            train_split=float(params.get('TRAIN_VAL_SPLIT', 0.7)),
            seed=int(params.get('SEED', 2000))
        )
        return train_loader, val_loader

    def stop_task(self):
        """Signals the training loop to stop."""
        self.logger.info(f"Received stop signal for task {self.task_info.get('task_id')}.")
        self.stop_requested = True