import time
import torch
from PyQt5.QtCore import QThread, pyqtSignal
from src.gateway.src.job_manager_qt import JobManager
from src.services.model_training.src.data_loader_service import DataLoaderService
from src.services.model_training.src.training_task_service import TrainingTaskService

class TrainingTaskManager:
    def __init__(self):
        self.job_manager = JobManager()
        self.data_loader_service = DataLoaderService()
        self.training_service = TrainingTaskService()
        self.current_task = None
        self.stop_requested = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training_thread = None  # Initialize the training thread here for PyQt


    def process_task(self, task, update_progress_callback):
        """Process a single training task."""
        try:
            self.current_task = task
            self.stop_requested = False

            # Ensure the task contains a valid model
            if 'model' not in task or task['model'] is None:
                raise ValueError("Task does not contain a valid model instance.")

            # Configuring DataLoader (send progress update via signal)
            update_progress_callback.emit({'status': 'Configuring DataLoader...'})

            # Create data loaders for the task
            train_loader, val_loader = self.create_data_loaders(task)

            update_progress_callback.emit({'status': f'Training LSTM model for {task["hyperparams"]["MAX_EPOCHS"]} epochs...'})
            # Call the training directly (this will be done within the QThread)
            self.run_training(task, update_progress_callback, train_loader, val_loader, self.device)

        except Exception as e:
            update_progress_callback.emit({'task_error': str(e)})


    def create_data_loaders(self, task):
        """Create data loaders for the current task."""
        lookback = task['data_loader_params']['lookback']
        batch_size = task['data_loader_params']['batch_size']
        num_workers = 4

        train_loader, val_loader = self.data_loader_service.create_data_loaders(
            folder_path=self.job_manager.get_train_folder(),  # Adjusted to use the correct folder
            lookback=lookback, 
            batch_size=batch_size, 
            num_workers=num_workers
        )

        return train_loader, val_loader

    def run_training(self, task, update_progress_callback, train_loader, val_loader, device):
        """Run the training process for a single task."""
        try:
            hyperparams = self.convert_hyperparams(task['hyperparams'])
            model = task['model'].to(device)
            max_epochs = hyperparams['MAX_EPOCHS']
            valid_freq = hyperparams['ValidFrequency']
            valid_patience = hyperparams['VALID_PATIENCE']
            lr_drop_period = hyperparams['LR_DROP_PERIOD']

            best_validation_loss = float('inf')
            patience_counter = 0
            start_time = time.time()
            last_validation_time = start_time

            optimizer = self.training_service.get_optimizer(model, lr=hyperparams['INITIAL_LR'])
            scheduler = self.training_service.get_scheduler(optimizer, lr_drop_period)

            for epoch in range(1, max_epochs + 1):
                if self.stop_requested:  # Ensure thread safety here
                    print("Stopping training...")
                    break

                # Initialize hidden states for training phase
                h_s = torch.zeros(model.num_layers, hyperparams['BATCH_SIZE'], model.hidden_units).to(device)
                h_c = torch.zeros(model.num_layers, hyperparams['BATCH_SIZE'], model.hidden_units).to(device)

                # Train the model for one epoch
                train_loss = self.training_service.train_epoch(model, train_loader, optimizer, h_s, h_c, epoch, device, self.stop_requested)

                if self.stop_requested:
                    print("Training stopped after training phase.")
                    break

                # Validate the model at the specified frequency
                if epoch == 1 or epoch % valid_freq == 0 or epoch == max_epochs:
                    h_s = torch.zeros(model.num_layers, hyperparams['BATCH_SIZE'], model.hidden_units).to(device)
                    h_c = torch.zeros(model.num_layers, hyperparams['BATCH_SIZE'], model.hidden_units).to(device)

                    val_loss = self.training_service.validate_epoch(model, val_loader, h_s, h_c, epoch, device, self.stop_requested)
                    if self.stop_requested:
                        print("Training stopped after validation phase.")
                        break

                    current_time = time.time()
                    delta_t_valid = current_time - last_validation_time
                    last_validation_time = current_time

                    current_lr = optimizer.param_groups[0]['lr']

                    progress_data = {
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'elapsed_time': current_time - start_time,
                        'delta_t_valid': delta_t_valid,
                        'learning_rate': current_lr,
                        'best_val_loss': best_validation_loss,
                    }

                    # Proper signal emission
                    update_progress_callback.emit(progress_data)

                    if val_loss < best_validation_loss:
                        best_validation_loss = val_loss
                        patience_counter = 0
                        self.save_model(task)
                    else:
                        patience_counter += 1

                    if patience_counter > valid_patience:
                        print(f"Early stopping at epoch {epoch} due to no improvement.")
                        break

                scheduler.step()

            if self.stop_requested:
                print("Training was stopped early. Saving Model...")
                self.save_model(task)

            # Emit task completion signal
            update_progress_callback.emit({'task_completed': True})

        except Exception as e:
            update_progress_callback.emit({'task_error': str(e)})


    def convert_hyperparams(self, hyperparams):
        """Converts all relevant hyperparameters to the correct types."""
        hyperparams['LAYERS'] = int(hyperparams['LAYERS'])
        hyperparams['HIDDEN_UNITS'] = int(hyperparams['HIDDEN_UNITS'])
        hyperparams['BATCH_SIZE'] = int(hyperparams['BATCH_SIZE'])
        hyperparams['MAX_EPOCHS'] = int(hyperparams['MAX_EPOCHS'])
        hyperparams['INITIAL_LR'] = float(hyperparams['INITIAL_LR'])
        hyperparams['LR_DROP_PERIOD'] = int(hyperparams['LR_DROP_PERIOD'])
        hyperparams['VALID_PATIENCE'] = int(hyperparams['VALID_PATIENCE'])
        hyperparams['ValidFrequency'] = int(hyperparams['ValidFrequency'])
        hyperparams['LOOKBACK'] = int(hyperparams['LOOKBACK'])
        hyperparams['REPETITIONS'] = int(hyperparams['REPETITIONS'])
        return hyperparams

    def save_model(self, task):
        """Save the trained model to disk."""
        model_path = task.get('model_path', None)
        if model_path is None:
            raise ValueError("Model path not found in task.")

        model = task['model']
        if model is None:
            raise ValueError("No model instance found in task.")

        # Save the model state dictionary
        torch.save(model.state_dict(), model_path)

    def stop_task(self):
        self.stop_requested = True  # Set the flag to request a stop
        if self.training_thread and self.training_thread.isRunning():  # Use isRunning() instead of is_alive()
            print("Waiting for the training thread to finish before saving the model...")
            self.training_thread.quit()  # Gracefully stop the thread
            self.training_thread.wait(7000)  # Wait for the thread to finish cleanly
            print("Training thread has finished. Proceeding to save the model.")
