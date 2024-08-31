import time
import torch
from threading import Thread
from src.gateway.src.job_manager import JobManager
from src.services.model_training.src.data_loader_service import DataLoaderService
from src.services.model_training.src.training_service import TrainingService

class TrainingTaskManager:
    def __init__(self):
        self.job_manager = JobManager()
        self.data_loader_service = DataLoaderService()
        self.training_service = TrainingService()
        self.current_task = None
        self.stop_requested = False

    def process_task(self, task, queue, update_progress_callback):
        """Process a single training task."""
        self.current_task = task
        self.stop_requested = False

        # Ensure that the task['model'] is correctly initialized
        if 'model' not in task or task['model'] is None:
            raise ValueError("Task does not contain a valid model instance.")

        # Create data loaders for the current task
        train_loader, val_loader = self.create_data_loaders(task)

        # Start training in a separate thread
        training_thread = Thread(target=self.run_training, args=(task, queue, update_progress_callback, train_loader, val_loader))
        training_thread.setDaemon(True)
        training_thread.start()

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

    def run_training(self, task, queue, update_progress_callback, train_loader, val_loader):
        """Run the training process for a single task."""
        try:
            hyperparams = task['hyperparams']
            model = task['model']  # Explicitly use the model from the task
            max_epochs = hyperparams['MAX_EPOCHS']
            valid_freq = hyperparams['ValidFrequency']
            valid_patience = hyperparams['VALID_PATIENCE']
            lr_drop_period = hyperparams['LR_DROP_PERIOD']
            
            best_validation_loss = float('inf')
            patience_counter = 0
            start_time = time.time()
            last_validation_time = start_time  # Initialize the last validation time

            optimizer = self.training_service._get_optimizer(model, lr=hyperparams['INITIAL_LR'])
            scheduler = self.training_service._get_scheduler(optimizer, lr_drop_period)

            for epoch in range(1, max_epochs + 1):
                if self.stop_requested:
                    break

                # Train the model for one epoch
                train_loss = self.training_service.train_epoch(model, train_loader, optimizer)

                # Validate the model at the specified frequency
                if epoch % valid_freq == 0 or epoch == max_epochs:
                    val_loss = self.training_service.validate_epoch(model, val_loader)

                    # Calculate the time delta since the last validation
                    current_time = time.time()
                    delta_t_valid = current_time - last_validation_time
                    last_validation_time = current_time  # Update the last validation time

                    # Prepare progress data
                    progress_data = {
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'elapsed_time': current_time - start_time,
                        'delta_t_valid': delta_t_valid  # Include the time delta
                    }

                    # Send progress update to the GUI via the callback
                    update_progress_callback(progress_data)

                    if val_loss < best_validation_loss:
                        best_validation_loss = val_loss
                        patience_counter = 0  # Reset patience counter if improvement is seen
                    else:
                        patience_counter += 1

                    if patience_counter > valid_patience:
                        print(f"Early stopping at epoch {epoch} due to no improvement.")
                        break

                scheduler.step()  # Adjust learning rate

            # Save the trained model after the task is complete
            self.save_model(task)

            # Signal task completion
            queue.put({'task_completed': True})

        except Exception as e:
            # Signal that an error occurred and send the error message to the queue
            queue.put({'task_error': str(e)})


    def save_model(self, task):
        """Save the trained model to disk."""
        model_path = task['hyperparams']['model_path']
        model = task['model']

        # Check if the model instance is valid before saving
        if model is None:
            raise ValueError("No model instance found in task.")

        # Save the model state dictionary
        torch.save(model.state_dict(), model_path)


        def stop_task(self):
            """Stop the currently running task."""
            self.stop_requested = True

