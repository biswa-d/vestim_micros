import time
import torch
from threading import Thread
from src.gateway.src.job_manager import JobManager
from src.services.model_training.src.data_loader_service import DataLoaderService
from src.services.model_training.src.training_task_service import TrainingTaskService

class TrainingTaskManager:
    def __init__(self):
        self.job_manager = JobManager()
        self.data_loader_service = DataLoaderService()
        self.training_service = TrainingTaskService()
        self.current_task = None
        self.stop_requested = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

    def process_task(self, task, queue, update_progress_callback):
        try:
            """Process a single training task."""
            self.current_task = task
            self.stop_requested = False

            # Determine the device and store it
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Ensure that the task['model'] is correctly initialized
            if 'model' not in task or task['model'] is None:
                raise ValueError("Task does not contain a valid model instance.")

            # Step 1: Configuring DataLoader
            update_progress_callback({'status': 'Configuring DataLoader...'})
            # Create data loaders for the current task
            train_loader, val_loader = self.create_data_loaders(task)

            # Step 2: Starting Training
            update_progress_callback({'status': f'Training started on {device}. Creating DataLoaders'})
            # Start training in a separate thread
            training_thread = Thread(target=self.run_training, args=(task, queue, update_progress_callback, train_loader, val_loader, device))
            training_thread.setDaemon(True)
            training_thread.start()

        except Exception as e:
            queue.put({'task_error': str(e)})

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

    def run_training(self, task, queue, update_progress_callback, train_loader, val_loader, device):
        """Run the training process for a single task."""
        try:
            hyperparams = self.convert_hyperparams(task['hyperparams'])  # Convert hyperparameters first
            model = task['model'].to(device)  # Explicitly use the model from the task and move it to the device
            max_epochs = hyperparams['MAX_EPOCHS']
            valid_freq = hyperparams['ValidFrequency']
            valid_patience = hyperparams['VALID_PATIENCE']
            lr_drop_period = hyperparams['LR_DROP_PERIOD']
            
            best_validation_loss = float('inf')
            patience_counter = 0
            start_time = time.time()
            last_validation_time = start_time  # Initialize the last validation time

            optimizer = self.training_service.get_optimizer(model, lr=hyperparams['INITIAL_LR'])
            scheduler = self.training_service.get_scheduler(optimizer, lr_drop_period)

            for epoch in range(1, max_epochs + 1):
                if self.stop_requested:
                    break

                # Initialize hidden states for training phase
                h_s = torch.zeros(model.num_layers, hyperparams['BATCH_SIZE'], model.hidden_units).to(device)
                h_c = torch.zeros(model.num_layers, hyperparams['BATCH_SIZE'], model.hidden_units).to(device)

                # Train the model for one epoch
                train_loss = self.training_service.train_epoch(model, train_loader, optimizer, h_s, h_c, epoch, device)
                
                # Retrieve the current learning rate from the optimizer
                current_lr = optimizer.param_groups[0]['lr']
                print(f"current_lr: {current_lr}")
                # Validate the model at the specified frequency
                
                if epoch == 1 or epoch % valid_freq == 0 or epoch == max_epochs:
                    # Reinitialize hidden states for validation phase
                    h_s = torch.zeros(model.num_layers, hyperparams['BATCH_SIZE'], model.hidden_units).to(device)
                    h_c = torch.zeros(model.num_layers, hyperparams['BATCH_SIZE'], model.hidden_units).to(device)

                    val_loss = self.training_service.validate_epoch(model, val_loader, h_s, h_c, epoch, device)

                    # Calculate the time delta since the last validation
                    current_time = time.time()
                    delta_t_valid = current_time - last_validation_time
                    last_validation_time = current_time  # Update the last validation time

                    # Retrieve the current learning rate from the optimizer
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"current_lr: {current_lr}")
                    
                    # Prepare progress data
                    progress_data = {
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'elapsed_time': current_time - start_time,
                        'delta_t_valid': delta_t_valid,  # Include the time delta
                        'learning_rate': current_lr, # Include the current learning rate
                        'best_val_loss': best_validation_loss,  # Include the best validation loss so far
                    }

                    # Send progress update to the GUI via the callback
                    update_progress_callback(progress_data)

                    if val_loss < best_validation_loss:
                        best_validation_loss = val_loss
                        patience_counter = 0  # Reset patience counter if improvement is seen
                        self.save_model(task)  # Save the best model so far
                    else:
                        patience_counter += 1

                    if patience_counter > valid_patience:
                        print(f"Early stopping at epoch {epoch} due to no improvement.")
                        break

                scheduler.step()  # Adjust learning rate

            # Signal task completion
            queue.put({'task_completed': True})

        except Exception as e:
            # Signal that an error occurred and send the error message to the queue
            queue.put({'task_error': str(e)})

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

