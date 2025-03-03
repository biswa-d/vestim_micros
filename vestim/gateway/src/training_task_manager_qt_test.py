import time, os, sys, math
import csv
import sqlite3
import torch
from PyQt5.QtCore import QThread, pyqtSignal
from vestim.gateway.src.job_manager_qt import JobManager
from vestim.gateway.src.training_setup_manager_qt_test import VEstimTrainingSetupManager
from vestim.services.model_training.src.data_loader_service_test import DataLoaderService
from vestim.services.model_training.src.training_task_service_test import TrainingTaskService
import logging, wandb

class TrainingTaskManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.job_manager = JobManager()
        self.data_loader_service = DataLoaderService()
        self.training_service = TrainingTaskService()
        self.training_setup_manager = VEstimTrainingSetupManager()
        self.current_task = None
        self.stop_requested = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    def log_to_csv(self, task, epoch, train_loss, val_loss, elapsed_time, current_lr, best_val_loss, delta_t_epoch):
        """Log richer data to CSV file."""
        csv_log_file = task['csv_log_file']  # Fetch the csv log file path from the task
        with open(csv_log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Epoch', 'Train Loss', 'Val Loss', 'Elapsed Time', 'Learning Rate', 'Best Val Loss', 'Train Time Per Epoch'])
            writer.writerow({
                'Epoch': epoch,
                'Train Loss': train_loss,
                'Val Loss': val_loss,
                'Elapsed Time': elapsed_time,
                'Learning Rate': current_lr,
                'Best Val Loss': best_val_loss,
                'Train Time Per Epoch': delta_t_epoch
            })

    def log_to_sqlite(self, task, epoch, train_loss, val_loss, best_val_loss, elapsed_time, avg_batch_time, early_stopping, model_memory_usage, current_lr):
        """Log epoch-level data to a SQLite database with the updated learning rate."""
        sqlite_db_file = task['db_log_file']
        conn = sqlite3.connect(sqlite_db_file)
        cursor = conn.cursor()

        # Insert data with updated learning rate
        cursor.execute('''INSERT INTO task_logs (task_id, epoch, train_loss, val_loss, elapsed_time, avg_batch_time, learning_rate, 
                        best_val_loss, num_learnable_params, batch_size, lookback, max_epochs, early_stopping, model_memory_usage, device)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (task['task_id'], epoch, train_loss, val_loss, elapsed_time, avg_batch_time, current_lr, best_val_loss,
                    task['hyperparams']['NUM_LEARNABLE_PARAMS'], task['hyperparams']['BATCH_SIZE'], task['hyperparams']['LOOKBACK'], 
                    task['hyperparams']['MAX_EPOCHS'], early_stopping, model_memory_usage, self.device.type))

        conn.commit()
        conn.close()

    def process_task(self, task, update_progress_callback):
        """Process a single training task and set up logging."""
        try:
            self.logger.info(f"Starting task with hyperparams: {task['hyperparams']}")

            # Setup logging (SQL and CSV) for the job
            self.setup_job_logging(task)

            # Task initialization and logging
            self.current_task = task
            self.stop_requested = False

            # Ensure the task contains a valid model
            if 'model' not in task or task['model'] is None:
                raise ValueError("Task does not contain a valid model instance.")

            # Configuring DataLoader (send progress update via signal)
            self.logger.info("Configuring DataLoader")
            update_progress_callback.emit({'status': 'Configuring DataLoader...'})

            # Create data loaders for the task
            train_loader, val_loader = self.create_data_loaders(task)
            self.logger.info(f"DataLoader configured for task: {task['hyperparams']}")

            # Update progress for starting training
            update_progress_callback.emit({'status': f'Training LSTM model for {task["hyperparams"]["MAX_EPOCHS"]} epochs...'})

            # Call the training method and pass logging information (task_id, db path)
            self.run_training(task, update_progress_callback, train_loader, val_loader, self.device)

        except Exception as e:
            self.logger.error(f"Error during task processing: {str(e)}")
            update_progress_callback.emit({'task_error': str(e)})

    def setup_job_logging(self, task):
        """
        Set up the database and logging environment for the job.
        This ensures that the database tables are created if they do not exist.
        """
        job_id = self.job_manager.get_job_id()  # Get the job ID
        model_dir = task.get('model_dir')  # Path where task-related logs are stored

        # Retrieve log file paths from the task
        csv_log_file = task.get('csv_log_file')
        db_log_file = task.get('db_log_file')

        # Log information about the task and log file paths
        print(f"Setting up logging for job: {job_id}")
        print(f"Model directory: {model_dir}")
        print(f"Log files for task {task['task_id']} are: {csv_log_file}, {db_log_file}")

        # Ensure the model_dir exists
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)  # Create the directory if it does not exist

        # Create SQLite tables if they do not exist
        self.create_sql_tables(db_log_file)

    def create_sql_tables(self, db_log_file):
        """Create the necessary SQL tables for task-level and batch-level logging."""
        try:
            # Ensure the database file path is valid
            if not os.path.isfile(db_log_file):
                self.logger.info(f"Creating new database file at: {db_log_file}")

            # Connect to the database and create tables
            conn = sqlite3.connect(db_log_file)
            cursor = conn.cursor()

            # Create table for high-level task logs (epoch-level)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS task_logs (
                    task_id TEXT,
                    epoch INTEGER,
                    train_loss REAL,
                    val_loss REAL,
                    elapsed_time REAL,
                    avg_batch_time REAL,  -- New column for average batch time
                    learning_rate REAL,
                    best_val_loss REAL,
                    num_learnable_params INTEGER,
                    batch_size INTEGER,
                    lookback INTEGER,
                    max_epochs INTEGER,
                    early_stopping INTEGER,  -- New column for early stopping flag (1 if stopped early, 0 otherwise)
                    model_memory_usage REAL,  -- New column for memory usage (optional)
                    device TEXT,  -- Add this new column for the device
                    PRIMARY KEY(task_id, epoch)
                )
            ''')

            # Create table for fine-grained batch logs
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS batch_logs (
                    task_id TEXT,
                    epoch INTEGER,
                    batch_idx INTEGER,
                    batch_time REAL,
                    phase TEXT,
                    learning_rate REAL,
                    num_learnable_params INTEGER,
                    batch_size INTEGER,
                    lookback INTEGER,
                    device TEXT,  -- Add the device column here
                    FOREIGN KEY(task_id, epoch) REFERENCES task_logs(task_id, epoch)
                )
            ''')

            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            self.logger.error(f"SQLite error: {e}")
            raise e

    def create_data_loaders(self, task):
        """Create data loaders for the current task."""
        lookback = task['data_loader_params']['lookback']
        batch_size = task['data_loader_params']['batch_size']
        num_workers = 4
        feature_cols = task['data_loader_params']['feature_columns']
        target_col = task['data_loader_params']['target_column']

        self.logger.info("Creating data loaders")
        train_loader, val_loader = self.data_loader_service.create_data_loaders(
            folder_path=self.job_manager.get_train_folder(),  # Adjusted to use the correct folder
            lookback=lookback,
            feature_cols=feature_cols,
            target_col=target_col, 
            batch_size=batch_size, 
            num_workers=num_workers
        )

        return train_loader, val_loader

    def run_training(self, task, update_progress_callback, train_loader, val_loader, device):
        """Run the training process for a single task."""
        try:
            self.logger.info("Starting training loop")
            hyperparams = self.convert_hyperparams(task['hyperparams'])
            model = task['model'].to(device)
            
            max_epochs = hyperparams['MAX_EPOCHS']
            valid_freq = hyperparams['ValidFrequency']
            valid_patience = hyperparams['VALID_PATIENCE']
            #patience_threshold = int(valid_patience * 0.5) 
            current_lr = hyperparams['INITIAL_LR']
            lr_drop_period = hyperparams['LR_DROP_PERIOD']
            lr_drop_factor = hyperparams.get('LR_DROP_FACTOR', 0.1)
            # Define a buffer period after which LR drops can happen again, e.g., 100 epochs.
            lr_drop_buffer = 50
            last_lr_drop_epoch = 0  # Initialize the epoch of the last LR drop
            # weight_decay = hyperparams.get('WEIGHT_DECAY', 1e-5)

            best_validation_loss = float('inf')
            patience_counter = 0
            start_time = time.time()
            last_validation_time = start_time
            early_stopping = False  # Initialize early stopping flag

            self.optimizer = torch.optim.Adam(model.parameters(), lr=current_lr)
            # self.scheduler = self.training_service.get_scheduler(self.optimizer, gamma=lr_drop_factor)
            #self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=lr_drop_factor)
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=lr_drop_period,  # Number of epochs between drops
                gamma=lr_drop_factor       # Multiplicative factor for the drop
            )
            optimizer = self.optimizer
            scheduler = self.scheduler

            # Log the training progress for each epoch
            def format_time(seconds):
                """Format time into mm:ss format."""
                minutes = seconds // 60
                seconds = seconds % 60
                return f"{int(minutes)}:{int(seconds):02d}"

            # Training loop
            for epoch in range(1, max_epochs + 1):
                if self.stop_requested:  # Ensure thread safety here
                    self.logger.info("Training stopped by user")
                    print("Stopping training...")
                    break

                # Initialize hidden states for training phase
                h_s = torch.zeros(model.num_layers, hyperparams['BATCH_SIZE'], model.hidden_units).to(device)
                h_c = torch.zeros(model.num_layers, hyperparams['BATCH_SIZE'], model.hidden_units).to(device)

                # Measure time for the training loop
                epoch_start_time = time.time()

                # Train the model for one epoch
                avg_batch_time, train_loss = self.training_service.train_epoch(model, train_loader, optimizer, h_s, h_c, epoch, device, self.stop_requested, task)

                epoch_end_time = time.time()
                epoch_duration = epoch_end_time - epoch_start_time
                formatted_epoch_time = format_time(epoch_duration)  # Convert epoch time to mm:ss format

                if self.stop_requested:
                    self.logger.info("Training stopped by user")
                    print("Training stopped after training phase.")
                    self.logger.info("Training stopped after training phase.")
                    break

                # Only validate at specified frequency
                if epoch == 1 or epoch % valid_freq == 0 or epoch == max_epochs:
                    h_s = torch.zeros(model.num_layers, hyperparams['BATCH_SIZE'], model.hidden_units).to(device)
                    h_c = torch.zeros(model.num_layers, hyperparams['BATCH_SIZE'], model.hidden_units).to(device)

                    val_loss = self.training_service.validate_epoch(model, val_loader, h_s, h_c, epoch, device, self.stop_requested, task)
                    self.logger.info(f"Epoch {epoch} | Train Loss: {train_loss} | Val Loss: {val_loss} | Epoch Time: {formatted_epoch_time} | Best Val Loss: {best_validation_loss}")

                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    delta_t_epoch = (current_time - last_validation_time) / valid_freq
                    last_validation_time = current_time

                    current_lr = optimizer.param_groups[0]['lr']
                    
                    if val_loss < best_validation_loss:
                        print(f"Validation loss improved from {best_validation_loss:.6f} to {val_loss:.6f}. Saving model...")
                        best_validation_loss = val_loss
                        self.save_model(task)
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    progress_data = {
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'elapsed_time': elapsed_time,
                        'delta_t_epoch': formatted_epoch_time,
                        'learning_rate': current_lr,
                        'best_val_loss': best_validation_loss,
                    }
                    # Emit progress after validation
                    update_progress_callback.emit(progress_data) 

                    if patience_counter > valid_patience:
                        early_stopping = True
                        print(f"Early stopping at epoch {epoch} due to no improvement.")
                        self.logger.info(f"Early stopping at epoch {epoch} due to no improvement.")
                        
                        # Ensure that we log the final epoch before breaking out
                        model_memory_usage = torch.cuda.memory_allocated() if torch.cuda.is_available() else sys.getsizeof(model)
                        model_memory_usage_mb = model_memory_usage / (1024 * 1024)  # Convert to MB
                        # self.log_to_csv(task, epoch, train_loss, val_loss, elapsed_time, current_lr, best_validation_loss, delta_t_epoch)
                        self.log_to_sqlite(
                            task=task,
                            epoch=epoch,
                            train_loss=train_loss,
                            val_loss=val_loss,
                            best_val_loss=best_validation_loss,
                            elapsed_time=elapsed_time,
                            avg_batch_time=avg_batch_time,
                            early_stopping=early_stopping,
                            model_memory_usage=round(model_memory_usage_mb, 3),  # Memory in MB
                            current_lr=current_lr  # Pass updated learning rate here
                        )

                        break

                # Log data to CSV and SQLite after each epoch (whether validated or not)
                print(f"Checking log files for the task: {task['task_id']}: task['csv_log_file'], task['db_log_file']")

                # Save log data to CSV and SQLite
                # self.log_to_csv(task, epoch, train_loss, val_loss, elapsed_time, current_lr, best_validation_loss, delta_t_epoch)
                model_memory_usage = torch.cuda.memory_allocated() if torch.cuda.is_available() else sys.getsizeof(model)
                model_memory_usage_mb = model_memory_usage / (1024 * 1024)  # Convert to MB
                
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Current learning rate updated at epoch {epoch}: {current_lr: .8f}\n")
                logging.info(f"Current learning rate updated at epoch {epoch}: {current_lr: .8f}\n")
                # Scheduler step condition: Either when lr_drop_period is reached or patience_counter exceeds the threshold
                # Scheduler step condition: Check for drop period or patience_counter with buffer consideration
                # if (epoch % lr_drop_period == 0 or patience_counter > patience_threshold) and (epoch - last_lr_drop_epoch > lr_drop_buffer):
                #     print(f"Learning rate before scheduler step: {optimizer.param_groups[0]['lr']: .8f}\n")
                #     scheduler.step()
                #     current_lr = optimizer.param_groups[0]['lr']
                #     print(f"Current learning rate updated at epoch {epoch}: {current_lr: .8f}\n")
                #     logging.info(f"Current learning rate updated at epoch {epoch}: {current_lr: .8f}\n")
                #     last_lr_drop_epoch = epoch
                # else:
                #     print(f"Epoch {epoch}: No LR drop. patience_counter={patience_counter}, patience_threshold={patience_threshold}\n")
    
                # Log data to SQLite
                #commented out for testing db error
                # self.log_to_sqlite(
                #     task=task,
                #     epoch=epoch,
                #     train_loss=train_loss,
                #     val_loss=val_loss,
                #     best_val_loss=best_validation_loss,
                #     elapsed_time=elapsed_time,
                #     avg_batch_time=avg_batch_time,
                #     early_stopping=early_stopping,
                #     model_memory_usage=round(model_memory_usage_mb, 3),  # Memory in MB
                #     current_lr=current_lr  # Pass updated learning rate here
                # )

            if self.stop_requested:
                print("Training was stopped early. Exiting...")
                self.logger.info("Training was stopped early. Exiting...")

            update_progress_callback.emit({'task_completed': True})
            self.logger.info("Training task completed")

        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            update_progress_callback.emit({'task_error': str(e)})


    def convert_hyperparams(self, hyperparams):
        """Converts all relevant hyperparameters to the correct types."""
        hyperparams['LAYERS'] = int(hyperparams['LAYERS'])
        hyperparams['HIDDEN_UNITS'] = int(hyperparams['HIDDEN_UNITS'])
        hyperparams['BATCH_SIZE'] = int(hyperparams['BATCH_SIZE'])
        hyperparams['MAX_EPOCHS'] = int(hyperparams['MAX_EPOCHS'])
        hyperparams['INITIAL_LR'] = float(hyperparams['INITIAL_LR'])
        hyperparams['LR_DROP_PERIOD'] = int(hyperparams['LR_DROP_PERIOD'])
        hyperparams['LR_DROP_FACTOR'] = float(hyperparams['LR_DROP_FACTOR'])
        hyperparams['VALID_PATIENCE'] = int(hyperparams['VALID_PATIENCE'])
        hyperparams['ValidFrequency'] = int(hyperparams['ValidFrequency'])
        hyperparams['LOOKBACK'] = int(hyperparams['LOOKBACK'])
        hyperparams['REPETITIONS'] = int(hyperparams['REPETITIONS'])
        return hyperparams

    def save_model(self, task):
        """Save the trained model to disk."""
        model_path = task.get('model_path', None)
        if model_path is None:
            self.logger.error("Model path not found in task.")
            raise ValueError("Model path not found in task.")

        model = task['model']
        if model is None:
            self.logger.error("No model instance found in task.")
            raise ValueError("No model instance found in task.")

        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    def stop_task(self):
        self.stop_requested = True  # Set the flag to request a stop
        if self.training_thread and self.training_thread.isRunning():  # Use isRunning() instead of is_alive()
            print("Waiting for the training thread to finish before saving the model...")
            self.training_thread.quit()  # Gracefully stop the thread
            self.training_thread.wait(7000)  # Wait for the thread to finish cleanly
            print("Training thread has finished. Proceeding to save the model.")
            self.logger.info("Training thread has finished. Proceeding to save the model.")
