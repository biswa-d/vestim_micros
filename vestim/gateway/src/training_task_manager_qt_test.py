import time, os
import csv
import sqlite3
import torch
from PyQt5.QtCore import QThread, pyqtSignal
from vestim.gateway.src.job_manager_qt import JobManager
from vestim.services.model_training.src.data_loader_service_padfil import DataLoaderService
from vestim.services.model_training.src.training_task_service import TrainingTaskService
import logging, wandb

class TrainingTaskManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.job_manager = JobManager()
        self.data_loader_service = DataLoaderService()
        self.training_service = TrainingTaskService()
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


    def setup_csv_and_sql_logging(self, task):
        """Setup CSV and SQLite logging files for richer training logs."""
        model_dir = task['model_dir']
        
        # CSV setup
        csv_log_file = os.path.join(model_dir, "train_log.csv")
        file_exists = os.path.isfile(csv_log_file)
        self.csv_log_file = csv_log_file

        with open(csv_log_file, 'a', newline='') as f:
            fieldnames = ['Epoch', 'Train Loss', 'Val Loss', 'Elapsed Time', 'Learning Rate', 'Best Val Loss', 'Train Time Per Epoch']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()  # Only write the header once

        # SQLite setup
        self.sqlite_db_file = os.path.join(model_dir, "train_log.db")
        conn = sqlite3.connect(self.sqlite_db_file)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS training_log 
                            (epoch INTEGER, train_loss REAL, val_loss REAL, elapsed_time REAL, 
                            learning_rate REAL, best_val_loss REAL, train_time_epoch REAL)''')
        conn.commit()
        conn.close()

    def log_to_csv(self, epoch, train_loss, val_loss, elapsed_time, current_lr, best_val_loss, delta_t_epoch):
        """ Log richer data to CSV file """
        with open(self.csv_log_file, 'a', newline='') as f:
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

    def log_to_sqlite(self, epoch, train_loss, val_loss, elapsed_time, current_lr, best_val_loss, delta_t_epoch):
        """ Log richer data to SQLite database """
        conn = sqlite3.connect(self.sqlite_db_file)
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO training_log (epoch, train_loss, val_loss, elapsed_time, learning_rate, best_val_loss, train_time_epoch) 
                          VALUES (?, ?, ?, ?, ?, ?, ?)''', 
                       (epoch, train_loss, val_loss, elapsed_time, current_lr, best_val_loss, delta_t_epoch))
        conn.commit()
        conn.close()

    def process_task(self, task, update_progress_callback):
        """Process a single training task and set up logging."""
        try:
            self.logger.info(f"Starting task with hyperparams: {task['hyperparams']}")
            
            # Generate a unique task ID for this task (e.g., based on time or task details)
            task_id = f"task_{int(time.time())}"  # Example of task_id generation

            # Set the task ID in the task dictionary for future reference
            task['task_id'] = task_id

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
        Set up the database and logging environment for the job. This ensures
        that logging is consistent across all tasks in this job.
        """
        job_id = self.job_manager.get_job_id  # Assuming the job_id is available
        model_dir = task.get('model_dir')  # Path where task-related logs will be stored

        # Define paths for CSV and SQLite logs
        csv_log_file = os.path.join(model_dir, f"{job_id}_train_log.csv")
        db_log_file = os.path.join(model_dir, f"{job_id}_train_log.db")
        
        # Store the log paths in the task dictionary for future reference
        task['csv_log_file'] = csv_log_file
        task['db_log_file'] = db_log_file

        # Create SQLite tables if they do not exist
        self.create_sql_tables(db_log_file)

    def create_sql_tables(self, db_log_file):
        """Create the necessary SQL tables for task-level and batch-level logging."""
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
                learning_rate REAL,
                best_val_loss REAL,
                num_learnable_params INTEGER,  -- Task metadata
                batch_size INTEGER,             -- Task metadata
                lookback INTEGER,               -- Task metadata
                max_epochs INTEGER,             -- Task metadata
                PRIMARY KEY(task_id, epoch)
            )
        ''')

        # Create table for fine-grained batch logs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS batch_logs (
                task_id TEXT,
                epoch INTEGER,
                batch_idx INTEGER,
                batch_train_loss REAL,
                batch_time REAL,                -- Time per batch
                learning_rate REAL,
                num_learnable_params INTEGER,   -- Task metadata (repeated)
                batch_size INTEGER,             -- Task metadata (repeated)
                lookback INTEGER,               -- Task metadata (repeated)
                FOREIGN KEY(task_id, epoch) REFERENCES task_logs(task_id, epoch)
            )
        ''')

        conn.commit()
        conn.close()

    def create_data_loaders(self, task):
        """Create data loaders for the current task."""
        lookback = task['data_loader_params']['lookback']
        batch_size = task['data_loader_params']['batch_size']
        num_workers = 4

        self.logger.info("Creating data loaders")
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
            self.logger.info("Starting training loop")
            hyperparams = self.convert_hyperparams(task['hyperparams'])
            model = task['model'].to(device)
            max_epochs = hyperparams['MAX_EPOCHS']
            valid_freq = hyperparams['ValidFrequency']
            valid_patience = hyperparams['VALID_PATIENCE']
            lr_drop_period = hyperparams['LR_DROP_PERIOD']

            # Log paths
            model_dir = task["model_dir"]
            csv_log_file = os.path.join(model_dir, "train_log.csv")
            sqlite_db_file = os.path.join(model_dir, "train_log.db")

            # Setup CSV and SQLite logging (in a centralized manner for the entire task)
            self.setup_csv_and_sql_logging(csv_log_file, sqlite_db_file)

            best_validation_loss = float('inf')
            patience_counter = 0
            start_time = time.time()
            last_validation_time = start_time

            optimizer = self.training_service.get_optimizer(model, lr=hyperparams['INITIAL_LR'])
            scheduler = self.training_service.get_scheduler(optimizer, lr_drop_period)

            # Setup CSV and SQLite logging
            self.setup_csv_and_sql_logging(task)

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
                train_loss = self.training_service.train_epoch(model, train_loader, optimizer, h_s, h_c, epoch, device, self.stop_requested, csv_log_file, sqlite_db_file)

                epoch_end_time = time.time()
                epoch_duration = epoch_end_time - epoch_start_time
                formatted_epoch_time = format_time(epoch_duration)  # Convert epoch time to mm:ss format
                
                # Log the training progress for each epoch
                def format_time(seconds):
                    """Format time into mm:ss format."""
                    minutes = seconds // 60
                    seconds = seconds % 60
                    return f"{int(minutes)}:{int(seconds):02d}"

                if self.stop_requested:
                    self.logger.info("Training stopped by user")
                    print("Training stopped after training phase.")
                    break

                # Validate the model at the specified frequency
                if epoch == 1 or epoch % valid_freq == 0 or epoch == max_epochs:
                    h_s = torch.zeros(model.num_layers, hyperparams['BATCH_SIZE'], model.hidden_units).to(device)
                    h_c = torch.zeros(model.num_layers, hyperparams['BATCH_SIZE'], model.hidden_units).to(device)

                    val_loss = self.training_service.validate_epoch(model, val_loader, h_s, h_c, epoch, device, self.stop_requested, csv_log_file, sqlite_db_file)
                    self.logger.info(f"Epoch {epoch} | Train Loss: {train_loss} | Val Loss: {val_loss} | Epoch Time: {formatted_epoch_time}")

                    if self.stop_requested:
                        self.logger.info("Training stopped by user")
                        print("Training stopped after validation phase.")
                        break

                    current_time = time.time()
                    delta_t_epoch = (current_time - last_validation_time) / valid_freq
                    last_validation_time = current_time

                    current_lr = optimizer.param_groups[0]['lr']

                    progress_data = {
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'elapsed_time': current_time - start_time,
                        'delta_t_epoch': formatted_epoch_time,  # Use the formatted time here
                        'learning_rate': current_lr,
                        'best_val_loss': best_validation_loss,
                    }

                    # Log richer data to CSV and SQLite
                    # Save log data to CSV and SQLite
                    self.log_to_csv(
                        epoch=epoch,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        elapsed_time=progress_data['elapsed_time'],
                        current_lr=current_lr,
                        best_val_loss=best_validation_loss,
                        delta_t_epoch=delta_t_epoch
                    )

                    self.log_to_sqlite(
                        epoch=epoch,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        elapsed_time=progress_data['elapsed_time'],
                        current_lr=current_lr,
                        best_val_loss=best_validation_loss,
                        delta_t_epoch=delta_t_epoch
                    )
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

        # Save the model state dictionary
        torch.save(model.state_dict(), model_path)

    def stop_task(self):
        self.stop_requested = True  # Set the flag to request a stop
        if self.training_thread and self.training_thread.isRunning():  # Use isRunning() instead of is_alive()
            print("Waiting for the training thread to finish before saving the model...")
            self.training_thread.quit()  # Gracefully stop the thread
            self.training_thread.wait(7000)  # Wait for the thread to finish cleanly
            print("Training thread has finished. Proceeding to save the model.")
            self.logger.info("Training thread has finished. Proceeding to save the model.")
