import time, os, sys, math
import csv
import sqlite3
import torch
from PyQt5.QtCore import QThread, pyqtSignal
from vestim.gateway.src.job_manager_qt import JobManager
from vestim.gateway.src.training_setup_manager_qt import VEstimTrainingSetupManager
from vestim.services.model_training.src.data_loader_service import DataLoaderService
from vestim.services.model_training.src.training_task_service import TrainingTaskService
import logging, wandb

def format_time(seconds):
    """Convert seconds to mm:ss format."""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

class TrainingTaskManager:
    def __init__(self, global_params=None):
        self.logger = logging.getLogger(__name__)
        self.job_manager = JobManager()
        self.data_loader_service = DataLoaderService()
        self.training_service = TrainingTaskService()
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
                # Attempt to use the specific CUDA device. torch.device will raise an error if invalid.
                self.device = torch.device(selected_device_str)
                if not torch.cuda.is_available() or torch.cuda.current_device() != self.device.index:
                    # This check is a bit redundant if torch.device(selected_device_str) worked,
                    # but good for an explicit log if a specific CUDA device isn't the one torch ends up using.
                    # More robust check would be to try a small tensor operation on that device.
                    # For now, we assume torch.device handles the validation.
                    pass # self.logger.info(f"Successfully set device to {selected_device_str}")
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
            print(f" dataloader size, Train: {len(train_loader)} | Validation: {len(val_loader)}")

            # Update progress for starting training
            update_progress_callback.emit({'status': f'Training LSTM model for {task["hyperparams"]["MAX_EPOCHS"]} epochs...'})

            # Run training with all necessary parameters
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
        feature_cols = task['data_loader_params']['feature_columns']
        target_col = task['data_loader_params']['target_column']
        train_val_split = float(task['data_loader_params'].get('train_val_split', 0.7))
        num_workers = int(task['hyperparams'].get('NUM_WORKERS', 4))
        seed = int(task['hyperparams'].get('SEED', 2000))
        
        training_method = task['hyperparams'].get('TRAINING_METHOD', 'Sequence-to-Sequence') # Default if not present
        model_type = task['hyperparams'].get('MODEL_TYPE', 'LSTM') # Default if not present

        self.logger.info(f"Selected Training Method: {training_method}, Model Type: {model_type}")

        if training_method == 'Whole Sequence' and model_type in ['LSTM', 'GRU']: # Check for RNN types
            self.logger.info("Using concatenated whole sequence loader for RNN.")
            train_loader, val_loader = self.data_loader_service.create_concatenated_whole_sequence_loaders(
                folder_path=self.job_manager.get_train_folder(),
                feature_cols=feature_cols,
                target_col=target_col,
                num_workers=num_workers,
                train_split=train_val_split,
                seed=seed
            )
        else: # Default to lookback-based sequence loading
            if training_method == 'Whole Sequence' and model_type not in ['LSTM', 'GRU']:
                 self.logger.info(f"Training method is 'Whole Sequence' but model type is {model_type}. Using standard sequence loader (this path is typically for FNNs with whole_sequence_fnn_data_handler or similar).")
            
            lookback = int(task['data_loader_params'].get('lookback', 50)) # Default lookback
            user_batch_size = int(task['data_loader_params'].get('batch_size', 32))
            
            batch_training_enabled = task['hyperparams'].get('BATCH_TRAINING', True)
            use_full_train_batch_flag = not batch_training_enabled

            self.logger.info(f"Using standard sequence loader. Batch training enabled: {batch_training_enabled}, User batch size: {user_batch_size}, Use full train batch flag: {use_full_train_batch_flag}, Lookback: {lookback}")
            
            train_loader, val_loader = self.data_loader_service.create_data_loaders(
                folder_path=self.job_manager.get_train_folder(),
                training_method=training_method,
                lookback=lookback,
                feature_cols=feature_cols,
                target_col=target_col,
                batch_size=user_batch_size,
                num_workers=num_workers,
                # use_full_train_batch=use_full_train_batch_flag, # Removed as it's not an accepted arg by production DataLoaderService
                train_split=train_val_split,
                seed=seed
            )

        return train_loader, val_loader

    def run_training(self, task, update_progress_callback, train_loader, val_loader, device):
        """Run the training process for a single task."""
        try:
            self.logger.info("Starting training loop")
            hyperparams = self.convert_hyperparams(task['hyperparams']) # This ensures BATCH_SIZE is int if it exists
            model = task['model'].to(device)
            
            # Ensure BATCH_SIZE from hyperparams (which might be the string from QLineEdit) is correctly converted and available
            # The actual batch size used by train_loader is now determined by DataLoaderService based on use_full_train_batch_flag
            # However, other parts of the code might still refer to hyperparams['BATCH_SIZE']
            # For logging or other purposes, ensure it's an int.
            # The convert_hyperparams method already handles BATCH_SIZE if it's a direct hyperparam.
            # If BATCH_SIZE is under data_loader_params, it's handled in create_data_loaders above.
            
            max_epochs = hyperparams['MAX_EPOCHS']
            valid_freq = hyperparams['ValidFrequency']
            valid_patience = hyperparams['VALID_PATIENCE']
            #patience_threshold = int(valid_patience * 0.5) 
            current_lr = hyperparams['INITIAL_LR']
            lr_drop_period = hyperparams['LR_DROP_PERIOD']
            lr_drop_factor = hyperparams['LR_DROP_FACTOR']
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

            # Initialize CSV logging
            csv_log_file = task['csv_log_file']
            with open(csv_log_file, 'w') as f:
                f.write("epoch,train_loss,val_loss,learning_rate\n")  # CSV header

            # Training loop
            for epoch in range(1, max_epochs + 1):
                if self.stop_requested:  # Ensure thread safety here
                    self.logger.info("Training stopped by user")
                    print("Stopping training...")
                    break

                # Initialize hidden states for training phase
                # Use the actual batch size from the train_loader
                actual_train_batch_size = train_loader.batch_size
                if actual_train_batch_size is None:
                    self.logger.warning(f"train_loader.batch_size is None. Falling back to hyperparams BATCH_SIZE ({hyperparams.get('BATCH_SIZE', 'N/A')}) for training hidden state init.")
                    actual_train_batch_size = int(hyperparams.get('BATCH_SIZE', 32)) # Default if all else fails

                self.logger.info(f"Initializing training hidden state with batch size: {actual_train_batch_size}")
                h_s = torch.zeros(model.num_layers, actual_train_batch_size, model.hidden_units).to(device)
                h_c = torch.zeros(model.num_layers, actual_train_batch_size, model.hidden_units).to(device)

                # Measure time for the training loop
                epoch_start_time = time.time()

                # Train the model for one epoch
                model_type = task.get('model_metadata', {}).get('model_type', task.get('hyperparams', {}).get('MODEL_TYPE', 'LSTM')) # Get model_type
                avg_batch_time, train_loss = self.training_service.train_epoch(model, model_type, train_loader, optimizer, h_s, h_c, epoch, device, self.stop_requested, task)

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
                    # Initialize hidden states for validation phase
                    # Use the actual batch size from the val_loader
                    actual_val_batch_size = val_loader.batch_size
                    if actual_val_batch_size is None:
                        self.logger.warning(f"val_loader.batch_size is None. Falling back to hyperparams BATCH_SIZE ({hyperparams.get('BATCH_SIZE', 'N/A')}) for validation hidden state init.")
                        actual_val_batch_size = int(hyperparams.get('BATCH_SIZE', 32)) # Default if all else fails
                    
                    self.logger.info(f"Initializing validation hidden state with batch size: {actual_val_batch_size}")
                    h_s_val = torch.zeros(model.num_layers, actual_val_batch_size, model.hidden_units).to(device)
                    h_c_val = torch.zeros(model.num_layers, actual_val_batch_size, model.hidden_units).to(device)

                    model_type = task.get('model_metadata', {}).get('model_type', task.get('hyperparams', {}).get('MODEL_TYPE', 'LSTM')) # Get model_type
                    val_loss = self.training_service.validate_epoch(model, model_type, val_loader, h_s_val, h_c_val, epoch, device, self.stop_requested, task)

                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    delta_t_epoch = (current_time - last_validation_time) / valid_freq
                    last_validation_time = current_time

                    current_lr = optimizer.param_groups[0]['lr']
                    
                    if val_loss < best_validation_loss:
                        print(f"Epoch: {epoch}, Validation loss improved from {best_validation_loss:.6f} to {val_loss:.6f}. Saving model...")
                        best_validation_loss = val_loss
                        self.save_model(task)
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    # Determine target variable specifics for error reporting
                    target_column = task['data_loader_params']['target_column']
                    error_unit_label = "RMS Error"  # Default
                    multiplier = 1.0

                    if "voltage" in target_column.lower():
                        error_unit_label = "RMS Error [mV]"
                        multiplier = 1000.0
                    elif "soc" in target_column.lower():
                        error_unit_label = "RMS Error [% SOC]"
                        multiplier = 100.0
                    elif "soe" in target_column.lower(): # Added SOE
                        error_unit_label = "RMS Error [% SOE]"
                        multiplier = 100.0
                    elif "sop" in target_column.lower(): # Added SOP
                        error_unit_label = "RMS Error [% SOP]"
                        multiplier = 100.0
                    elif "temperature" in target_column.lower() or "temp" in target_column.lower():
                        error_unit_label = "RMS Error [Deg C]"
                        multiplier = 1.0
                    
                    # Calculate scaled RMSE values, handling potential NaN or None for losses
                    train_rmse_scaled = float('nan')
                    if train_loss is not None and not math.isnan(train_loss):
                        train_rmse_scaled = math.sqrt(max(0, train_loss)) * multiplier

                    val_rmse_scaled = float('nan')
                    if val_loss is not None and not math.isnan(val_loss):
                        val_rmse_scaled = math.sqrt(max(0, val_loss)) * multiplier
                    
                    best_val_rmse_scaled = float('inf') # Default if best_validation_loss is inf
                    if best_validation_loss != float('inf') and not math.isnan(best_validation_loss):
                         best_val_rmse_scaled = math.sqrt(max(0, best_validation_loss)) * multiplier

                    self.logger.info(f"Epoch {epoch} | Train Loss (MSE): {train_loss:.6f} | Val Loss (MSE): {val_loss:.6f} | {error_unit_label} (Train): {train_rmse_scaled:.4f} | {error_unit_label} (Val): {val_rmse_scaled:.4f} | LR: {current_lr} | Epoch Time: {formatted_epoch_time} | Best Val Loss (MSE): {best_validation_loss:.6f} | Best {error_unit_label}: {best_val_rmse_scaled:.4f} | Patience Counter: {patience_counter}")
                    progress_data = {
                        'epoch': epoch,
                        'train_loss': train_loss,  # Original MSE for logging or other non-display purposes
                        'val_loss': val_loss,      # Original MSE
                        'train_rmse_scaled': train_rmse_scaled,
                        'val_rmse_scaled': val_rmse_scaled,
                        'error_unit_label': error_unit_label,
                        'elapsed_time': elapsed_time,
                        'delta_t_epoch': formatted_epoch_time,
                        'learning_rate': current_lr,
                        'best_val_loss': best_validation_loss, # Original MSE
                        'best_val_rmse_scaled': best_val_rmse_scaled,
                        'patience_counter': patience_counter,
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
                #print(f"Checking log files for the task: {task['task_id']}: task['csv_log_file'], task['db_log_file']")

                # Save log data to CSV and SQLite
                # self.log_to_csv(task, epoch, train_loss, val_loss, elapsed_time, current_lr, best_validation_loss, delta_t_epoch)
                model_memory_usage = torch.cuda.memory_allocated() if torch.cuda.is_available() else sys.getsizeof(model)
                model_memory_usage_mb = model_memory_usage / (1024 * 1024)  # Convert to MB
                
                scheduler.step()

                new_lr = optimizer.param_groups[0]['lr']
                if new_lr != current_lr:
                    print(f"Learning rate changed from {current_lr:.8f} to {new_lr:.8f} at epoch {epoch}")
                    logging.info(f"Learning rate changed from {current_lr:.8f} to {new_lr:.8f} at epoch {epoch}")
                    current_lr = new_lr

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

            # Final save and cleanup
            self.save_model(task)
            
            # Log final summary
            with open(os.path.join(task['model_dir'], 'training_summary.txt'), 'w') as f:
                f.write(f"Training completed\n")
                f.write(f"Best validation loss: {best_validation_loss:.6f}\n")
                f.write(f"Final learning rate: {optimizer.param_groups[0]['lr']:.8f}\n")
                f.write(f"Stopped at epoch: {epoch}/{max_epochs}\n")

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
        
        # Update scheduler parameter names to match task info
        if hyperparams['SCHEDULER_TYPE'] == 'StepLR':
            hyperparams['LR_DROP_PERIOD'] = int(hyperparams['LR_PERIOD'])  # Map LR_PERIOD to LR_DROP_PERIOD
            hyperparams['LR_DROP_FACTOR'] = float(hyperparams['LR_PARAM'])  # Map LR_PARAM to LR_DROP_FACTOR
        else:
            hyperparams['PLATEAU_PATIENCE'] = int(hyperparams['PLATEAU_PATIENCE'])
            hyperparams['PLATEAU_FACTOR'] = float(hyperparams['PLATEAU_FACTOR'])
        hyperparams['VALID_PATIENCE'] = int(hyperparams['VALID_PATIENCE'])
        hyperparams['ValidFrequency'] = int(hyperparams['ValidFrequency'])
        hyperparams['LOOKBACK'] = int(hyperparams['LOOKBACK'])
        hyperparams['REPETITIONS'] = int(hyperparams['REPETITIONS'])
        return hyperparams

    def save_model(self, task):
        """Save the trained model to disk in both internal and portable formats."""
        try:
            model_path = task.get('model_path', None)
            if model_path is None:
                self.logger.error("Model path not found in task.")
                raise ValueError("Model path not found in task.")

            model = task['model']
            if model is None:
                self.logger.error("No model instance found in task.")
                raise ValueError("No model instance found in task.")

            # Save full model for internal use (current workflow)
            torch.save(model, model_path)
            print(f"Full model saved")

            # Save portable version
            task_dir = os.path.dirname(model_path)
            export_path = os.path.join(task_dir, 'model_export.pt')
            
            # Get model definition code
            model_def = """
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_units, num_layers, output_size=1, device='cpu'):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.output_size = output_size
        self.device = device

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_units,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_units, output_size)
    
    def forward(self, x, h_s=None, h_c=None):
        # Initialize hidden state and cell state if not provided
        if h_s is None or h_c is None:
            h_s = torch.zeros(self.num_layers, x.size(0), self.hidden_units).to(self.device)
            h_c = torch.zeros(self.num_layers, x.size(0), self.hidden_units).to(self.device)
        
        # Forward pass through LSTM
        out, (h_s, h_c) = self.lstm(x, (h_s, h_c))
        
        # Get output from last time step
        out = self.fc(out[:, -1, :])
        return out, (h_s, h_c)
"""

            # Create export dictionary with all necessary information
            export_dict = {
                'state_dict': model.state_dict(),
                'model_definition': model_def,
                'model_metadata': task['model_metadata'],
                'hyperparams': {
                    'input_size': task['hyperparams']['INPUT_SIZE'],
                    'hidden_size': task['hyperparams']['HIDDEN_UNITS'],
                    'num_layers': task['hyperparams']['LAYERS'],
                    'output_size': task['hyperparams']['OUTPUT_SIZE']
                },
                'data_config': {
                    'feature_columns': task['data_loader_params']['feature_columns'],
                    'target_column': task['data_loader_params']['target_column'],
                    'lookback': task['data_loader_params']['lookback']
                },
                'model_type': task['model_metadata']['model_type'],
                'export_timestamp': time.strftime("%Y%m%d-%H%M%S")
            }

            # Save portable version
            torch.save(export_dict, export_path)
            print(f"Portable model saved!")

            # Create a README file with multiple loading options
            readme_path = os.path.join(task_dir, 'MODEL_LOADING_INSTRUCTIONS.md')
            readme_content = f"""# Model Loading Instructions

## Model Details
- Model Type: {task['model_metadata']['model_type']}
- Input Size: {task['hyperparams']['INPUT_SIZE']}
- Hidden Units: {task['hyperparams']['HIDDEN_UNITS']}
- Layers: {task['hyperparams']['LAYERS']}
- Output Size: {task['hyperparams']['OUTPUT_SIZE']}
- Lookback: {task['data_loader_params']['lookback']}

## Feature Configuration
- Input Features: {', '.join(task['data_loader_params']['feature_columns'])}
- Target Variable: {task['data_loader_params']['target_column']}

## Loading Options

### Option 1: Using VEstim Environment
```python
import torch
from vestim.services.model_training.src.LSTM_model_service_test import LSTMModelService

# Load the exported model
checkpoint = torch.load('model_export.pt')

# Create model instance
model_service = LSTMModelService()
model = model_service.create_model(
    input_size=checkpoint['hyperparams']['input_size'],
    hidden_size=checkpoint['hyperparams']['hidden_size'],
    num_layers=checkpoint['hyperparams']['num_layers'],
    output_size=checkpoint['hyperparams']['output_size']
)

# Load state dict
model.load_state_dict(checkpoint['state_dict'])
model.eval()  # Set to evaluation mode
```

### Option 2: Standalone Usage (No VEstim Required)
```python
import torch
import torch.nn as nn

# Load the checkpoint
checkpoint = torch.load('model_export.pt')

# Execute the model definition code (included in the checkpoint)
exec(checkpoint['model_definition'])

# Create model instance
model = LSTMModel(
    input_size=checkpoint['hyperparams']['input_size'],
    hidden_units=checkpoint['hyperparams']['hidden_size'],
    num_layers=checkpoint['hyperparams']['num_layers'],
    output_size=checkpoint['hyperparams']['output_size']
)

# Load state dict
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Example usage:
def predict(model, input_data):
    with torch.no_grad():
        output, _ = model(input_data)
    return output
```

## Input Data Format
- Input shape should be: (batch_size, lookback, input_size)
- Features should be in order: {', '.join(task['data_loader_params']['feature_columns'])}
- All inputs should be normalized using the same scaling as training data

## Example Preprocessing
```python
import numpy as np

def preprocess_data(data, lookback={task['data_loader_params']['lookback']}):
    # Ensure data is normalized using the same scaling as training
    # Create sequences of length 'lookback'
    sequences = []
    for i in range(len(data) - lookback + 1):
        sequences.append(data[i:(i + lookback)])
    return torch.FloatTensor(np.array(sequences))
```

## Making Predictions
```python
# Example prediction
input_sequence = preprocess_data(your_data)  # Shape: (1, lookback, input_size)
with torch.no_grad():
    prediction, _ = model(input_sequence)
```
"""
            
            with open(readme_path, 'w') as f:
                f.write(readme_content)
            #print(f"Loading instructions saved to {readme_path}")

        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise

    def stop_task(self):
        self.stop_requested = True  # Set the flag to request a stop
        if self.training_thread and self.training_thread.isRunning():  # Use isRunning() instead of is_alive()
            print("Waiting for the training thread to finish before saving the model...")
            self.training_thread.quit()  # Gracefully stop the thread
            self.training_thread.wait(7000)  # Wait for the thread to finish cleanly
            print("Training thread has finished. Proceeding to save the model.")
            self.logger.info("Training thread has finished. Proceeding to save the model.")
