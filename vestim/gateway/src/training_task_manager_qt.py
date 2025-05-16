import time, os, sys
import csv
import sqlite3
import torch
from PyQt5.QtCore import QThread, pyqtSignal
from vestim.gateway.src.job_manager_qt import JobManager
from vestim.gateway.src.training_setup_manager_qt import VEstimTrainingSetupManager
from vestim.services.model_training.src.data_loader_service import DataLoaderService
from vestim.services.model_training.src.training_task_service import TrainingTaskService
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

    def log_to_sqlite(self, task, epoch, train_loss, val_loss, best_val_loss, elapsed_time, avg_batch_time, early_stopping, model_memory_usage):
        """Log epoch-level data to a SQLite database."""
        sqlite_db_file = task['db_log_file']
        conn = sqlite3.connect(sqlite_db_file)
        cursor = conn.cursor()

        cursor.execute('''INSERT INTO task_logs (task_id, epoch, train_loss, val_loss, elapsed_time, avg_batch_time, learning_rate, 
                        best_val_loss, num_learnable_params, batch_size, lookback, max_epochs, early_stopping, model_memory_usage, device)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (task['task_id'], epoch, train_loss, val_loss, elapsed_time, avg_batch_time, task['hyperparams']['INITIAL_LR'], best_val_loss,
                        task['hyperparams']['NUM_LEARNABLE_PARAMS'], task['hyperparams']['BATCH_SIZE'], task['hyperparams']['LOOKBACK'],task['hyperparams']['MAX_EPOCHS'],
                        early_stopping, model_memory_usage, self.device.type))

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
        dl_params = task.get('data_loader_params', {})
        
        # Essential parameters from the task
        training_method = dl_params.get('training_method', 'SequenceRNN') # Default to existing behavior
        feature_cols = dl_params.get('feature_cols') # Should be set by VEstimTrainingSetupManager
        target_col = dl_params.get('target_col')   # Should be set by VEstimTrainingSetupManager
        
        if feature_cols is None or target_col is None:
            # Fallback to old hardcoded values if not provided, with a warning.
            # Ideally, VEstimTrainingSetupManager should always provide these.
            self.logger.warning("feature_cols or target_col not found in data_loader_params. Falling back to defaults.")
            self.logger.warning("This may lead to errors if the data doesn't match ['SOC', 'Current', 'Temp'] and 'Voltage'.")
            feature_cols = dl_params.get('feature_cols', ['SOC', 'Current', 'Temp']) # Example fallback
            target_col = dl_params.get('target_col', 'Voltage') # Example fallback

        lookback = dl_params.get('lookback', 50) # Default lookback if not specified, adjust as needed
        batch_size = dl_params.get('batch_size', 32) # Default batch_size
        concatenate_raw_data = dl_params.get('concatenate_raw_data', False) # Default for SequenceRNN
        
        num_workers = dl_params.get('num_workers', 4) # Or a global default
        train_split = dl_params.get('train_split', 0.7)
        seed = dl_params.get('seed', None)


        self.logger.info(f"Creating data loaders with method: {training_method}, features: {feature_cols}, target: {target_col}, lookback: {lookback}")
        
        train_loader, val_loader = self.data_loader_service.create_data_loaders(
            folder_path=self.job_manager.get_train_folder(),
            training_method=training_method,
            feature_cols=feature_cols,
            target_col=target_col,
            batch_size=batch_size,
            num_workers=num_workers,
            lookback=lookback if training_method == "SequenceRNN" else None, # Pass lookback only if relevant
            concatenate_raw_data=concatenate_raw_data if training_method == "SequenceRNN" else False,
            train_split=train_split,
            seed=seed
        )

        return train_loader, val_loader

    def run_training(self, task, update_progress_callback, train_loader, val_loader, device):
        """Run the training process for a single task."""
        try:
            self.logger.info("Starting training loop")
            hyperparams = self.convert_hyperparams(task['hyperparams']) # Ensures types are correct
            model = task['model']
            if model is None:
                self.logger.error(f"Task {task.get('task_id')} has a None model. Skipping training.")
                update_progress_callback.emit({'task_error': f"Model for task {task.get('task_id')} is None."})
                return # Skip this task if model is None (e.g. FNN/GRU service not ready)
            
            model = model.to(device)
            
            model_type = hyperparams.get('MODEL_TYPE', 'LSTM').upper() # Get model type, default LSTM
            self.logger.info(f"Preparing to train model of type: {model_type}")

            max_epochs = hyperparams['MAX_EPOCHS']
            valid_freq = hyperparams['ValidFrequency'] # This was from old params, ensure it's 'VALID_FREQUENCY' or similar from new
            valid_patience = hyperparams['VALID_PATIENCE']
            lr_drop_period = hyperparams['LR_DROP_PERIOD']
            lr_drop_factor = hyperparams.get('LR_DROP_FACTOR', 0.1) # Default if not in hyperparams
            weight_decay = hyperparams.get('WEIGHT_DECAY', 0.0) # Default if not in hyperparams
            batch_size = hyperparams['BATCH_SIZE'] # Needed for hidden state initialization

            best_validation_loss = float('inf')
            patience_counter = 0
            start_time = time.time()
            last_validation_time = start_time
            early_stopping = False

            optimizer = self.training_service.get_optimizer(model, lr=hyperparams['INITIAL_LR'], weight_decay=weight_decay)
            scheduler = self.training_service.get_scheduler(optimizer, step_size=lr_drop_period, gamma=lr_drop_factor)

            def format_time(seconds):
                minutes = seconds // 60
                seconds = seconds % 60
                return f"{int(minutes)}:{int(seconds):02d}"

            for epoch in range(1, max_epochs + 1):
                if self.stop_requested:
                    self.logger.info("Training stopped by user")
                    break

                h_s_initial, h_c_initial = None, None
                if model_type in ["LSTM", "GRU"]: # Check if model is RNN type
                    # These should come from the model's actual attributes or build_params
                    # For LSTM: model.num_layers, model.hidden_units
                    # For GRU: model.num_layers, model.hidden_units
                    # For FNN: these are not applicable.
                    # Assuming these are stored in hyperparams or accessible via model attributes
                    # For safety, get from hyperparams if available, else try model attributes
                    num_layers = hyperparams.get('LAYERS', getattr(model, 'num_layers', 1)) # Default to 1 if not found
                    hidden_units = hyperparams.get('HIDDEN_UNITS', getattr(model, 'hidden_units', 0)) # Default to 0 if not found
                    
                    if hidden_units > 0 : # Only initialize if hidden_units is meaningful
                        h_s_initial = torch.zeros(num_layers, batch_size, hidden_units, device=device)
                        if model_type == "LSTM":
                            h_c_initial = torch.zeros(num_layers, batch_size, hidden_units, device=device)
                    else:
                        self.logger.warning(f"Hidden units is 0 or not found for {model_type}. Hidden states not initialized.")
                
                epoch_start_time = time.time()

                avg_batch_time, train_loss = self.training_service.train_epoch(
                    model, model_type, train_loader, optimizer,
                    h_s_initial, h_c_initial,
                    epoch, device, self.stop_requested, task
                )

                epoch_end_time = time.time()
                epoch_duration = epoch_end_time - epoch_start_time
                formatted_epoch_time = format_time(epoch_duration)

                if self.stop_requested:
                    self.logger.info("Training stopped by user after training epoch.")
                    break
                
                val_loss = float('nan') # Default val_loss if not validated this epoch
                if epoch == 1 or epoch % valid_freq == 0 or epoch == max_epochs:
                    # Re-initialize hidden states for validation if RNN
                    h_s_val_initial, h_c_val_initial = None, None
                    if model_type in ["LSTM", "GRU"]:
                        num_layers = hyperparams.get('LAYERS', getattr(model, 'num_layers', 1))
                        hidden_units = hyperparams.get('HIDDEN_UNITS', getattr(model, 'hidden_units', 0))
                        if hidden_units > 0:
                            h_s_val_initial = torch.zeros(num_layers, batch_size, hidden_units, device=device)
                            if model_type == "LSTM":
                                h_c_val_initial = torch.zeros(num_layers, batch_size, hidden_units, device=device)
                    
                    val_loss = self.training_service.validate_epoch(
                        model, model_type, val_loader,
                        h_s_val_initial, h_c_val_initial,
                        epoch, device, self.stop_requested, task
                    )
                    self.logger.info(f"Epoch {epoch} | Train Loss: {train_loss} | Val Loss: {val_loss} | Epoch Time: {formatted_epoch_time}")

                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    delta_t_epoch = (current_time - last_validation_time) / valid_freq
                    last_validation_time = current_time

                    current_lr = optimizer.param_groups[0]['lr']

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

                    if val_loss < best_validation_loss:
                        best_validation_loss = val_loss
                        patience_counter = 0
                        self.save_model(task)
                    else:
                        patience_counter += 1

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
                            early_stopping=early_stopping,  # Mark the early stopping in the log
                            model_memory_usage=round(model_memory_usage_mb, 3),  # Memory in MB, rounded to 2 decimal places
                        )
                        break

                # Log data to CSV and SQLite after each epoch (whether validated or not)
                print(f"Checking log files for the task: {task['task_id']}: task['csv_log_file'], task['db_log_file']")

                # Save log data to CSV and SQLite
                # self.log_to_csv(task, epoch, train_loss, val_loss, elapsed_time, current_lr, best_validation_loss, delta_t_epoch)
                model_memory_usage = torch.cuda.memory_allocated() if torch.cuda.is_available() else sys.getsizeof(model)
                model_memory_usage_mb = model_memory_usage / (1024 * 1024)  # Convert to MB
                self.log_to_sqlite(
                    task=task,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    best_val_loss=best_validation_loss,
                    elapsed_time=elapsed_time,
                    avg_batch_time=avg_batch_time,
                    early_stopping=early_stopping,
                    model_memory_usage=round(model_memory_usage_mb, 3),  # Memory in MB, rounded to 2 decimal places
                )

                scheduler.step()

            if self.stop_requested:
                print("Training was stopped early. Saving Model...")
                self.logger.info("Training was stopped early. Saving Model...")
                self.save_model(task)

            update_progress_callback.emit({'task_completed': True})
            self.logger.info("Training task completed")

        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            update_progress_callback.emit({'task_error': str(e)})


    def convert_hyperparams(self, hyperparams_orig: dict) -> dict:
        """
        Converts hyperparameters to their correct types based on MODEL_TYPE.
        Ensures that only relevant keys for the specific model type are present
        or correctly formatted for downstream services.
        The input `hyperparams_orig` is expected to be `task['hyperparams']`
        which is populated by VEstimTrainingSetupManager.
        """
        hyperparams = hyperparams_orig.copy()  # Work on a copy
        model_type = hyperparams.get('MODEL_TYPE', 'LSTM').upper() # Default to LSTM if not specified
        self.logger.info(f"Starting hyperparameter conversion for MODEL_TYPE: {model_type}.")
        self.logger.debug(f"Original hyperparams for conversion: {json.dumps(hyperparams_orig, indent=2)}")

        # --- Common Parameter Conversions (ensure correct types) ---
        # These parameters are expected to be present in task['hyperparams']
        # and should have been processed to some extent by VEstimTrainingSetupManager.
        
        # BATCH_SIZE is critical and should be an int.
        if 'BATCH_SIZE' in hyperparams:
            hyperparams['BATCH_SIZE'] = int(hyperparams['BATCH_SIZE'])
        else:
            self.logger.error("'BATCH_SIZE' not found in hyperparams during conversion.")
            raise KeyError("'BATCH_SIZE' is required in hyperparams.")

        hyperparams['MAX_EPOCHS'] = int(hyperparams.get('MAX_EPOCHS', 100))
        hyperparams['INITIAL_LR'] = float(hyperparams.get('INITIAL_LR', 0.001))
        hyperparams['LR_DROP_PERIOD'] = int(hyperparams.get('LR_DROP_PERIOD', 10))
        hyperparams['LR_DROP_FACTOR'] = float(hyperparams.get('LR_DROP_FACTOR', 0.1))
        hyperparams['VALID_PATIENCE'] = int(hyperparams.get('VALID_PATIENCE', 5))
        
        # Consolidate VALID_FREQUENCY handling
        valid_freq_val = hyperparams.get('VALID_FREQUENCY', hyperparams.get('ValidFrequency'))
        if valid_freq_val is not None:
            hyperparams['VALID_FREQUENCY'] = int(valid_freq_val)
            if 'ValidFrequency' in hyperparams and 'VALID_FREQUENCY' in hyperparams and 'ValidFrequency' != 'VALID_FREQUENCY':
                del hyperparams['ValidFrequency'] # Remove old key if new one is used
        else:
            hyperparams['VALID_FREQUENCY'] = 1 # Default if neither key is found
            self.logger.warning("'VALID_FREQUENCY' or 'ValidFrequency' not found, defaulting to 1.")

        hyperparams['REPETITIONS'] = int(hyperparams.get('REPETITIONS', 1)) # Should be set by TrainingSetupManager
        hyperparams['WEIGHT_DECAY'] = float(hyperparams.get('WEIGHT_DECAY', 0.0))
        
        # NUM_LEARNABLE_PARAMS is added by TrainingSetupManager
        if 'NUM_LEARNABLE_PARAMS' in hyperparams:
             hyperparams['NUM_LEARNABLE_PARAMS'] = int(hyperparams['NUM_LEARNABLE_PARAMS'])
        else:
            self.logger.warning("'NUM_LEARNABLE_PARAMS' not found in hyperparams.")
            hyperparams['NUM_LEARNABLE_PARAMS'] = 0 # Default to 0 if not found


        # --- Model-Specific Conversions and Cleanup ---
        if model_type in ['LSTM', 'GRU']:
            self.logger.info(f"Processing {model_type}-specific parameters.")
            if 'LAYERS' not in hyperparams: raise KeyError(f"LAYERS is required for {model_type} but not found. Keys: {list(hyperparams.keys())}")
            hyperparams['LAYERS'] = int(hyperparams['LAYERS'])
            
            if 'HIDDEN_UNITS' not in hyperparams: raise KeyError(f"HIDDEN_UNITS is required for {model_type} but not found. Keys: {list(hyperparams.keys())}")
            hyperparams['HIDDEN_UNITS'] = int(hyperparams['HIDDEN_UNITS'])
            
            # DROPOUT_PROB for RNNs (should be float, set by TrainingSetupManager from model_build_params)
            hyperparams['DROPOUT_PROB'] = float(hyperparams.get('DROPOUT_PROB', 0.0))
            
            # LOOKBACK is relevant for RNNs, especially with SequenceRNN training method
            if 'LOOKBACK' in hyperparams and hyperparams['LOOKBACK'] is not None:
                 hyperparams['LOOKBACK'] = int(hyperparams['LOOKBACK'])
            elif 'LOOKBACK' not in hyperparams and hyperparams.get('TRAINING_METHOD') == 'SequenceRNN':
                 self.logger.warning(f"LOOKBACK not found for {model_type} with SequenceRNN training method.")
            
            # Remove FNN specific keys if they were somehow included
            fnn_keys_to_remove = ['HIDDEN_LAYER_SIZES', 'FNN_HIDDEN_LAYERS', 'FNN_DROPOUT_PROB']
            for key_to_remove in fnn_keys_to_remove:
                if key_to_remove in hyperparams:
                    self.logger.debug(f"For {model_type}, removing FNN key '{key_to_remove}'.")
                    del hyperparams[key_to_remove]

        elif model_type == 'FNN':
            self.logger.info("Processing FNN-specific parameters.")
            # VEstimTrainingSetupManager should have placed 'HIDDEN_LAYER_SIZES' (list of int)
            # and 'DROPOUT_PROB' (float) into task['hyperparams'] from model_build_params.
            if 'HIDDEN_LAYER_SIZES' not in hyperparams or \
               not isinstance(hyperparams['HIDDEN_LAYER_SIZES'], list) or \
               not all(isinstance(x, int) for x in hyperparams['HIDDEN_LAYER_SIZES']):
                self.logger.error(f"FNN model expects 'HIDDEN_LAYER_SIZES' as a list of int. Got: {hyperparams.get('HIDDEN_LAYER_SIZES')}")
                raise ValueError("FNN 'HIDDEN_LAYER_SIZES' is missing or malformed in task hyperparams.")

            # DROPOUT_PROB for FNN (should be float, set by TrainingSetupManager from model_build_params)
            if 'DROPOUT_PROB' not in hyperparams or not isinstance(hyperparams['DROPOUT_PROB'], float):
                self.logger.error(f"FNN model expects 'DROPOUT_PROB' as a float. Got: {hyperparams.get('DROPOUT_PROB')}")
                hyperparams['DROPOUT_PROB'] = float(hyperparams.get('DROPOUT_PROB', 0.0)) # Ensure float

            # Explicitly remove keys not relevant for FNN to prevent any downstream misuse
            rnn_and_gru_keys_to_remove = [
                'LOOKBACK', 'LAYERS', 'HIDDEN_UNITS', # General RNN
                'FNN_HIDDEN_LAYERS', 'FNN_DROPOUT_PROB', # Original string versions from GUI, should not be here
                'GRU_LAYERS', 'GRU_HIDDEN_UNITS', 'GRU_DROPOUT_PROB' # GRU specific
            ]
            for key in rnn_and_gru_keys_to_remove:
                if key in hyperparams:
                    self.logger.info(f"For FNN model, removing irrelevant key '{key}' from hyperparams during conversion.")
                    del hyperparams[key]
        else:
            self.logger.error(f"Unsupported MODEL_TYPE '{model_type}' encountered in convert_hyperparams.")
            raise ValueError(f"Unsupported MODEL_TYPE: {model_type}")

        self.logger.info(f"Successfully converted hyperparameters for {model_type}.")
        self.logger.debug(f"Final converted hyperparams: {json.dumps(hyperparams, indent=2)}")
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
