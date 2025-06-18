import time, os, sys, math, json # Added json
import csv
import sqlite3
import torch
import logging
import traceback
from vestim.backend.src.services.training_service import TrainingService
from vestim.backend.src.managers.training_setup_manager_qt import VEstimTrainingSetupManager

def format_time(seconds):
    """Convert seconds to mm:ss format."""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

class TrainingTaskManager:
    """
    Manages the execution of a training task in a background process.
    This class is UI-agnostic and designed to be called by the JobManager.
    """
    def __init__(self):
        """
        Initializes the TrainingTaskManager.
        Note: Heavy initialization should be done inside the process_task_in_background        method to ensure it runs in the separate process.
        """
        self.logger = logging.getLogger(__name__)

    def process_task_in_background(self, status_queue, task_info):
        """
        This method is the entry point for the background process started by the JobManager.
        It handles the entire lifecycle of a training task.
        Based on the working implementation from training_task_manager_qt.py.bak
        """
        job_id = task_info.get('job_id')
        stop_flag = task_info.get('stop_flag')
        
        try:
            self.logger.info(f"[{job_id}] Background training process started.")
            status_queue.put((job_id, 'initializing', {"message": "Setting up training environment..."}))

            # --- Initialization within the process ---
            from vestim.backend.src.managers.training_setup_manager_qt import VEstimTrainingSetupManager
            from vestim.backend.src.services.model_training.src.data_loader_service import DataLoaderService
            from vestim.backend.src.services.model_training.src.training_task_service import TrainingTaskService
            
            training_setup_manager = VEstimTrainingSetupManager()
            data_loader_service = DataLoaderService()
            training_service = TrainingTaskService()
            
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

            # Process the task using the working logic from .bak file
            self.process_task_with_api_updates(task_info, status_queue, device, 
                                             data_loader_service, training_service, stop_flag)

        except Exception as e:
            error_msg = f"[{job_id}] Error during training task: {e}"
            self.logger.error(error_msg, exc_info=True)
            status_queue.put((job_id, 'error', {                "message": str(e),
                "error": error_msg,
                "traceback": traceback.format_exc()
            }))

    def process_task_with_api_updates(self, task_info, status_queue, device, data_loader_service, training_service, stop_flag):
        """
        Process a single training task adapted from the .bak file logic.
        Uses status_queue instead of progress callbacks for API integration.
        """
        job_id = task_info.get('job_id')
        task_id = task_info.get('task_id')
        
        try:
            # Extract hyperparameters from task_info
            hyperparams = task_info.get('hyperparams', {})
            if not hyperparams:
                raise ValueError("No hyperparameters found in task_info")
                
            self.logger.info(f"Starting task {task_id} with hyperparams: {hyperparams}")
            
            # Setup job logging (SQLite and CSV)
            status_queue.put((job_id, 'setup_logging', {"message": "Setting up logging..."}))
            self.setup_job_logging_api(task_info, status_queue)
            
            # Create data loaders
            status_queue.put((job_id, 'creating_dataloader', {"message": "Creating data loaders..."}))
            train_loader, val_loader = self.create_data_loaders_api(task_info, data_loader_service)
            
            status_queue.put((job_id, 'dataloader_created', {
                "message": f"Data loaders created. Train: {len(train_loader)}, Val: {len(val_loader)}",
                "train_batches": len(train_loader),
                "val_batches": len(val_loader)
            }))
            
            # Start training loop
            status_queue.put((job_id, 'training_started', {
                "message": f"Starting training for {hyperparams.get('MAX_EPOCHS', 10)} epochs...",
                "total_epochs": int(hyperparams.get('MAX_EPOCHS', 10))
            }))
            
            # Run the actual training
            self.run_training_api(task_info, status_queue, train_loader, val_loader, device, stop_flag, training_service)
            
        except Exception as e:
            self.logger.error(f"Error during task processing: {str(e)}", exc_info=True)
            status_queue.put((job_id, 'task_error', {"message": str(e), "traceback": traceback.format_exc()}))

    def create_data_loaders_api(self, task_info, data_loader_service):
        """Create data loaders based on task hyperparameters - adapted from .bak file"""
        hyperparams = task_info.get('hyperparams', {})
        
        # Extract data loader parameters from hyperparams
        feature_cols = hyperparams.get('FEATURE_COLUMNS', ['Current', 'Temp', 'Voltage'])
        target_col = hyperparams.get('TARGET_COLUMN', 'SOC')
        train_val_split = float(hyperparams.get('TRAIN_VAL_SPLIT', 0.8))
        training_method = hyperparams.get('TRAINING_METHOD', 'Sequence-to-Sequence')
        model_type = hyperparams.get('MODEL_TYPE', 'LSTM')
        lookback = int(hyperparams.get('LOOKBACK', 400))
        batch_size = int(hyperparams.get('BATCH_SIZE', 100))
        batch_training = hyperparams.get('BATCH_TRAINING', True)
        
        self.logger.info(f"Creating data loaders - Method: {training_method}, Model: {model_type}, Lookback: {lookback}, Batch size: {batch_size}")
        
        # Get the job folder to find training data
        job_folder = task_info.get('job_folder')
        if not job_folder:
            raise ValueError("No job_folder specified in task_info")
            
        # Assume training data is in job_folder (you may need to adjust this path)
        train_data_path = job_folder  # This may need adjustment based on your file structure
        
        # Create data loaders using the same logic as .bak file
        train_loader, val_loader = data_loader_service.create_data_loaders(
            folder_path=train_data_path,
            training_method=training_method,
            lookback=lookback,
            feature_cols=feature_cols, 
            target_col=target_col,
            batch_size=batch_size,
            num_workers=4,  # Default from .bak file
            train_split=train_val_split,
            seed=2000  # Default from .bak file
        )
        
        return train_loader, val_loader

    def run_training_api(self, task_info, status_queue, train_loader, val_loader, device, stop_flag, training_service):
        """Run training loop with detailed progress updates - adapted from .bak file"""
        job_id = task_info.get('job_id') 
        task_id = task_info.get('task_id')
        hyperparams = task_info.get('hyperparams', {})
        
        # Training parameters
        max_epochs = int(hyperparams.get('MAX_EPOCHS', 10))
        model_type = hyperparams.get('MODEL_TYPE', 'LSTM')
        
        # Initialize training history for persistence
        training_history = {
            'epoch_data': [],
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'best_epoch': 0,
            'best_val_loss': float('inf'),
            'status': 'training',
            'current_epoch': 0,
            'total_epochs': max_epochs
        }
        
        try:
            # This is a simplified training loop - you'll need to implement the full logic from .bak file
            # including model creation, optimizer setup, etc.
            
            for epoch in range(1, max_epochs + 1):
                # Check for stop signal
                if stop_flag and stop_flag.is_set():
                    training_history['status'] = 'stopped'
                    status_queue.put((job_id, 'training_stopped', {
                        "message": f"Training stopped at epoch {epoch}",
                        "training_history": training_history
                    }))
                    break
                
                # Simulate training epoch (replace with actual training logic)
                epoch_start_time = time.time()
                
                # TODO: Implement actual training epoch using training_service
                # train_loss = training_service.train_epoch(...)
                # val_loss = training_service.validate_epoch(...)
                
                # For now, simulate progress
                import math
                train_loss = 1.0 * math.exp(-epoch / 10)  # Simulated decreasing loss
                val_loss = 1.1 * math.exp(-epoch / 8)    # Simulated decreasing loss
                
                epoch_end_time = time.time()
                epoch_duration = epoch_end_time - epoch_start_time
                
                # Update training history
                epoch_data = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'epoch_time': epoch_duration,
                    'timestamp': time.time()
                }
                
                training_history['epoch_data'].append(epoch_data)
                training_history['train_losses'].append(train_loss)
                training_history['val_losses'].append(val_loss)
                training_history['current_epoch'] = epoch
                
                # Track best model
                if val_loss < training_history['best_val_loss']:
                    training_history['best_val_loss'] = val_loss
                    training_history['best_epoch'] = epoch
                
                # Send detailed progress update
                status_queue.put((job_id, 'training_progress', {
                    "message": f"Epoch {epoch}/{max_epochs} completed",
                    "current_epoch": epoch,
                    "total_epochs": max_epochs,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "epoch_time": epoch_duration,
                    "progress_percent": (epoch / max_epochs) * 100,
                    "training_history": training_history,
                    f"task_progress.{task_id}": training_history  # Store task-specific progress
                }))
                
                self.logger.info(f"Epoch {epoch}/{max_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Training completed
            training_history['status'] = 'completed'
            status_queue.put((job_id, 'training_completed', {
                "message": "Training completed successfully",
                "training_history": training_history,
                f"task_progress.{task_id}": training_history
            }))
            
        except Exception as e:
            training_history['status'] = 'error'
            training_history['error'] = str(e)
            status_queue.put((job_id, 'training_error', {
                "message": f"Training failed: {str(e)}",
                "training_history": training_history,
                "traceback": traceback.format_exc()
            }))
            raise

    def setup_job_logging_api(self, task_info, status_queue):
        """Setup logging adapted from .bak file for API usage"""
        job_id = task_info.get('job_id')
        job_folder = task_info.get('job_folder')
        
        if not job_folder:
            raise ValueError("No job_folder specified for logging setup")
            
        # Create logging directory
        log_dir = os.path.join(job_folder, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup file paths
        task_id = task_info.get('task_id')
        csv_log_file = os.path.join(log_dir, f"{task_id}_training.csv")
        db_log_file = os.path.join(log_dir, f"{task_id}_training.db")
        
        # Store paths in task_info for later use
        task_info['csv_log_file'] = csv_log_file
        task_info['db_log_file'] = db_log_file
        
        self.logger.info(f"Setup logging for task {task_id} - CSV: {csv_log_file}, DB: {db_log_file}")
        
        # Create database tables (adapted from .bak file)
        self.create_sql_tables_api(db_log_file)
        
        # Create CSV headers
        self.create_csv_headers_api(csv_log_file)

    def create_sql_tables_api(self, db_log_file):
        """Create SQL tables adapted from .bak file"""
        try:
            import sqlite3
            conn = sqlite3.connect(db_log_file)
            cursor = conn.cursor()
            
            # Create table for epoch-level logs (from .bak file)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS task_logs (
                    task_id TEXT,
                    epoch INTEGER,
                    train_loss REAL,
                    val_loss REAL,
                    elapsed_time REAL,
                    learning_rate REAL,
                    best_val_loss REAL,
                    PRIMARY KEY(task_id, epoch)
                )
            ''')
            
            conn.commit()
            conn.close()
            self.logger.info(f"Created SQL tables in {db_log_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating SQL tables: {e}", exc_info=True)

    def create_csv_headers_api(self, csv_log_file):
        """Create CSV headers adapted from .bak file"""
        try:
            import csv
            with open(csv_log_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['Epoch', 'Train Loss', 'Val Loss', 'Elapsed Time', 'Learning Rate'])
                writer.writeheader()
            self.logger.info(f"Created CSV headers in {csv_log_file}")
        except Exception as e:
            self.logger.error(f"Error creating CSV headers: {e}", exc_info=True)

    def log_to_csv(self, task, epoch, train_loss, val_loss, elapsed_time, current_lr, best_val_loss, delta_t_epoch):
        """Log richer data to CSV file."""
        csv_log_file = task['csv_log_file']
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
        """Log epoch-level data to a SQLite database."""
        sqlite_db_file = task['db_log_file']
        conn = sqlite3.connect(sqlite_db_file)
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO task_logs (task_id, epoch, train_loss, val_loss, elapsed_time, avg_batch_time, learning_rate,
                        best_val_loss, num_learnable_params, batch_size, lookback, max_epochs, early_stopping, model_memory_usage, device)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (task['task_id'], epoch, train_loss, val_loss, elapsed_time, avg_batch_time, current_lr, best_val_loss,
                    task['hyperparams']['NUM_LEARNABLE_PARAMS'], task['hyperparams']['BATCH_SIZE'], task['hyperparams']['LOOKBACK'],
                    task['hyperparams']['MAX_EPOCHS'], early_stopping, model_memory_usage, self.device.type))
        conn.commit()
        conn.close()

    def convert_hyperparams(self, params):
        """Convert specific hyperparameters from string to the correct type, like int or float."""
        converted = params.copy()
        for key, value in converted.items():
            if key in ['LAYERS', 'HIDDEN_UNITS', 'BATCH_SIZE', 'MAX_EPOCHS', 'LR_DROP_PERIOD', 'PLATEAU_PATIENCE', 'VALID_PATIENCE', 'ValidFrequency', 'LOOKBACK', 'REPETITIONS']:
                try:
                    converted[key] = int(value)
                except (ValueError, TypeError):
                    self.logger.warning(f"Could not convert hyperparameter {key} with value '{value}' to int. Using default if available.")
            elif key in ['INITIAL_LR', 'LR_DROP_FACTOR', 'PLATEAU_FACTOR']:
                try:
                    converted[key] = float(value)
                except (ValueError, TypeError):
                    self.logger.warning(f"Could not convert hyperparameter {key} with value '{value}' to float. Using default if available.")
        return converted

    def save_model(self, task, save_path):
        """Save the model state."""
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(task['model'].state_dict(), save_path)
            self.logger.info(f"Model saved to {save_path}")
        except Exception as e:
            self.logger.error(f"Error saving model to {save_path}: {e}")

    def process_task(self, task, update_progress_callback):
        """Process a single training task and set up logging."""
        try:
            # Concise log for starting task
            h_params_summary = {
                k: task['hyperparams'].get(k) for k in ['MODEL_TYPE', 'LAYERS', 'HIDDEN_UNITS', 'MAX_EPOCHS', 'INITIAL_LR', 'BATCH_SIZE', 'LOOKBACK'] if k in task['hyperparams']
            }
            self.logger.info(f"Starting task_id: {task.get('task_id', 'N/A')} with key hyperparams: {h_params_summary}")

            # Load normalization metadata and scaler if applicable
            self.load_normalization_info_and_scaler()

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
            self.logger.info(f"DataLoaders configured for task_id: {task.get('task_id', 'N/A')}")
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
        output_dir_root = os.path.dirname(self.job_manager.get_job_folder()) # e.g., 'output'
        
        log_model_dir = os.path.relpath(model_dir, output_dir_root) if model_dir and output_dir_root in model_dir else model_dir
        log_csv_file = os.path.relpath(csv_log_file, output_dir_root) if csv_log_file and output_dir_root in csv_log_file else csv_log_file
        log_db_file = os.path.relpath(db_log_file, output_dir_root) if db_log_file and output_dir_root in db_log_file else db_log_file

        print(f"Setting up logging for job: {job_id}")
        print(f"Model directory (relative to output): {log_model_dir}")
        print(f"Log files for task {task['task_id']} (relative to output): CSV: {log_csv_file}, DB: {log_db_file}")

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
                output_dir_root_db = os.path.dirname(self.job_manager.get_job_folder())
                log_db_path_create = os.path.relpath(db_log_file, output_dir_root_db) if db_log_file and output_dir_root_db in db_log_file else db_log_file
                self.logger.info(f"Creating new database file at (relative to output): {log_db_path_create}")

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

    def load_normalization_info_and_scaler(self):
        """Loads normalization metadata and the scaler if normalization was applied."""
        self.loaded_scaler = None
        self.scaler_metadata = {}
        job_folder = self.job_manager.get_job_folder()
        if not job_folder:
            self.logger.warning("Job folder not set in JobManager. Cannot load normalization info.")
            return

        metadata_file_path = os.path.join(job_folder, "job_metadata.json")
        if os.path.exists(metadata_file_path):
            try:
                with open(metadata_file_path, 'r') as f_meta:
                    job_meta = json.load(f_meta)
                
                if job_meta.get('normalization_applied', False):
                    scaler_path_relative = job_meta.get('scaler_path')
                    normalized_columns = job_meta.get('normalized_columns')
                    # Assuming target_column is available in task later, or we can add it to job_meta if it's globally unique
                    
                    if scaler_path_relative and normalized_columns:
                        scaler_path_absolute = os.path.join(job_folder, scaler_path_relative)
                        from vestim.services import normalization_service # Local import
                        self.loaded_scaler = normalization_service.load_scaler(scaler_path_absolute)
                        if self.loaded_scaler:
                            self.scaler_metadata = {
                                'scaler_path': scaler_path_absolute,
                                'normalized_columns': normalized_columns,
                                'normalization_applied': True
                                # 'target_column' will be derived from task params later
                            }
                            self.logger.info(f"Successfully loaded scaler from {scaler_path_absolute} and normalization metadata.")
                        else:
                            self.logger.error(f"Failed to load scaler from {scaler_path_absolute}. Reporting will be on normalized scale.")
                    else:
                        self.logger.warning("Normalization metadata incomplete (scaler_path or normalized_columns missing). Reporting on normalized scale.")
                else:
                    self.logger.info("Normalization was not applied according to job_metadata.json.")
            except Exception as e:
                self.logger.error(f"Error loading normalization metadata from {metadata_file_path}: {e}")
        else:
            self.logger.info(f"job_metadata.json not found at {metadata_file_path}. Assuming no normalization or scaler to load.")


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
            self.logger.info(f"--- Starting run_training for task: {task['task_id']} ---") # Added detailed log
            self.logger.info("Starting training loop")
            # Initialize/reset task-specific best original scale validation RMSE tracker
            # Using a unique attribute name per task to avoid conflicts if manager instance is reused for different tasks sequentially
            # though typically a new manager or thread might be used. This is safer.
            setattr(self, f'_task_{task["task_id"]}_best_val_rmse_orig', float('inf'))

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
            loop_start_time = time.time() # Renamed from start_time to avoid confusion with overall training start
            last_validation_time = loop_start_time
            early_stopping = False  # Initialize early stopping flag

            # Max training time logic
            max_training_time_seconds = int(task.get('training_params', {}).get('max_training_time_seconds', 0))
            overall_training_start_time = time.time() # For max training time check
            self.logger.info(f"Max training time set to: {max_training_time_seconds} seconds.")

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

            # Initialize CSV logging for epoch-wise data
            csv_log_file = task['csv_log_file']
            # Ensure the directory for csv_log_file exists (it's task_dir/logs/)
            os.makedirs(os.path.dirname(csv_log_file), exist_ok=True)
            with open(csv_log_file, 'w', newline='') as f: # Added newline=''
                csv_writer = csv.writer(f)
                csv_writer.writerow(["epoch", "train_loss_norm", "val_loss_norm", "best_val_loss_norm", "learning_rate", "elapsed_time_sec", "avg_batch_time_sec", "patience_counter", "model_memory_mb"]) # Header

            # Training loop
            for epoch in range(1, max_epochs + 1):
                if self.stop_requested:  # Ensure thread safety here
                    self.logger.info("Training stopped by user")
                    print("Stopping training...")
                    break
                
                # Check for max training time exceeded
                if max_training_time_seconds > 0:
                    current_training_duration = time.time() - overall_training_start_time
                    if current_training_duration > max_training_time_seconds:
                        self.logger.info(f"Max training time ({max_training_time_seconds}s) exceeded. Stopping training.")
                        print(f"Max training time ({max_training_time_seconds}s) exceeded. Stopping training.")
                        self.stop_requested = True # Use existing flag to gracefully stop
                        early_stopping = True # Indicate it was a form of early stop
                        # Also update task results if possible here or after loop
                        task['results']['early_stopped_reason'] = 'Max training time exceeded'
                        break # Exit epoch loop

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
                # train_epoch now returns: avg_batch_time, avg_loss (normalized), all_train_y_pred_normalized, all_train_y_true_normalized
                avg_batch_time, train_loss_norm, epoch_train_preds_norm, epoch_train_trues_norm = self.training_service.train_epoch(
                    model, model_type, train_loader, optimizer, h_s, h_c, epoch, device, self.stop_requested, task
                )

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
                    # validate_epoch now returns: avg_loss (normalized), all_val_y_pred_normalized, all_val_y_true_normalized
                    val_loss_norm, epoch_val_preds_norm, epoch_val_trues_norm = self.training_service.validate_epoch(
                        model, model_type, val_loader, h_s_val, h_c_val, epoch, device, self.stop_requested, task
                    )

                    current_time = time.time()
                    elapsed_time = current_time - loop_start_time # Use loop_start_time for per-epoch/validation cycle timing
                    delta_t_epoch = (current_time - last_validation_time) / valid_freq
                    last_validation_time = current_time

                    current_lr = optimizer.param_groups[0]['lr']
                    
                    if val_loss_norm < best_validation_loss: # best_validation_loss is also on normalized scale
                        print(f"Epoch: {epoch}, Validation loss improved from {best_validation_loss:.6f} to {val_loss_norm:.6f}. Saving model...")
                        best_validation_loss = val_loss_norm
                        # Save to best_model_path
                        best_model_save_path = task.get('training_params', {}).get('best_model_path')
                        if best_model_save_path:
                            self.save_model(task, save_path=best_model_save_path)
                            self.logger.info(f"Best model saved to: {best_model_save_path}")
                        else:
                            self.logger.warning(f"best_model_path not found in task for epoch {epoch}. Best model not saved.")
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
                    
                    # Calculate scaled RMSE values.
        except Exception as e:
            self.logger.error(f"Error during training for task {task.get('task_id', 'N/A')}: {e}", exc_info=True)
            # Optionally, re-raise the exception or handle it by, for example, emitting an error signal
            if 'update_progress_callback' in locals() and hasattr(update_progress_callback, 'emit'):
                update_progress_callback.emit({'task_error': f"A critical error occurred: {e}"})