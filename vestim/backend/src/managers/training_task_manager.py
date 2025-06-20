import time, os, sys, math, json
import csv
import sqlite3
import torch
import logging
import traceback
from vestim.backend.src.services.training_service import TrainingService
from vestim.backend.src.managers.training_setup_manager import VEstimTrainingSetupManager

def format_time(seconds):
    """Convert seconds to mm:ss format."""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

class TrainingTaskManager:
    """
    Manages the execution of training tasks in a background process.
    This class handles multiple tasks sequentially within a job and provides
    detailed progress updates for the GUI through status queues.
    Based on the working implementation from training_task_manager_qt.py.bak
    """
    def __init__(self):
        """
        Initialize the TrainingTaskManager.
        Heavy initialization is done in the background process.
        """
        self.logger = logging.getLogger(__name__)
        self.stop_requested = False
        self.current_task = None
        

    def process_all_tasks_in_background(self, status_queue, job_info):
        """
        Entry point for background process - handles all tasks in a job sequentially.
        This method processes multiple tasks within a single job.
        """
        job_container = job_info.get('job_container')
        if not job_container:
            raise ValueError("JobContainer not provided in job_info")

        job_id = job_container.job_id
        training_tasks = job_container.get_training_tasks()
        stop_flag = job_container.stop_flag
        
        try:
            if not training_tasks:
                raise ValueError(f"No training tasks found in job_info for job {job_id}")

            self.logger.info(f"[{job_id}] Starting background training for {len(training_tasks)} tasks")
            status_queue.put((job_id, 'initializing', {"message": f"Starting training for {len(training_tasks)} tasks..."}))

            # Initialize services in the background process
            from vestim.backend.src.services.model_training.src.data_loader_service import DataLoaderService
            from vestim.backend.src.services.model_training.src.training_task_service import TrainingTaskService
            from vestim.backend.src.services.model_training.src.LSTM_model_service import LSTMModelService
            
            data_loader_service = DataLoaderService()
            training_service = TrainingTaskService()
            model_service = LSTMModelService()
            
            # Determine device from first task (all tasks in job should use same device)
            if training_tasks:
                hyperparams = training_tasks[0].get('hyperparams', {})
                selected_device_str = hyperparams.get('DEVICE', 'cpu')
                if selected_device_str.startswith("cuda") and torch.cuda.is_available():
                    device = torch.device(selected_device_str)
                else:
                    device = torch.device("cpu")
                self.logger.info(f"[{job_id}] Using device: {device}")
            else:
                raise ValueError("No training tasks found")

            # Process all tasks sequentially
            completed_tasks = 0
            total_tasks = len(training_tasks)
            
            # Initialize job-level training history
            job_training_history = {
                'job_id': job_id,
                'total_tasks': total_tasks,
                'completed_tasks': 0,
                'current_task': None,
                'task_histories': {},
                'job_start_time': time.time(),
                'status': 'training'
            }
            
            for task_idx, task_info_loop in enumerate(training_tasks):
                task_id = task_info_loop.get('task_id')
                
                # Check for stop signal
                if stop_flag and stop_flag.is_set():
                    self.logger.info(f"[{job_id}] Stop signal received before task {task_id}")
                    job_training_history['status'] = 'stopped'
                    status_queue.put((job_id, 'training_stopped', {
                        "message": f"Job stopped before task {task_id}",
                        "job_training_history": job_training_history
                    }))
                    break
                
                # Update job progress
                job_training_history['current_task'] = task_id
                status_queue.put((job_id, 'task_starting', {
                    "message": f"Starting task {task_idx + 1}/{total_tasks}: {task_id}",
                    "current_task_index": task_idx + 1,
                    "total_tasks": total_tasks,
                    "task_id": task_id,
                    "job_training_history": job_training_history
                }))
                
                # Add job context to task_info
                task_info_loop['job_id'] = job_id
                task_info_loop['job_folder'] = job_info.get('job_folder')
                task_info_loop['stop_flag'] = stop_flag
                
                try:
                    # Process individual task
                    self.process_single_task_api(task_info_loop, status_queue, device, 
                                               data_loader_service, training_service, model_service, stop_flag)
                    
                    completed_tasks += 1
                    job_training_history['completed_tasks'] = completed_tasks
                    
                    status_queue.put((job_id, 'task_completed', {
                        "message": f"Completed task {task_idx + 1}/{total_tasks}: {task_id}",
                        "completed_tasks": completed_tasks,
                        "total_tasks": total_tasks,
                        "task_id": task_id,
                        "job_training_history": job_training_history
                    }))
                    
                except Exception as e:
                    self.logger.error(f"[{job_id}] Error in task {task_id}: {e}", exc_info=True)
                    status_queue.put((job_id, 'task_error', {
                        "message": f"Task {task_id} failed: {str(e)}",
                        "task_id": task_id,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                        "job_training_history": job_training_history
                    }))
                    # Continue with next task instead of stopping the entire job
                    continue
            
            # All tasks completed
            if job_training_history['status'] == 'training':
                job_training_history['status'] = 'completed'
                status_queue.put((job_id, 'job_training_completed', {
                    "message": f"All {total_tasks} tasks completed successfully",
                    "completed_tasks": completed_tasks,
                    "total_tasks": total_tasks,
                    "job_training_history": job_training_history
                }))

        except Exception as e:
            error_msg = f"[{job_id}] Error during job training: {e}"
            self.logger.error(error_msg, exc_info=True)
            status_queue.put((job_id, 'error', {
                "message": str(e),
                "error": error_msg,
                "traceback": traceback.format_exc()
            }))

    def process_single_task_api(self, task_info, status_queue, device, data_loader_service, training_service, model_service, stop_flag):
        """
        Process a single training task with comprehensive progress tracking.
        Adapted from .bak file logic with API integration.
        """
        job_id = task_info.get('job_id')
        task_id = task_info.get('task_id')
        hyperparams = task_info.get('hyperparams', {})
        
        if not hyperparams:
            raise ValueError("No hyperparameters found in task_info")
            
        self.logger.info(f"Processing task {task_id} with hyperparams keys: {list(hyperparams.keys())}")
        
        try:
            # Setup job logging
            status_queue.put((job_id, 'setup_logging', {"message": f"Setting up logging for task {task_id}..."}))
            self.setup_job_logging_api(task_info, status_queue)
            
            # Create data loaders
            status_queue.put((job_id, 'creating_dataloader', {"message": f"Creating data loaders for task {task_id}..."}))
            train_loader, val_loader = self.create_data_loaders_api(task_info, data_loader_service)
            
            status_queue.put((job_id, 'dataloader_created', {
                "message": f"Data loaders created. Train: {len(train_loader)}, Val: {len(val_loader)}",
                "train_batches": len(train_loader),
                "val_batches": len(val_loader),
                "task_id": task_id
            }))
            
            # Create model
            status_queue.put((job_id, 'creating_model', {"message": f"Creating model for task {task_id}..."}))
            model = self.create_model_api(task_info, model_service, device)
            
            # Start training
            status_queue.put((job_id, 'training_started', {
                "message": f"Starting training for task {task_id} - {hyperparams.get('MAX_EPOCHS', 10)} epochs",
                "total_epochs": int(hyperparams.get('MAX_EPOCHS', 10)),
                "task_id": task_id
            }))
            
            # Run the actual training with detailed progress tracking
            self.run_training_api(task_info, status_queue, model, train_loader, val_loader, device, stop_flag, training_service)
            
        except Exception as e:
            self.logger.error(f"Error during task processing: {str(e)}", exc_info=True)
            status_queue.put((job_id, 'task_error', {
                "message": f"Task {task_id} failed: {str(e)}", 
                "task_id": task_id,
                "traceback": traceback.format_exc()
            }))

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
        
        # Get the job folder and construct path to training data
        job_id = task_info.get('job_id')
        
        # Construct the job folder path (assuming it's in output/ directory)
        import os
        # Get the current working directory and construct the path to the job folder
        current_dir = os.getcwd()
        job_folder = os.path.join(current_dir, 'output', job_id)
        
        # Check if job folder exists
        if not os.path.exists(job_folder):
            # Fallback to data/ directory if job-specific folder doesn't exist
            data_folder = os.path.join(current_dir, 'data')
            if os.path.exists(data_folder):
                train_data_path = data_folder
                self.logger.info(f"Using fallback data directory: {train_data_path}")
            else:
                raise ValueError(f"Neither job folder {job_folder} nor data folder {data_folder} exists")
        else:
            train_data_path = job_folder
            self.logger.info(f"Using job-specific data directory: {train_data_path}")
        
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
        
        self.logger.info(f"Data loaders created - Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        return train_loader, val_loader

    def run_training_api(self, task_info, status_queue, train_loader, val_loader, device, stop_flag, training_service):
        """Run training loop with detailed progress updates and full model training"""
        job_id = task_info.get('job_id') 
        task_id = task_info.get('task_id')
        hyperparams = task_info.get('hyperparams', {})
        
        # Training parameters
        max_epochs = int(hyperparams.get('MAX_EPOCHS', 10))
        model_type = hyperparams.get('MODEL_TYPE', 'LSTM')
        validation_freq = int(hyperparams.get('VALIDATION_FREQ', 1))
        early_stopping_patience = int(hyperparams.get('EARLY_STOPPING_PATIENCE', 10))
        initial_lr = float(hyperparams.get('INITIAL_LR', 0.001))
        
        # Initialize comprehensive training history for GUI persistence
        training_start_time = time.time()
        training_history = {
            # Core epoch data - each entry is a complete epoch record
            'epoch_logs': [],  # List of dicts with all epoch details
            
            # Arrays for plotting (parallel arrays for efficiency)
            'train_losses': [],    # For loss plots
            'val_losses': [],      # For loss plots  
            'learning_rates': [],  # For LR tracking
            'epoch_times': [],     # For performance monitoring
            'patience_counters': [], # For early stopping tracking
            
            # Training state tracking
            'best_epoch': 0,
            'best_val_loss': float('inf'),
            'current_epoch': 0,
            'total_epochs': max_epochs,
            'status': 'training',
            'training_start_time': training_start_time,
            
            # Task metadata
            'job_id': job_id,
            'task_id': task_id,
            'hyperparameters': hyperparams,
            
            # GUI state data
            'current_train_loss': 0.0,
            'current_val_loss': 0.0,
            'current_lr': initial_lr,
            'time_elapsed': 0.0,
            'progress_percent': 0.0
        }
        
        try:
            # Create model using LSTM model service
            from vestim.backend.src.services.model_training.src.LSTM_model_service import LSTMModelService
            model_service = LSTMModelService()
            
            # Build model with hyperparameters
            model_params = {
                'INPUT_SIZE': len(hyperparams.get('FEATURE_COLUMNS', ['Current', 'Temp', 'Voltage'])),
                'HIDDEN_UNITS': int(hyperparams.get('HIDDEN_UNITS', 50)),
                'LAYERS': int(hyperparams.get('LAYERS', 1)),
                'DROPOUT_PROB': float(hyperparams.get('DROPOUT_PROB', 0.5))
            }
            
            model = model_service.build_lstm_model(model_params)
            model = model.to(device)
            
            # Create optimizer and scheduler
            optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
            scheduler_type = hyperparams.get('SCHEDULER_TYPE', 'StepLR') 
            if scheduler_type == 'StepLR':
                lr_period = int(hyperparams.get('LR_PERIOD', 5))
                lr_factor = float(hyperparams.get('LR_PARAM', 0.1))
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_period, gamma=lr_factor)
            
            self.logger.info(f"Starting training - Model: {model_type}, Epochs: {max_epochs}, Device: {device}")
            
            # Training variables
            patience_counter = 0
            
            # Main training loop
            for epoch in range(1, max_epochs + 1):
                # Check for stop signal
                if stop_flag and stop_flag.is_set():
                    training_history['status'] = 'stopped'
                    status_queue.put((job_id, 'training_stopped', {
                        "message": f"Training stopped at epoch {epoch}",
                        "training_history": training_history
                    }))
                    break
                
                epoch_start_time = time.time()
                
                # Initialize hidden states for training
                batch_size = train_loader.batch_size or int(hyperparams.get('BATCH_SIZE', 32))
                h_s = torch.zeros(model.num_layers, batch_size, model.hidden_units).to(device)
                h_c = torch.zeros(model.num_layers, batch_size, model.hidden_units).to(device)
                
                # Train one epoch using training service
                avg_batch_time, train_loss, _, _ = training_service.train_epoch(
                    model, model_type, train_loader, optimizer, h_s, h_c, 
                    epoch, device, stop_flag, task_info
                )
                
                # Validation (only at specified frequency)
                val_loss = None
                if epoch == 1 or epoch % validation_freq == 0 or epoch == max_epochs:
                    # Initialize hidden states for validation
                    val_batch_size = val_loader.batch_size or batch_size
                    h_s_val = torch.zeros(model.num_layers, val_batch_size, model.hidden_units).to(device)
                    h_c_val = torch.zeros(model.num_layers, val_batch_size, model.hidden_units).to(device)
                      # Validate using training service
                    val_loss, _, _ = training_service.validate_epoch(
                        model, model_type, val_loader, h_s_val, h_c_val, epoch, device, stop_flag, task_info
                    )
                    
                    # Update best validation loss and patience
                    if val_loss < training_history['best_val_loss']:
                        training_history['best_val_loss'] = val_loss
                        training_history['best_epoch'] = epoch
                        patience_counter = 0
                    else:
                        patience_counter += 1
                else:
                    # Use previous validation loss if not validating this epoch
                    val_loss = training_history['val_losses'][-1] if training_history['val_losses'] else train_loss
                
                # Update learning rate
                if scheduler:
                    scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                
                epoch_end_time = time.time()
                epoch_duration = epoch_end_time - epoch_start_time
                total_time_elapsed = epoch_end_time - training_start_time
                
                # Create comprehensive epoch log entry
                epoch_log = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_val_loss': training_history['best_val_loss'],
                    'learning_rate': current_lr,
                    'epoch_time': epoch_duration,
                    'avg_batch_time': avg_batch_time,
                    'patience_counter': patience_counter,
                    'timestamp': epoch_end_time,
                    'epochs_since_best': epoch - training_history['best_epoch'],
                    'validation_patience_left': max(0, early_stopping_patience - patience_counter)
                }
                
                # Update training history arrays for GUI plotting
                training_history['epoch_logs'].append(epoch_log)
                training_history['train_losses'].append(train_loss)
                training_history['val_losses'].append(val_loss)
                training_history['learning_rates'].append(current_lr)
                training_history['epoch_times'].append(epoch_duration)
                training_history['patience_counters'].append(patience_counter)
                
                # Update current state for GUI
                training_history['current_epoch'] = epoch
                training_history['current_train_loss'] = train_loss
                training_history['current_val_loss'] = val_loss
                training_history['current_lr'] = current_lr
                training_history['time_elapsed'] = total_time_elapsed
                training_history['progress_percent'] = (epoch / max_epochs) * 100
                
                # Send comprehensive progress update with all GUI data
                status_queue.put((job_id, 'training_progress', {
                    "message": f"Epoch {epoch}/{max_epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}, Best: {training_history['best_val_loss']:.6f}",
                    "training_history": training_history,  # Complete persistent data structure
                    f"task_progress.{task_id}": training_history,  # Store task-specific progress
                    "hyperparameters": hyperparams,
                    "training_status": "in_progress"
                }))
                
                self.logger.info(f"Epoch {epoch}/{max_epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}, Best: {training_history['best_val_loss']:.6f} (Epoch {training_history['best_epoch']}) - Patience: {patience_counter}/{early_stopping_patience}")
                
                # Check for early stopping
                if patience_counter >= early_stopping_patience:
                    self.logger.info(f"Early stopping triggered at epoch {epoch}")
                    training_history['status'] = 'early_stopped'
                    status_queue.put((job_id, 'training_early_stopped', {
                        "message": f"Training stopped early at epoch {epoch} due to validation patience",
                        "training_history": training_history,
                        f"task_progress.{task_id}": training_history,
                        "reason": "early_stopping"
                    }))
                    break
            
            # Training completed successfully
            if training_history['status'] == 'training':
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
        job_id = task.get('job_id')
        model_dir = task.get('model_dir')  # Path where task-related logs are stored

        # Retrieve log file paths from the task
        csv_log_file = task.get('csv_log_file')
        db_log_file = task.get('db_log_file')

        # Log information about the task and log file paths
        output_dir_root = os.path.dirname(task.get('job_folder')) # e.g., 'output'
        
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
                output_dir_root_db = os.path.dirname(db_log_file)
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