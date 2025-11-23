import time, os, sys, math, json
import csv
import sqlite3
import torch
import gc
try:
    import fireducks.pandas as pd
except ImportError:
    import pandas as pd
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from vestim.gateway.src.job_manager_qt import JobManager
from vestim.gateway.src.training_setup_manager_qt import VEstimTrainingSetupManager
from vestim.gateway.src.hyper_param_manager_qt import VEstimHyperParamManager
from vestim.services.model_training.src.data_loader_service import DataLoaderService
from vestim.services.model_training.src.training_task_service import TrainingTaskService
try:
    from vestim.services.model_training.src.cuda_graphs_training_service import CUDAGraphsTrainingService
    CUDA_GRAPHS_AVAILABLE = True
except ImportError as e:
    print(f"CUDA Graphs optimization not available: {e}")
    CUDA_GRAPHS_AVAILABLE = False
import logging
import multiprocessing as mp
import platform, sys

def train_one_task(task_cfg, progress_queue=None, job_folder=None, global_params=None):
    """
    This function runs the entire training for one task in a separate process.
    Each task runs in its own isolated process which is terminated after completion,
    ensuring all resources (including DataLoader workers) are properly freed.
    
    Args:
        task_cfg: Task configuration dictionary
        progress_queue: Multiprocessing queue for sending progress updates to GUI
        job_folder: Path to job folder
        global_params: Global parameters dictionary
    """
    import os
    task_id = task_cfg.get('task_id', 'unknown')
    pid = os.getpid()
    print(f"\n{'='*80}")
    print(f"[PROCESS START] Task {task_id} | PID: {pid}")
    print(f"{'='*80}\n")
    
    if platform.system() == "Linux":
        try:
            import resource
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            target = 8192
            new_soft = min(max(soft, target), hard)
            if new_soft != soft:
                resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
        except Exception as e:
            print(f"[WARN] Could not adjust RLIMIT_NOFILE: {e}", file=sys.stderr)

    # Reconstruct a minimal JobManager and set the job_id from the job_folder path
    # This ensures the child process uses the correct job folder
    job_manager = JobManager()
    if job_folder:
        # Extract job_id from the job_folder path
        job_manager.job_id = os.path.basename(job_folder)
        
    task_manager = TrainingTaskManager(job_manager=job_manager, global_params=global_params)
    
    try:
        # Run the task - the task_manager.process_task already sends updates via progress_queue
        task_manager.process_task_with_queue(task_cfg, progress_queue)
        
        # Send completion message with training results
        if progress_queue is not None:
            # Get the training results from the task manager (populated during training)
            training_results = task_manager.training_results.get(task_id, {})
            print(f"[SUBPROCESS] Sending training results for task {task_id}: {training_results}")
            progress_queue.put({
                "status": "done",
                "training_results": training_results  # Include training results in completion message
            })
        
        print(f"\n[PROCESS SUCCESS] Task {task_id} completed successfully | PID: {pid}")
    
    except Exception as e:
        print(f"\n[PROCESS ERROR] Task {task_id} failed | PID: {pid} | Error: {e}", file=sys.stderr)
        if progress_queue is not None:
            progress_queue.put({'task_error': str(e)})
        raise  # Re-raise to ensure the process exits with non-zero code
    finally:
        # Explicit cleanup to ensure resources are freed before process termination
        print(f"\n[CLEANUP START] Task {task_id} | PID: {pid}")
        try:
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print(f"[CLEANUP] CUDA cache cleared | PID: {pid}")
            
            # Force garbage collection
            gc.collect()
            print(f"[CLEANUP] Garbage collection completed | PID: {pid}")
            
            # Delete the task_manager to close any open file handles
            del task_manager
            del job_manager
            print(f"[CLEANUP] Managers deleted | PID: {pid}")
            
        except Exception as cleanup_error:
            print(f"[WARN] Error during cleanup: {cleanup_error}", file=sys.stderr)
        
        print(f"\n{'='*80}")
        print(f"[PROCESS END] Task {task_id} | PID: {pid} | Process will now terminate")
        print(f"{'='*80}\n")


def train_one_task_optuna(task_cfg, progress_queue, result_queue, job_folder, global_params, trial_number):
    """
    Top-level function for running ONE Optuna trial in a completely isolated subprocess.
    
    This function MUST be at module level (not nested) for multiprocessing.spawn to work.
    Each Optuna trial runs in its own process, ensuring complete resource cleanup after
    the trial completes. This prevents "too many open files" errors during multi-trial
    optimization (e.g., 50 trials = 50 clean process lifecycles).
    
    Architecture mirrors train_one_task() but sends results via result_queue instead of
    storing in training_results dict (since each trial is isolated).
    """
    import sys
    import platform
    
    task_id = task_cfg.get('task_id', 'unknown')
    pid = os.getpid()
    print(f"\n{'='*80}")
    print(f"[OPTUNA PROCESS START] Trial {trial_number} | Task {task_id} | PID: {pid}")
    print(f"{'='*80}\n")
    
    # Increase file descriptor limit on Linux
    if platform.system() == "Linux":
        try:
            import resource
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            target = 8192
            new_soft = min(max(soft, target), hard)
            if new_soft != soft:
                resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
        except Exception as e:
            print(f"[WARN] Could not adjust RLIMIT_NOFILE: {e}", file=sys.stderr)

    # Reconstruct JobManager
    job_manager = JobManager()
    if job_folder:
        job_manager.job_id = os.path.basename(job_folder)
    
    task_manager = TrainingTaskManager(job_manager=job_manager, global_params=global_params)
    
    try:
        # Run Optuna training task
        task_manager.process_task_with_queue(task_cfg, progress_queue)
        
        # Extract results and send via result_queue
        final_results = task_cfg.get('results', {})
        best_val_loss = final_results.get('best_validation_loss_normalized', float('inf'))
        
        result_data = {
            'best_validation_loss': best_val_loss,
            'pruned': False,
            'trial_number': trial_number
        }
        
        if result_queue is not None:
            result_queue.put(result_data)
            print(f"[OPTUNA SUBPROCESS] Trial {trial_number} result sent: {best_val_loss:.6f}")
        
        print(f"\n[OPTUNA PROCESS SUCCESS] Trial {trial_number} completed | PID: {pid}")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n[OPTUNA PROCESS ERROR] Trial {trial_number} failed | PID: {pid}", file=sys.stderr)
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        
        # Send error result
        if result_queue is not None:
            result_queue.put({
                'best_validation_loss': float('inf'),
                'pruned': False,
                'trial_number': trial_number,
                'error': str(e)
            })
        
        sys.exit(1)


def format_time(seconds):
    """Convert seconds to mm:ss format."""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def format_time_long(seconds):
    """Convert seconds to hh:mm:ss format for longer durations."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}h:{minutes:02d}m:{seconds:02d}s"

class TrainingTaskManager:
    def __init__(self, job_manager=None, global_params=None):
        self.logger = logging.getLogger(__name__)
        self.job_manager = job_manager if job_manager else JobManager()
        self.data_loader_service = DataLoaderService()
        
        # Store global_params first before using it
        self.global_params = global_params if global_params else {}
        self.logger.info(f"TrainingTaskManager: Received global_params with DEVICE_SELECTION: {self.global_params.get('DEVICE_SELECTION', 'NOT_FOUND')}")
        
        # Initialize training service with CUDA Graphs if available and enabled
        use_cuda_graphs = self.global_params.get('USE_CUDA_GRAPHS', False)
        
        # Auto-enable CUDA Graphs for FNN models on CUDA devices (smart default)
        if not use_cuda_graphs and self.global_params:
            model_type = self.global_params.get('MODEL_TYPE', 'LSTM')
            device_selection = self.global_params.get('DEVICE_SELECTION', 'cpu')
            if model_type == 'FNN' and 'cuda' in device_selection.lower() and CUDA_GRAPHS_AVAILABLE:
                use_cuda_graphs = True
                self.logger.info("Auto-enabling CUDA Graphs for FNN model on CUDA device")
        
        # Determine device based on global_params or fallback
        selected_device_str = self.global_params.get('DEVICE_SELECTION', 'cpu')  # FIXED: Default to 'cpu' instead of 'cuda:0'
        self.logger.info(f"TrainingTaskManager: Device selection from params: '{selected_device_str}' (type: {type(selected_device_str)})")
        self.logger.info(f"TrainingTaskManager: CUDA available: {torch.cuda.is_available()}, CUDA device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
        try:
            if selected_device_str.lower().startswith("cuda") and not torch.cuda.is_available():
                self.logger.warning(f"CUDA device {selected_device_str} selected, but CUDA is not available. Falling back to CPU.")
                self.device = torch.device("cpu")
            elif selected_device_str.lower().startswith("cuda") or selected_device_str.upper() == "CUDA":
                # Handle both "cuda:0", "cuda", and "CUDA" formats
                if selected_device_str.upper() == "CUDA":
                    # Convert generic "CUDA" to "cuda:0"
                    device_str = "cuda:0"
                else:
                    device_str = selected_device_str
                self.device = torch.device(device_str)
                self.logger.info(f"TrainingTaskManager: Successfully set device to CUDA: {self.device}")
            elif selected_device_str.upper() == "CPU":  # FIXED: Use case-insensitive comparison
                self.device = torch.device("cpu")
                self.logger.info(f"TrainingTaskManager: Successfully set device to CPU: {self.device}")
            else: # Default fallback if string is unrecognized
                self.logger.warning(f"Unrecognized device selection '{selected_device_str}'. Falling back to CPU.")
                self.device = torch.device("cpu")  # FIXED: Always fallback to CPU
        except Exception as e:
            self.logger.error(f"Error setting device to '{selected_device_str}': {e}. Falling back to CPU.")
            self.device = torch.device("cpu")

        if use_cuda_graphs and CUDA_GRAPHS_AVAILABLE:
            # FIXED: Pass the device to CUDA Graphs service so it respects GUI selection
            self.training_service = CUDAGraphsTrainingService(device=self.device)
            self.logger.info("CUDA Graphs training service initialized for RTX 5070 optimization")
        else:
            # FIXED: Pass the device to TrainingTaskService so it respects GUI selection
            self.training_service = TrainingTaskService(device=self.device)
            if use_cuda_graphs and not CUDA_GRAPHS_AVAILABLE:
                self.logger.warning("CUDA Graphs requested but not available, using standard training")
            else:
                self.logger.info("Using standard training service")
        
        self.training_setup_manager = VEstimTrainingSetupManager(job_manager=self.job_manager)
        self.current_task = None
        self.stop_requested = False
        self.loaded_scaler = None # For storing the loaded scaler
        self.scaler_metadata = {} # For storing normalization metadata (path, columns, target)
        self.training_results = {}
        
        # Initialize job-level timer (will be set when first task starts)
        self.job_start_time = None
        
        # CUDA Graphs batch size tracking for dynamic reset
        self.previous_batch_size = None
        
        self.logger.info(f"TrainingTaskManager initialized with device: {self.device}")

        self.training_thread = None  # Initialize the training thread here for PyQt
       
        # WandB disabled for distribution
        self.use_wandb = False  # WandB functionality removed
        self.wandb_enabled = False

    def train_one_task(self, task_cfg, progress_queue=None):
        train_one_task(task_cfg, progress_queue, self.job_manager.get_job_folder(), self.global_params)
    
    def process_task_with_queue(self, task, progress_queue):
        """
        Process a task using a multiprocessing queue instead of PyQt signals.
        This is used when running tasks in separate processes.
        """
        # Store the progress queue for stop signal checking
        self.progress_queue = progress_queue
        
        # Create a simple callback wrapper that puts messages in the queue
        class QueueCallback:
            def emit(self, data):
                if progress_queue is not None:
                    progress_queue.put(data)
        
        # Call the existing process_task method with our queue-based callback
        self.process_task(task, QueueCallback())

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
        """
        Process a single training task. This method acts as a router,
        directing the task to the appropriate training loop (standard or Optuna).
        """
        # Initialize job-level timer on first task
        if self.job_start_time is None:
            self.job_start_time = time.time()
            self.logger.info("Job-level timer started for first task")
        
        # Initialize task timer at the very beginning
        self.task_start_time = time.time()
        
        train_loader, val_loader = None, None  # Initialize to ensure they exist for the finally block
        try:
            # Reset CUDA graphs state before starting a new task to prevent state leakage
            if hasattr(self.training_service, 'reset_cuda_graphs'):
                self.logger.info("Resetting CUDA Graphs state for new task.")
                self.training_service.reset_cuda_graphs()

            # Concise log for starting task - only include model-appropriate parameters
            task_hyperparams = task['hyperparams']
            model_type = task_hyperparams.get('MODEL_TYPE', 'LSTM')
            
            # Common parameters
            essential_keys = ['MODEL_TYPE', 'MAX_EPOCHS', 'INITIAL_LR', 'BATCH_SIZE', 'LOOKBACK']
            
            # Add model-specific parameters
            if model_type in ['LSTM', 'GRU']:
                essential_keys.extend(['RNN_LAYER_SIZES', 'LAYERS', 'HIDDEN_UNITS', 'GRU_LAYERS', 'GRU_HIDDEN_UNITS'])
            elif model_type == 'FNN':
                essential_keys.extend(['HIDDEN_LAYER_SIZES', 'DROPOUT_PROB'])
            
            h_params_summary = {
                k: task_hyperparams.get(k) for k in essential_keys if k in task_hyperparams
            }
            repetition_info = f" (Repetition {task.get('repetition', 1)} of {task_hyperparams.get('REPETITIONS', 1)})" if int(task_hyperparams.get('REPETITIONS', 1)) > 1 else ""
            self.logger.info(f"Starting task_id: {task.get('task_id', 'N/A')}{repetition_info} with key hyperparams: {h_params_summary}")

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

            # Send device information to the GUI log
            device_str = str(self.device)
            if device_str == 'cpu':
                device_display = 'CPU'
            elif 'cuda' in device_str:
                try:
                    if torch.cuda.is_available() and 'cuda' in device_str:
                        gpu_idx = int(device_str.split(':')[1]) if ':' in device_str else 0
                        gpu_name = torch.cuda.get_device_name(gpu_idx)
                        device_display = f"{device_str.upper()} ({gpu_name})"
                    else:
                        device_display = device_str.upper()
                except Exception:
                    device_display = device_str.upper()
            else:
                device_display = device_str

            # Send initial device log message to GUI
            update_progress_callback.emit({
                'initial_log_message': f"<b>Starting training on device:</b> {device_display}"
            })

            # ROUTER: Check if this is an Optuna task and call the appropriate loop
            if 'optuna_trial' in task and task['optuna_trial'] is not None:
                self.logger.info(f"Task {task['task_id']} is an Optuna trial. Using Optuna-specific training loop.")
                update_progress_callback.emit({'status': f'Running Optuna Trial {task["optuna_trial"].number} for {task["hyperparams"]["MAX_EPOCHS"]} epochs...'})
                self.run_optuna_training(task, update_progress_callback, train_loader, val_loader, self.device)
            else:
                self.logger.info(f"Task {task['task_id']} is a standard training task. Using standard training loop.")
                update_progress_callback.emit({'status': f'Training {model_type} model for {task["hyperparams"]["MAX_EPOCHS"]} epochs...'})
                self.run_training(task, update_progress_callback, train_loader, val_loader, self.device)

        except Exception as e:
            # Catch TrialPruned exception specifically to avoid logging it as an error
            # as it's an expected outcome for Optuna.
            try:
                import optuna
                if isinstance(e, optuna.exceptions.TrialPruned):
                    self.logger.info(f"Optuna trial pruned for task {task.get('task_id', 'N/A')}.")
                    raise e # Re-raise to be handled by the Optuna study
            except ImportError:
                pass # Optuna not installed, can't be a TrialPruned exception.
            
            self.logger.error(f"Error during task processing: {str(e)}", exc_info=True)
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

        # Ensure the model_dir exists
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)  # Create the directory if it does not exist

        # Create SQLite tables if they do not exist
        self.create_sql_tables(db_log_file)

    def create_sql_tables(self, db_log_file):
        """Create the necessary SQL tables for task-level and batch-level logging."""
        if not db_log_file:
            self.logger.warning("Database log file path is not provided. Skipping SQL table creation.")
            return
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
                        from vestim.services.data_processor.src import normalization_service # Local import
                        self.loaded_scaler = normalization_service.load_scaler(scaler_path_absolute)
                        if self.loaded_scaler:
                            self.scaler_metadata = {
                                'scaler_path': scaler_path_absolute,
                                'normalized_columns': normalized_columns,
                                'normalization_applied': True
                                # 'target_column' will be derived from task params later
                            }
                            self.logger.info(f"Successfully loaded scaler from {scaler_path_absolute} and normalization metadata.")
                            
                            # Debug: Print scaler parameters to understand scaling factors
                            try:
                                if hasattr(self.loaded_scaler, 'data_min_') and hasattr(self.loaded_scaler, 'data_max_'):
                                    self.logger.info("=== SCALER DEBUG INFO ===")
                                    self.logger.info(f"Scaler type: {type(self.loaded_scaler).__name__}")
                                    self.logger.info(f"Normalized columns: {normalized_columns}")
                                    for i, col in enumerate(normalized_columns):
                                        if i < len(self.loaded_scaler.data_min_) and i < len(self.loaded_scaler.data_max_):
                                            data_min = float(self.loaded_scaler.data_min_[i])
                                            data_max = float(self.loaded_scaler.data_max_[i])
                                            data_range = data_max - data_min
                                            self.logger.info(f"  {col}: min={data_min:.6f}, max={data_max:.6f}, range={data_range:.6f}")
                                            # For voltage, show expected multiplier effect
                                            if "voltage" in col.lower():
                                                expected_rmse_scaling = data_range * 1000  # V to mV
                                                self.logger.info(f"    -> Expected RMSE scaling for normalized loss to mV: ~{expected_rmse_scaling:.1f}x")
                                            elif "soc" in col.lower() or "soe" in col.lower() or "sop" in col.lower():
                                                expected_rmse_scaling = data_range * 100  # fraction to percentage
                                                self.logger.info(f"    -> Expected RMSE scaling for normalized loss to %: ~{expected_rmse_scaling:.1f}x")
                                elif hasattr(self.loaded_scaler, 'scale_') and hasattr(self.loaded_scaler, 'min_'):
                                    self.logger.info("=== SCALER DEBUG INFO (StandardScaler) ===")
                                    self.logger.info(f"Scaler type: {type(self.loaded_scaler).__name__}")
                                    self.logger.info(f"Normalized columns: {normalized_columns}")
                                    for i, col in enumerate(normalized_columns):
                                        if i < len(self.loaded_scaler.scale_):
                                            scale = float(self.loaded_scaler.scale_[i])
                                            self.logger.info(f"  {col}: scale={scale:.6f}")
                                            # For voltage, show expected multiplier effect
                                            if "voltage" in col.lower():
                                                expected_rmse_scaling = (1.0 / scale) * 1000  # Standard scaling to mV
                                                self.logger.info(f"    -> Expected RMSE scaling for normalized loss to mV: ~{expected_rmse_scaling:.1f}x")
                                self.logger.info("=== END SCALER DEBUG ===")
                            except Exception as debug_e:
                                self.logger.warning(f"Could not extract scaler debug info: {debug_e}")
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

    def get_optimal_num_workers(self, model_type="LSTM"):
        """Automatically determine optimal number of workers based on system specs and model type.
        
        Args:
            model_type: Model type (LSTM, GRU, FNN) to optimize for
        """
        import os
        try:
            import psutil
            cpu_cores = os.cpu_count()
            available_ram_gb = psutil.virtual_memory().available / (1024**3)  # GB
            
            # Model-specific recommendations
            # LSTM/GRU: Lower workers due to sequence processing overhead
            # FNN: Higher workers due to simpler data structure
            if model_type in ["LSTM", "GRU", "LSTM_EMA", "LSTM_LPF"]:
                if available_ram_gb < 8:
                    recommended_workers = 0  # Single-process for low RAM
                elif available_ram_gb < 16:
                    recommended_workers = 2  # Conservative for RNN
                else:
                    recommended_workers = min(4, cpu_cores // 2) if cpu_cores else 2
            else:  # FNN and other models
                if available_ram_gb < 8:
                    recommended_workers = 2
                elif available_ram_gb < 16:
                    recommended_workers = 4
                else:
                    recommended_workers = min(8, cpu_cores) if cpu_cores else 4
            
            self.logger.info(f"Auto-configured NUM_WORKERS for {model_type}: {recommended_workers} (CPU cores: {cpu_cores}, Available RAM: {available_ram_gb:.1f}GB)")
            return recommended_workers
            
        except ImportError:
            # Fallback if psutil not available
            cpu_cores = os.cpu_count()
            if model_type in ["LSTM", "GRU", "LSTM_EMA", "LSTM_LPF"]:
                recommended_workers = min(2, cpu_cores // 2) if cpu_cores else 2
            else:
                recommended_workers = min(4, cpu_cores) if cpu_cores else 4
            self.logger.info(f"Auto-configured NUM_WORKERS for {model_type}: {recommended_workers} (CPU cores: {cpu_cores})")
            return recommended_workers
        except Exception as e:
            self.logger.warning(f"Error determining optimal workers, using default: {e}")
            return 2 if model_type in ["LSTM", "GRU", "LSTM_EMA", "LSTM_LPF"] else 4

    def _cleanup_orphaned_workers(self):
        """Clean up orphaned DataLoader worker processes on Linux systems."""
        try:
            import platform
            if platform.system() != 'Linux':
                return  # Windows handles this automatically
                
            import psutil
            current_process = psutil.Process()
            
            # Find potential DataLoader worker processes
            worker_count = 0
            for child in current_process.children(recursive=False):  # Direct children only
                try:
                    cmdline = child.cmdline()
                    if (cmdline and len(cmdline) > 1 and 
                        'python' in cmdline[0].lower() and 
                        any('dataloader' in str(arg).lower() or 'worker' in str(arg).lower() for arg in cmdline[1:])):
                        
                        self.logger.debug(f"Terminating DataLoader worker process: PID {child.pid}")
                        child.terminate()
                        worker_count += 1
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            if worker_count > 0:
                self.logger.info(f"Cleaned up {worker_count} orphaned DataLoader worker processes")
                
        except ImportError:
            # psutil not available - skip cleanup
            pass
        except Exception as e:
            self.logger.warning(f"Error during orphaned worker cleanup: {e}")


    def create_data_loaders(self, task):
        """Create data loaders for the current task using separate train, val, test folders."""
        feature_cols = task['data_loader_params']['feature_columns']
        target_col = task['data_loader_params']['target_column']
        
        # Consolidate parameters from both global_params and task['hyperparams']
        # Task-specific hyperparams will override global settings
        combined_params = {**self.global_params, **task['hyperparams']}
        
        # Get model type for optimized worker configuration
        model_type = combined_params.get('MODEL_TYPE', 'LSTM')

        # Enhanced data loading configuration from combined params
        # These values come from the hyperparam GUI and are stored in hyperparams.json
        # Also check data_loader_params for backward compatibility
        num_workers = int(combined_params.get('NUM_WORKERS', task['data_loader_params'].get('num_workers', self.get_optimal_num_workers(model_type))))
        pin_memory = bool(combined_params.get('PIN_MEMORY', task['data_loader_params'].get('pin_memory', True)))
        
        # Prefetch factor: how many batches to pre-load per worker (user-configurable via GUI)
        # Check both hyperparams and data_loader_params (stored during task setup)
        default_prefetch = 2 if model_type in ["LSTM", "GRU", "LSTM_EMA", "LSTM_LPF"] else 4
        user_prefetch = int(combined_params.get('PREFETCH_FACTOR', task['data_loader_params'].get('prefetch_factor', default_prefetch)))
        # Cap at user_prefetch value (no artificial limit) - user knows their system
        prefetch_factor = user_prefetch if num_workers > 0 else None
        
        # Persistent workers: reuse worker processes across epochs for performance
        # User can disable via GUI, otherwise auto-enable when num_workers > 0
        # Safe because subprocess-per-task ensures all workers are killed after task completion
        user_persistent = combined_params.get('PERSISTENT_WORKERS', None)
        persistent_workers = bool(user_persistent) if user_persistent is not None else (num_workers > 0)

        self.logger.info(f"Initial data loading optimization settings for {model_type}:")
        self.logger.info(f"  - CPU Threads (NUM_WORKERS): {num_workers}")
        self.logger.info(f"  - Fast CPU-GPU Transfer (PIN_MEMORY): {pin_memory}")
        self.logger.info(f"  - Batch Pre-loading (PREFETCH_FACTOR): {prefetch_factor}")
        self.logger.info(f"  - Persistent Workers: {persistent_workers}")
        
        seed = int(task['hyperparams'].get('SEED', 2000))
        training_method = task['hyperparams'].get('TRAINING_METHOD', 'Sequence-to-Sequence')
        model_type = task['hyperparams'].get('MODEL_TYPE', 'LSTM')
        job_folder_path = self.job_manager.get_job_folder()

        if model_type in ['LSTM', 'GRU'] and training_method == "Sequence-to-Sequence":
            try:
                train_X, train_y, _ = self.data_loader_service.get_processed_data(
                    os.path.join(job_folder_path, 'train_data', 'processed_data'),
                    training_method, feature_cols, target_col,
                    lookback=int(task['data_loader_params'].get('lookback', 50)),
                    model_type=model_type
                )
                val_X, val_y, _ = self.data_loader_service.get_processed_data(
                    os.path.join(job_folder_path, 'val_data', 'processed_data'),
                    training_method, feature_cols, target_col,
                    lookback=int(task['data_loader_params'].get('lookback', 50)),
                    model_type=model_type
                )

                total_size_mb = (train_X.nbytes + train_y.nbytes + val_X.nbytes + val_y.nbytes) / (1024 * 1024)
                self.logger.info(f"Accurate total processed data size for RNN: {total_size_mb:.2f} MB")

                if total_size_mb > 300 and sys.platform == "win32":
                    self.logger.warning(f"Data size ({total_size_mb:.2f} MB) exceeds 300 MB on Windows. Forcing num_workers=0 to avoid shared memory errors.")
                    num_workers = 0
                    prefetch_factor = None
                    persistent_workers = False  # Must be False when num_workers=0
                elif total_size_mb > 300:
                    self.logger.warning(f"Data size ({total_size_mb:.2f} MB) is large. Proceeding with user-defined num_workers={num_workers} on non-Windows OS.")
                    # Keep the prefetch_factor and persistent_workers from combined_params (already set above)

                train_loader = self.data_loader_service._create_loader_from_tensors(
                    train_X, train_y, int(task['data_loader_params'].get('batch_size', 32)),
                    num_workers, True, "train", pin_memory=pin_memory, prefetch_factor=prefetch_factor, persistent_workers=persistent_workers
                )
                val_loader = self.data_loader_service._create_loader_from_tensors(
                    val_X, val_y, int(task['data_loader_params'].get('batch_size', 32)),
                    num_workers, False, "validation", pin_memory=pin_memory, prefetch_factor=prefetch_factor, persistent_workers=persistent_workers
                )

            except Exception as e:
                self.logger.error(f"Error during RNN data size calculation: {e}", exc_info=True)
                num_workers = 0
                prefetch_factor = None
                persistent_workers = False
                train_loader, val_loader = self.data_loader_service.create_data_loaders_from_separate_folders(
                    job_folder_path=job_folder_path, training_method=training_method, feature_cols=feature_cols,
                    target_col=target_col, batch_size=int(task['data_loader_params'].get('batch_size', 32)),
                    num_workers=num_workers, lookback=int(task['data_loader_params'].get('lookback', 50)),
                    seed=seed, model_type=model_type, create_test_loader=False, pin_memory=pin_memory,
                    prefetch_factor=prefetch_factor, persistent_workers=persistent_workers
                )
        else:
            self.logger.info(f"Using standard data loader creation for model type: {model_type}")
            self.logger.info(f"Following reference code: sequences will be shuffled during training")
            # Use the optimization settings already configured above
            # persistent_workers is True when num_workers > 0, ensuring performance while
            # subprocess-per-task architecture guarantees cleanup
            
            train_loader, val_loader = self.data_loader_service.create_data_loaders_from_separate_folders(
                job_folder_path=job_folder_path, training_method=training_method, feature_cols=feature_cols,
                target_col=target_col, batch_size=int(task['data_loader_params'].get('batch_size', 32)),
                num_workers=num_workers, lookback=int(task['data_loader_params'].get('lookback', 50)),
                concatenate_raw_data=(training_method == 'Whole Sequence' and model_type in ['LSTM', 'GRU']),
                seed=seed, model_type=model_type, create_test_loader=False, pin_memory=pin_memory,
                prefetch_factor=prefetch_factor, persistent_workers=persistent_workers
            )

        try:
            if train_loader and hasattr(train_loader.dataset, 'tensors'):
                train_size_bytes = train_loader.dataset.tensors[0].storage().nbytes()
                val_size_bytes = val_loader.dataset.tensors[0].storage().nbytes()
                train_loader_size_mb = train_size_bytes / (1024 * 1024)
                val_loader_size_mb = val_size_bytes / (1024 * 1024)
                self.logger.info(f"DataLoader memory size: Train={train_loader_size_mb:.2f}MB, Validation={val_loader_size_mb:.2f}MB")
        except Exception as e:
            self.logger.warning(f"Could not calculate DataLoader memory size: {e}")
            
        return train_loader, val_loader

    def run_training(self, task, update_progress_callback, train_loader, val_loader, device):
        """Run the training process for a single task."""
        try:
            self._run_training_loop(task, update_progress_callback, train_loader, val_loader, device)
        except Exception as e:
            # Check if we're using CUDA Graphs and this is any CUDA-related error
            if isinstance(self.training_service, CUDAGraphsTrainingService):
                error_msg = str(e).lower()
                
                # Check for any CUDA-related error that suggests CUDA Graphs incompatibility
                cuda_related_errors = [
                    'cuda', 'gpu', 'device', 'graph', 'capture', 'memory', 'invalid_value', 
                    'synchronize', 'stream', 'kernel', 'driver', 'runtime'
                ]
                
                is_cuda_error = (
                    isinstance(e, (RuntimeError, torch.cuda.OutOfMemoryError)) or
                    any(cuda_term in error_msg for cuda_term in cuda_related_errors) or
                    'torch' in error_msg
                )
                
                if is_cuda_error:
                    self.logger.warning(f"CUDA Graphs training failed for task {task.get('task_id', 'N/A')}: {str(e)}")
                    self.logger.info("Attempting to fall back to standard training with CUDA state reset.")
                    
                    try:
                        # Move model to CPU first to avoid corrupted CUDA state
                        model = task.get('model')
                        if model is not None:
                            model.cpu()
                            self.logger.info("Moved model to CPU to escape corrupted CUDA state")
                        
                        # Complete CUDA state cleanup
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                            torch.cuda.reset_peak_memory_stats()
                        
                        # Force garbage collection
                        import gc
                        gc.collect()
                        
                        # Reload the untrained model from CPU (avoiding corrupted CUDA memory)
                        untrained_model_path = task.get('untrained_model_path')
                        if untrained_model_path and os.path.exists(untrained_model_path):
                            # Load to CPU first, then move to CUDA after cleanup
                            state_dict = torch.load(untrained_model_path, map_location='cpu')
                            model.load_state_dict(state_dict)
                            self.logger.info(f"Reloaded untrained model state from {untrained_model_path} (via CPU)")
                        else:
                            self.logger.warning("Untrained model template not found. Continuing with current model state.")
                        
                        # Now move model back to CUDA with clean state
                        if model is not None:
                            model.to(device)
                            self.logger.info(f"Moved model back to {device} after CUDA reset")
                        
                        # Switch to standard training service
                        self.training_service = TrainingTaskService(device=self.device)
                        self.logger.info("Switched to standard TrainingTaskService with CUDA state reset.")
                        
                        update_progress_callback.emit({'status': 'CUDA Graphs failed. Falling back to standard training...'})
                        
                        # Retry with standard training
                        self._run_training_loop(task, update_progress_callback, train_loader, val_loader, device)
                        
                    except Exception as fallback_error:
                        self.logger.error(f"Fallback to standard training also failed: {fallback_error}")
                        raise fallback_error
                else:
                    # Non-CUDA error with CUDA Graphs - re-raise original error
                    self.logger.error(f"Non-CUDA error occurred during CUDA Graphs training: {str(e)}")
                    raise
            else:
                # Not using CUDA Graphs or different training service - re-raise original error
                raise
        finally:
            best_model_path_final = task.get('training_params', {}).get('best_model_path', 'N/A')
            job_folder_final = self.job_manager.get_job_folder()
            if best_model_path_final != 'N/A' and job_folder_final in best_model_path_final:
                relative_best_model_path = os.path.relpath(best_model_path_final, job_folder_final)
            else:
                relative_best_model_path = best_model_path_final
            self.logger.info(f"Finished run_training attempt for task {task.get('task_id', 'N/A')}. Best model (if saved): {relative_best_model_path}")

    def _run_training_loop(self, task, update_progress_callback, train_loader, val_loader, device):
        """The actual training loop logic."""
        # Proactive cleanup of any existing dataloaders before starting new training task
        try:
            import gc
            gc.collect()  # Initial cleanup
            self.logger.info("Performed initial garbage collection before starting new training task")
        except Exception as initial_cleanup_error:
            self.logger.warning(f"Error during initial cleanup: {initial_cleanup_error}")
        
        # Helper function to check for stop signal from progress queue
        def check_for_stop_signal(progress_queue):
            """Check if GUI sent stop signal via queue."""
            if progress_queue is None:
                return False
            try:
                import queue as pyqueue
                while True:
                    try:
                        msg = progress_queue.get_nowait()
                        if isinstance(msg, dict) and msg.get('stop_signal'):
                            self.logger.info("[SUBPROCESS] Received graceful stop signal from GUI")
                            return True
                        # Put back any other messages
                        progress_queue.put(msg)
                        break
                    except pyqueue.Empty:
                        break
            except Exception as e:
                self.logger.warning(f"Error checking stop signal: {e}")
            return False
        
        self.logger.info(f"--- Starting _run_training_loop for task: {task['task_id']} ---")
        self.logger.info(f"--- Starting run_training for task: {task['task_id']} ---") # Added detailed log
        self.logger.info("Starting training loop")
        
        # Log the device being used for training
        device_str = str(device)
        if device_str == 'cpu':
            device_display = 'CPU'
        elif 'cuda' in device_str:
            try:
                if torch.cuda.is_available() and 'cuda' in device_str:
                    gpu_idx = int(device_str.split(':')[1]) if ':' in device_str else 0
                    gpu_name = torch.cuda.get_device_name(gpu_idx)
                    device_display = f"{device_str.upper()} ({gpu_name})"
                else:
                    device_display = device_str.upper()
            except Exception:
                device_display = device_str.upper()
        else:
            device_display = device_str
        
        self.logger.info(f"Training will be executed on device: {device_display}")
        
        # Send training start device log message to GUI
        update_progress_callback.emit({
            'training_start_log_message': f"<span style='color: #0b6337;'><b>Training started on device:</b> {device_display}</span>"
        })
        
        # --- GATHER METADATA FOR LOGGING ---
        job_folder = self.job_manager.get_job_folder()
        data_ref_path = os.path.join(job_folder, 'data_files_reference.json')
        if os.path.exists(data_ref_path):
            with open(data_ref_path, 'r') as f:
                data_ref = json.load(f)
            training_samples = len(data_ref.get('train_files', []))
            validation_samples = len(data_ref.get('val_files', []))
        else:
            training_samples = 'N/A'
            validation_samples = 'N/A'

        model_total_params = sum(p.numel() for p in task['model'].parameters())
        mixed_precision = task['hyperparams'].get('USE_MIXED_PRECISION', False)
        cuda_graph_enabled = isinstance(self.training_service, CUDAGraphsTrainingService)
        # Pull optimizer type from hyperparams (may be a CSV if grid search; take the first)
        raw_opt = task.get('hyperparams', {}).get('OPTIMIZER_TYPE', 'Adam')
        optimizer_type = raw_opt.split(',')[0].strip() if isinstance(raw_opt, str) else str(raw_opt)
        
        model_type = task['hyperparams'].get('MODEL_TYPE', 'LSTM')
        training_method = task['hyperparams'].get('TRAINING_METHOD', 'Sequence-to-Sequence')
        batch_size = int(task['hyperparams'].get('BATCH_SIZE', 0))
        lookback_val = task['hyperparams'].get('LOOKBACK', 0)
        if lookback_val == 'N/A':
            lookback = 0
        else:
            lookback = int(lookback_val)

        if model_type in ['LSTM', 'GRU'] and training_method == 'Sequence-to-Sequence':
            samples_per_batch = batch_size * lookback
        else:
            samples_per_batch = batch_size

        device_str = str(self.device)
        if device_str == 'cpu':
            device_name = 'CPU'
        elif 'cuda' in device_str and torch.cuda.is_available():
            try:
                gpu_idx = int(device_str.split(':')[1]) if ':' in device_str else 0
                device_name = torch.cuda.get_device_name(gpu_idx)
            except Exception:
                device_name = device_str.upper()
        else:
            device_name = device_str.upper()

        metadata = {
            "training_samples": training_samples,
            "validation_samples": validation_samples,
            "device": str(self.device),
            "device_name": device_name,
            "model_type": model_type,
            "batch_size": batch_size,
            "samples_per_batch": samples_per_batch,
            "model_total_params": model_total_params,
            "mixed_precision": mixed_precision,
            "cuda_graph_enabled": cuda_graph_enabled,
            "optimizer_type": optimizer_type,
        }
        # --- END METADATA GATHERING ---

        # Initialize/reset task-specific best original scale validation RMSE tracker
        # Using a unique attribute name per task to avoid conflicts if manager instance is reused for different tasks sequentially
        # though typically a new manager or thread might be used. This is safer.
        setattr(self, f'_task_{task["task_id"]}_best_val_rmse_orig', float('inf'))
        setattr(self, f'_task_{task["task_id"]}_best_train_rmse_orig', float('inf'))  # Initialize best training loss attribute
        best_train_loss_norm = float('inf')
        best_train_loss_denorm = float('inf')

        # Lists to store epoch-wise loss history
        train_loss_history = []
        val_loss_history = []

        hyperparams = self.convert_hyperparams(task['hyperparams']) # This ensures BATCH_SIZE is int if it exists
        model = task['model']
        
        # Reload model from the untrained template to ensure a clean slate for each repetition
        model_path = task.get('model_path')
        if model_path and os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            self.logger.info(f"Reloaded untrained model state from {model_path} for task {task['task_id']}")
        else:
            self.logger.warning(f"Untrained model template not found at {model_path}. Model may not be reset for new repetition.")
        
        model = model.to(device)
        
        # Ensure BATCH_SIZE from hyperparams (which might be the string from QLineEdit) is correctly converted and available
        # The actual batch size used by train_loader is now determined by DataLoaderService based on use_full_train_batch_flag
        # However, other parts of the code might still refer to hyperparams['BATCH_SIZE']
        max_epochs = hyperparams['MAX_EPOCHS']
        valid_freq = hyperparams.get('VALID_FREQUENCY', 1)
        valid_patience = hyperparams['VALID_PATIENCE']
        current_patience = valid_patience # Initialize current patience
        #patience_threshold = int(valid_patience * 0.5)
        current_lr = hyperparams['INITIAL_LR']
        
        # Only access StepLR parameters if StepLR scheduler is being used
        scheduler_type = hyperparams.get('SCHEDULER_TYPE', 'None')
        if scheduler_type == 'StepLR':
            lr_drop_period = hyperparams.get('LR_DROP_PERIOD', 10)
            lr_drop_factor = hyperparams.get('LR_DROP_FACTOR', 0.5)
        else:
            lr_drop_period = 10  # Default values for backward compatibility
            lr_drop_factor = 0.5
            
        # Define a buffer period after which LR drops can happen again, e.g., 100 epochs.
        lr_drop_buffer = 50
        last_lr_drop_epoch = 0  # Initialize the epoch of the last LR drop
        # weight_decay = hyperparams.get('WEIGHT_DECAY', 1e-5)

        best_validation_loss = float('inf')
        patience_counter = 0
        loop_start_time = time.time() # Renamed from start_time to avoid confusion with overall training start
        last_validation_time = loop_start_time
        early_stopping = False  # Initialize early stopping flag
        exploit_mode = 0 # Initialize exploit mode counter

        # Max training time logic
        max_training_time_seconds = int(task.get('training_params', {}).get('max_training_time_seconds', 0))
        overall_training_start_time = time.time() # For max training time check
        self.logger.info(f"Max training time set to: {max_training_time_seconds} seconds.")

        # Check if CUDA Graphs will be used for this task
        model_type = task.get('model_metadata', {}).get('model_type', task.get('hyperparams', {}).get('MODEL_TYPE', 'LSTM'))
        use_cuda_graphs = (
            hasattr(self.training_service, 'train_epoch_with_graphs') and 
            device.type == 'cuda' and 
            model_type == 'FNN'
        )

        # Resolve weight decay (string from GUI -> float)
        try:
            weight_decay = float(hyperparams.get('WEIGHT_DECAY', 0.0))
        except Exception:
            weight_decay = 0.0

        if use_cuda_graphs:
            # CUDA Graphs requires capturable=True; both Adam and AdamW support it on recent PyTorch
            if optimizer_type.lower() == 'adamw':
                try:
                    self.optimizer = torch.optim.AdamW(model.parameters(), lr=current_lr, weight_decay=weight_decay, capturable=True)
                except TypeError:
                    self.optimizer = torch.optim.AdamW(model.parameters(), lr=current_lr, weight_decay=weight_decay)
            else:
                try:
                    self.optimizer = torch.optim.Adam(model.parameters(), lr=current_lr, weight_decay=weight_decay, capturable=True)
                except TypeError:
                    self.optimizer = torch.optim.Adam(model.parameters(), lr=current_lr, weight_decay=weight_decay)
            self.logger.info(f"Created CUDA Graphs-compatible optimizer ({optimizer_type}) for {model_type} model")
        else:
            # Standard optimizer for non-CUDA graphs training
            if optimizer_type.lower() == 'adamw':
                self.optimizer = torch.optim.AdamW(model.parameters(), lr=current_lr, weight_decay=weight_decay)
            else:
                self.optimizer = torch.optim.Adam(model.parameters(), lr=current_lr, weight_decay=weight_decay)
            self.logger.info(f"Created standard optimizer ({optimizer_type}) for {model_type} model")

        # Create scheduler based on type
        scheduler_type = hyperparams.get("SCHEDULER_TYPE", "StepLR")
        
        if scheduler_type == "StepLR":
            main_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=lr_drop_period,  # Number of epochs between drops
                gamma=lr_drop_factor       # Multiplicative factor for the drop
            )
        elif scheduler_type == "ReduceLROnPlateau":
            plateau_patience = int(hyperparams.get("PLATEAU_PATIENCE", 10))
            plateau_factor = float(hyperparams.get("PLATEAU_FACTOR", 0.1))
            main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',           # Reduce when metric stops decreasing
                factor=plateau_factor,
                patience=plateau_patience
            )
        elif scheduler_type == "CosineAnnealingWarmRestarts":
            cosine_t0 = int(hyperparams.get("COSINE_T0", 10))
            cosine_t_mult = int(hyperparams.get("COSINE_T_MULT", 2))
            cosine_eta_min = float(hyperparams.get("COSINE_ETA_MIN", 1e-6))
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=cosine_t0,
                T_mult=cosine_t_mult,
                eta_min=cosine_eta_min
            )
            self.logger.info(f"Created CosineAnnealingWarmRestarts scheduler: T_0={cosine_t0}, T_mult={cosine_t_mult}, eta_min={cosine_eta_min}")
        else:
            # Default fallback to StepLR
            main_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=lr_drop_period,
                gamma=lr_drop_factor
            )
            self.logger.warning(f"Unknown scheduler type '{scheduler_type}', defaulting to StepLR")
        scheduler = main_scheduler  # Start with the main scheduler
        optimizer = self.optimizer

        # Initialize CSV logging for epoch-wise data
        csv_log_file = task.get('csv_log_file')
        if csv_log_file:
            # Ensure the directory for csv_log_file exists (it's task_dir/logs/)
            os.makedirs(os.path.dirname(csv_log_file), exist_ok=True)
            with open(csv_log_file, 'w', newline='') as f:
                # Write metadata as commented header
                for key, value in metadata.items():
                    f.write(f"# {key}: {value}\n")
                
                csv_writer = csv.writer(f)
                header = [
                    "epoch", "train_loss_norm", "val_loss_norm", "best_val_loss_norm", 
                    "learning_rate", "elapsed_time_sec", "avg_batch_time_sec", 
                    "patience_counter", "model_memory_mb", "time_per_epoch_sec"
                ]
                csv_writer.writerow(header)

        # Training loop
        for epoch in range(1, max_epochs + 1):
            # Check for graceful stop signal from GUI first
            if hasattr(self, 'progress_queue') and check_for_stop_signal(self.progress_queue):
                self.logger.info("[SUBPROCESS] Graceful stop requested by GUI")
                self.stop_requested = True
            
            if self.stop_requested:
                self.logger.info("Training stopped by user")
                break
            
            # Check for max training time exceeded
            if max_training_time_seconds > 0:
                current_training_duration = time.time() - overall_training_start_time
                if current_training_duration > max_training_time_seconds:
                    self.logger.info(f"Max training time ({max_training_time_seconds}s) exceeded. Stopping training.")
                    self.stop_requested = True # Use existing flag to gracefully stop
                    early_stopping = True # Indicate it was a form of early stop
                    # Also update task results if possible here or after loop
                    task['results']['early_stopped_reason'] = 'Max training time exceeded'
                    break # Exit epoch loop

            # Initialize hidden states for training phase (only for RNN models)
            # Use the actual batch size from the train_loader
            actual_train_batch_size = train_loader.batch_size
            if actual_train_batch_size is None:
                self.logger.warning(f"train_loader.batch_size is None. Falling back to hyperparams BATCH_SIZE ({hyperparams.get('BATCH_SIZE', 'N/A')}) for training hidden state init.")
                actual_train_batch_size = int(hyperparams.get('BATCH_SIZE', 32)) # Default if all else fails

            # Initialize hidden states only for RNN models (LSTM, GRU)
            h_s, h_c = None, None
            model_type = task.get('model_metadata', {}).get('model_type', task.get('hyperparams', {}).get('MODEL_TYPE', 'LSTM'))
            if model_type in ['LSTM', 'GRU']:
                self.logger.info(f"Initializing {model_type} hidden state with batch size: {actual_train_batch_size}")
                h_s = torch.zeros(model.num_layers, actual_train_batch_size, model.hidden_units).to(device)
                if model_type == 'LSTM':
                    h_c = torch.zeros(model.num_layers, actual_train_batch_size, model.hidden_units).to(device)
            else:
                self.logger.info(f"Model type {model_type} does not require hidden state initialization")

            # Measure time for the training loop
            epoch_start_time = time.time()

            # Train the model for one epoch
            model_type = task.get('model_metadata', {}).get('model_type', task.get('hyperparams', {}).get('MODEL_TYPE', 'LSTM')) # Get model_type
            # train_epoch now returns: avg_batch_time, avg_loss (normalized), all_train_y_pred_normalized, all_train_y_true_normalized
            verbose = task.get('verbose', True)
            
            # Use CUDA Graphs if available and enabled (only for FNN models on CUDA)
            if (hasattr(self.training_service, 'train_epoch_with_graphs') and 
                device.type == 'cuda' and model_type == 'FNN'):
                
                # Check if batch size has changed and reset CUDA graphs if necessary
                current_batch_size = train_loader.batch_size
                if (self.previous_batch_size is not None and 
                    current_batch_size != self.previous_batch_size):
                    self.logger.info(f"Batch size changed from {self.previous_batch_size} to {current_batch_size}. Resetting CUDA Graphs.")
                    self.training_service.reset_cuda_graphs()
                    update_progress_callback.emit({
                        'training_start_log_message': f"<span style='color: #ff6600;'><b>CUDA Graphs reset for new batch size: {current_batch_size}</b></span>"
                    })
                
                # Enable CUDA graphs on first epoch or after reset
                if epoch == 1 or (self.previous_batch_size is not None and current_batch_size != self.previous_batch_size):
                    use_mixed_precision = task['hyperparams'].get('USE_MIXED_PRECISION', False)
                    graphs_enabled = self.training_service.enable_cuda_graphs(device, use_mixed_precision)
                    if graphs_enabled:
                        # Send log message about CUDA Graphs
                        update_progress_callback.emit({
                            'training_start_log_message': f"<span style='color: #ff6600;'><b>CUDA Graphs enabled!</b></span>"
                        })
                
                # Update the previous batch size for future tasks
                self.previous_batch_size = current_batch_size
                
                avg_batch_time, train_loss_norm, epoch_train_preds_norm, epoch_train_trues_norm = self.training_service.train_epoch_with_graphs(
                    model, train_loader, optimizer, epoch, device, self.stop_requested, task, verbose=verbose
                )
            else:
                # Standard training
                avg_batch_time, train_loss_norm, epoch_train_preds_norm, epoch_train_trues_norm = self.training_service.train_epoch(
                    model, model_type, train_loader, optimizer, h_s, h_c, epoch, device, self.stop_requested, task, verbose=verbose
                )
            
            train_loss_history.append(train_loss_norm)

            if train_loss_norm < best_train_loss_norm:
                best_train_loss_norm = train_loss_norm

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            formatted_epoch_time = format_time(epoch_duration)

            # Check stop signal immediately after training phase
            if hasattr(self, 'progress_queue') and check_for_stop_signal(self.progress_queue):
                self.logger.info("[SUBPROCESS] Stop requested after training phase")
                self.stop_requested = True
            
            if self.stop_requested:
                self.logger.info("Training stopped by user after training phase")
                break

            # Only validate at specified frequency
            if epoch == 1 or epoch % valid_freq == 0 or epoch == max_epochs:
                # Initialize hidden states for validation phase (only for RNN models)
                # Use the actual batch size from the val_loader
                actual_val_batch_size = val_loader.batch_size
                if actual_val_batch_size is None:
                    self.logger.warning(f"val_loader.batch_size is None. Falling back to hyperparams BATCH_SIZE ({hyperparams.get('BATCH_SIZE', 'N/A')}) for validation hidden state init.")
                    actual_val_batch_size = int(hyperparams.get('BATCH_SIZE', 32)) # Default if all else fails
                
                # Initialize hidden states only for RNN models (LSTM, GRU)
                h_s_val, h_c_val = None, None
                model_type = task.get('model_metadata', {}).get('model_type', task.get('hyperparams', {}).get('MODEL_TYPE', 'LSTM'))
                if model_type in ['LSTM', 'GRU', 'LSTM_EMA', 'LSTM_LPF']:
                    self.logger.info(f"Initializing {model_type} validation hidden state with batch size: {actual_val_batch_size}")
                    h_s_val = torch.zeros(model.num_layers, actual_val_batch_size, model.hidden_units).to(device)
                    if model_type in ['LSTM', 'LSTM_EMA', 'LSTM_LPF']:
                        h_c_val = torch.zeros(model.num_layers, actual_val_batch_size, model.hidden_units).to(device)
                else:
                    self.logger.info(f"Model type {model_type} does not require validation hidden state initialization")

                model_type = task.get('model_metadata', {}).get('model_type', task.get('hyperparams', {}).get('MODEL_TYPE', 'LSTM')) # Get model_type
                # validate_epoch now returns: avg_loss (normalized), all_val_y_pred_normalized, all_val_y_true_normalized
                verbose = task.get('verbose', True)
                
                # Use CUDA Graphs validation if available
                if (hasattr(self.training_service, 'validate_epoch_with_graphs') and 
                    device.type == 'cuda' and model_type == 'FNN'):
                    val_loss_norm, epoch_val_preds_norm, epoch_val_trues_norm = self.training_service.validate_epoch_with_graphs(
                        model, val_loader, epoch, device, self.stop_requested, task, verbose=verbose
                    )
                else:
                    # Standard validation
                    val_loss_norm, epoch_val_preds_norm, epoch_val_trues_norm = self.training_service.validate_epoch(
                        model, model_type, val_loader, h_s_val, h_c_val, epoch, device, self.stop_requested, task, verbose=verbose
                    )
                
                val_loss_history.append(val_loss_norm)

                current_time = time.time()
                elapsed_time = current_time - loop_start_time # Use loop_start_time for per-epoch/validation cycle timing
                delta_t_epoch = (current_time - last_validation_time) / valid_freq
                last_validation_time = current_time

                current_lr = optimizer.param_groups[0]['lr']
                
                if val_loss_norm < best_validation_loss: # best_validation_loss is also on normalized scale
                    best_validation_loss = val_loss_norm
                    
                    # Save to best_model_path
                    best_model_save_path = task.get('training_params', {}).get('best_model_path')
                    if best_model_save_path and task.get('training_params', {}).get('save_best_model', True):
                        self.save_model(task, save_path=best_model_save_path)
                        self.logger.info(f"Best model saved to: {best_model_save_path}")
                    elif not task.get('training_params', {}).get('save_best_model', True):
                        self.logger.info("Model saving is disabled for this task (e.g., Optuna trial).")
                    else:
                        self.logger.warning(f"best_model_path not found in task for epoch {epoch}. Best model not saved.")
                    
                    patience_counter = 0
                    
                    # If we find a better model during exploit mode, revert to the main training phase
                    # to capitalize on the new discovery with the original training strategy.
                    if exploit_mode > 0:
                        self.logger.info(f"New best model found during exploit mode. Reverting to main training phase.")
                        exploit_mode = 0  # Exit exploit mode
                        scheduler = main_scheduler  # Revert to the main scheduler
                        current_patience = valid_patience  # Revert to main patience
                        # Reset learning rate to initial value
                        current_lr = hyperparams["INITIAL_LR"]
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = current_lr
                        self.logger.info(f"Optimizer LR reset to {current_lr} after finding new best model.")
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
                train_rmse_for_gui = float('nan')
                val_rmse_for_gui = float('nan')
                # Retrieve the running best original scale validation RMSE for this task
                best_val_rmse_orig_scale_for_gui = getattr(self, f'_task_{task["task_id"]}_best_val_rmse_orig', float('inf'))

                target_col_for_scaler = task['data_loader_params']['target_column']

                if self.loaded_scaler and target_col_for_scaler in self.scaler_metadata.get('normalized_columns', []):
                    from vestim.services.data_processor.src import normalization_service # Local import
                    import pandas as pd # Local import for DataFrame
                    import numpy as np # Ensure numpy is imported

                    # --- Train RMSE on original scale (if epoch_train_preds_norm available) ---
                    if epoch_train_preds_norm is not None and epoch_train_trues_norm is not None and len(epoch_train_preds_norm) > 0:
                        try:
                            # Ensure tensors are on CPU and converted to numpy
                            e_t_p_n_cpu = epoch_train_preds_norm.cpu().numpy() if epoch_train_preds_norm.is_cuda else epoch_train_preds_norm.numpy()
                            e_t_t_n_cpu = epoch_train_trues_norm.cpu().numpy() if epoch_train_trues_norm.is_cuda else epoch_train_trues_norm.numpy()

                            # Use the safer single-column denormalization method
                            # Temporarily suppress UserWarning for this calculation
                            import warnings
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", UserWarning)
                                train_pred_orig = normalization_service.inverse_transform_single_column(
                                    e_t_p_n_cpu, self.loaded_scaler, target_col_for_scaler, self.scaler_metadata['normalized_columns']
                                )
                                train_true_orig = normalization_service.inverse_transform_single_column(
                                    e_t_t_n_cpu, self.loaded_scaler, target_col_for_scaler, self.scaler_metadata['normalized_columns']
                                )
                            
                            train_mse_orig = np.mean((train_pred_orig - train_true_orig)**2)
                            train_rmse_for_gui = np.sqrt(train_mse_orig) * multiplier
                            if train_rmse_for_gui < best_train_loss_denorm:
                                best_train_loss_denorm = train_rmse_for_gui
                                setattr(self, f'_task_{task["task_id"]}_best_train_rmse_orig', best_train_loss_denorm)  # Store best training loss
                        except Exception as e_inv_train:
                            self.logger.error(f"Error during inverse transform for training data (epoch {epoch}): {e_inv_train}. Falling back for train_rmse_for_gui.")
                            if train_loss_norm is not None and not math.isnan(train_loss_norm):
                                train_rmse_for_gui = math.sqrt(max(0, train_loss_norm)) * multiplier
                                if train_rmse_for_gui < best_train_loss_denorm:
                                    best_train_loss_denorm = train_rmse_for_gui
                                    setattr(self, f'_task_{task["task_id"]}_best_train_rmse_orig', best_train_loss_denorm)  # Store best training loss
                    else:
                         if train_loss_norm is not None and not math.isnan(train_loss_norm):
                            train_rmse_for_gui = math.sqrt(max(0, train_loss_norm)) * multiplier
                            if train_rmse_for_gui < best_train_loss_denorm:
                                best_train_loss_denorm = train_rmse_for_gui
                                setattr(self, f'_task_{task["task_id"]}_best_train_rmse_orig', best_train_loss_denorm)  # Store best training loss
                    
                    # --- Validation RMSE on original scale ---
                    if epoch_val_preds_norm is not None and epoch_val_trues_norm is not None and len(epoch_val_preds_norm) > 0:
                        try:
                            e_v_p_n_cpu = epoch_val_preds_norm.cpu().numpy() if epoch_val_preds_norm.is_cuda else epoch_val_preds_norm.numpy()
                            e_v_t_n_cpu = epoch_val_trues_norm.cpu().numpy() if epoch_val_trues_norm.is_cuda else epoch_val_trues_norm.numpy()

                            # Use the safer single-column denormalization method
                            # Temporarily suppress UserWarning for this calculation
                            import warnings
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", UserWarning)
                                val_pred_orig = normalization_service.inverse_transform_single_column(
                                    e_v_p_n_cpu, self.loaded_scaler, target_col_for_scaler, self.scaler_metadata['normalized_columns']
                                )
                                val_true_orig = normalization_service.inverse_transform_single_column(
                                    e_v_t_n_cpu, self.loaded_scaler, target_col_for_scaler, self.scaler_metadata['normalized_columns']
                                )

                            val_mse_orig = np.mean((val_pred_orig - val_true_orig)**2)
                            current_val_rmse_orig_scale = np.sqrt(val_mse_orig) * multiplier
                            val_rmse_for_gui = current_val_rmse_orig_scale
                            
                            if current_val_rmse_orig_scale < best_val_rmse_orig_scale_for_gui:
                                best_val_rmse_orig_scale_for_gui = current_val_rmse_orig_scale
                                setattr(self, f'_task_{task["task_id"]}_best_val_rmse_orig', best_val_rmse_orig_scale_for_gui) # Update the stored best
                            
                        except Exception as e_inv_val:
                            self.logger.error(f"Error during inverse transform for validation data (epoch {epoch}): {e_inv_val}. Falling back for val_rmse_for_gui.")
                            if val_loss_norm is not None and not math.isnan(val_loss_norm):
                                val_rmse_for_gui = math.sqrt(max(0, val_loss_norm)) * multiplier
                    else:
                        if val_loss_norm is not None and not math.isnan(val_loss_norm):
                            val_rmse_for_gui = math.sqrt(max(0, val_loss_norm)) * multiplier
                    
                    # Use the overall best original-scale validation RMSE for this task for display
                    best_val_rmse_for_gui = best_val_rmse_orig_scale_for_gui

                else: # No scaler loaded or target not in normalized columns - use normalized loss for GUI RMSE
                    if train_loss_norm is not None and not math.isnan(train_loss_norm):
                        train_rmse_for_gui = math.sqrt(max(0, train_loss_norm)) * multiplier
                        if train_rmse_for_gui < best_train_loss_denorm:
                           best_train_loss_denorm = train_rmse_for_gui
                           setattr(self, f'_task_{task["task_id"]}_best_train_rmse_orig', best_train_loss_denorm)  # Store best training loss
                    if val_loss_norm is not None and not math.isnan(val_loss_norm):
                        val_rmse_for_gui = math.sqrt(max(0, val_loss_norm)) * multiplier
                        # Update the best validation RMSE when no scaler is used
                        if val_rmse_for_gui < best_val_rmse_orig_scale_for_gui:
                            best_val_rmse_orig_scale_for_gui = val_rmse_for_gui
                            setattr(self, f'_task_{task["task_id"]}_best_val_rmse_orig', best_val_rmse_orig_scale_for_gui)
                    # If no scaler, best_val_rmse_for_gui is based on best_validation_loss (normalized)
                    if best_validation_loss != float('inf') and not math.isnan(best_validation_loss):
                         best_val_rmse_for_gui = math.sqrt(max(0, best_validation_loss)) * multiplier
                         # Ensure we update the stored best validation loss for non-scaler case
                         if best_val_rmse_for_gui < getattr(self, f'_task_{task["task_id"]}_best_val_rmse_orig', float('inf')):
                             setattr(self, f'_task_{task["task_id"]}_best_val_rmse_orig', best_val_rmse_for_gui)
                    else:
                         best_val_rmse_for_gui = float('inf') # Ensure it's inf if best_validation_loss is inf
                
                # Log to CSV (after validation)
                model_memory_usage_val = torch.cuda.memory_allocated(device=self.device) if self.device.type == 'cuda' else 0
                model_memory_usage_mb_val = model_memory_usage_val / (1024 * 1024) if model_memory_usage_val > 0 else 0
                if csv_log_file:
                    with open(csv_log_file, 'a', newline='') as f:
                        csv_writer_val = csv.writer(f)
                        csv_writer_val.writerow([
                            epoch,
                            f"{train_loss_norm:.8f}" if train_loss_norm is not None else 'nan',
                            f"{val_loss_norm:.8f}" if val_loss_norm is not None else 'nan',
                            f"{best_validation_loss:.8f}" if best_validation_loss is not None else 'nan',
                            f"{current_lr:.6e}" if current_lr is not None else 'nan',
                            f"{elapsed_time:.4f}" if elapsed_time is not None else 'nan',
                            f"{avg_batch_time:.6f}" if avg_batch_time is not None else 'nan',
                            patience_counter if patience_counter is not None else 'nan',
                            f"{model_memory_usage_mb_val:.4f}" if model_memory_usage_mb_val is not None else 'nan',
                            f"{epoch_duration:.4f}"
                        ])
                
                self.logger.info(f"Epoch {epoch} | Train Loss (Norm): {train_loss_norm:.6f} | Val Loss (Norm): {val_loss_norm:.6f} | GUI Train RMSE: {train_rmse_for_gui:.4f} {error_unit_label} | GUI Val RMSE: {val_rmse_for_gui:.4f} {error_unit_label} | LR: {current_lr} | Epoch Time: {formatted_epoch_time} | Best Val Loss (Norm): {best_validation_loss:.6f} | GUI Best Val RMSE: {best_val_rmse_for_gui:.4f} {error_unit_label} | Patience: {patience_counter}")
                
                # --- Debug logging for validation loss normalization issue ---
                self.logger.debug(f"Validation loss debug - Epoch {epoch}:")
                self.logger.debug(f"  - val_loss_norm (MSE in normalized space): {val_loss_norm}")
                self.logger.debug(f"  - Target column: {target_col_for_scaler}")
                self.logger.debug(f"  - Multiplier: {multiplier}")
                self.logger.debug(f"  - Scaler loaded: {hasattr(self, 'loaded_scaler') and self.loaded_scaler is not None}")
                if hasattr(self, 'scaler_metadata') and self.scaler_metadata:
                    self.logger.debug(f"  - Target in normalized columns: {target_col_for_scaler in self.scaler_metadata.get('normalized_columns', [])}")
                
                # Show what the fallback calculation would be (this is often wrong!)
                if val_loss_norm is not None and not math.isnan(val_loss_norm):
                    fallback_rmse = math.sqrt(max(0, val_loss_norm)) * multiplier
                    self.logger.debug(f"  - Fallback RMSE calculation: sqrt({val_loss_norm}) * {multiplier} = {fallback_rmse:.2f}")
                    self.logger.debug(f"    WARNING: This fallback method is often INCORRECT for denormalized values!")
                
                # Calculate task-level elapsed time
                task_elapsed_time = time.time() - self.task_start_time if hasattr(self, 'task_start_time') else elapsed_time
                
                # Calculate job-level elapsed time
                job_elapsed_time = time.time() - self.job_start_time if hasattr(self, 'job_start_time') and self.job_start_time is not None else 0
                
                progress_data = {
                    'epoch': epoch,
                    'train_loss': train_loss_norm,
                    'val_loss': val_loss_norm,
                    'train_rmse_scaled': train_rmse_for_gui,
                    'val_rmse_scaled': val_rmse_for_gui,
                    'error_unit_label': error_unit_label,
                    'elapsed_time': elapsed_time,
                    'task_elapsed_time': task_elapsed_time,  # Total task time
                    'job_elapsed_time': job_elapsed_time,   # Total job time (all tasks)
                    'formatted_task_time': format_time(task_elapsed_time),  # Formatted task time
                    'formatted_job_time': format_time_long(job_elapsed_time),  # Formatted job time
                    'delta_t_epoch': formatted_epoch_time,
                    'learning_rate': current_lr,
                    'best_val_loss': best_validation_loss, # This is normalized best MSE
                    'best_val_rmse_scaled': best_val_rmse_for_gui,
                    'patience_counter': patience_counter,
                }
                setattr(self, f'_task_{task["task_id"]}_last_train_rmse_orig', train_rmse_for_gui)
                setattr(self, f'_task_{task["task_id"]}_last_val_rmse_orig', val_rmse_for_gui)
                update_progress_callback.emit(progress_data)
                self.logger.info(f"GUI updated for validation epoch {epoch} (ValidFreq={valid_freq})")
                
                # Check for stop signal after validation completes
                if hasattr(self, 'progress_queue') and check_for_stop_signal(self.progress_queue):
                    self.logger.info("[SUBPROCESS] Graceful stop requested after validation")
                    self.stop_requested = True
                
                if self.stop_requested:
                    self.logger.info("Training stopped by user after validation phase.")
                    break

                if patience_counter > current_patience:
                    exploit_repetitions = hyperparams["EXPLOIT_REPETITIONS"]
                    if exploit_mode < exploit_repetitions:
                        exploit_mode += 1
                        exploit_epochs = hyperparams["EXPLOIT_EPOCHS"]
                        patience_counter = 0  # Reset patience counter
                        current_patience = exploit_epochs
                        
                        self.logger.info(f"Patience reached at epoch {epoch}. Entering exploit mode iteration {exploit_mode}/{exploit_repetitions}.")
                        self.logger.info(f"Patience counter reset to 0. Current patience set to {current_patience} for exploit phase.")
                        update_progress_callback.emit({'status': f"Patience reached. Exploiting for {exploit_repetitions} reps with patience {exploit_epochs} (Rep {exploit_mode})"})

                        # Load the best model state from memory or disk
                        best_model_path = task.get('training_params', {}).get('best_model_path')
                        if best_model_path and os.path.exists(best_model_path):
                            checkpoint = torch.load(best_model_path, map_location=device)
                            if 'model_state_dict' in checkpoint:
                                model.load_state_dict(checkpoint['model_state_dict'])
                                
                                # Check if we're using CUDA Graphs - if so, DON'T reload optimizer state
                                use_cuda_graphs = (
                                    hasattr(self.training_service, 'train_epoch_with_graphs') and 
                                    device.type == 'cuda' and 
                                    model_type == 'FNN'
                                )
                                
                                if use_cuda_graphs:
                                    # CUDA Graphs: Load optimizer state first, then reset graphs
                                    if 'optimizer_state_dict' in checkpoint:
                                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                                        self.logger.info(f"Loaded best model and optimizer state_dict from {best_model_path}")
                                        
                                        # Set the learning rate to the exploit_lr value  
                                        current_lr = hyperparams["EXPLOIT_LR"]
                                        for param_group in optimizer.param_groups:
                                            param_group['lr'] = current_lr
                                        self.logger.info(f"Optimizer LR updated to {current_lr} for exploit mode.")
                                    else:
                                        self.logger.info(f"Optimizer state not found in checkpoint. Using current optimizer state.")
                                        current_lr = hyperparams["EXPLOIT_LR"]
                                        for param_group in optimizer.param_groups:
                                            param_group['lr'] = current_lr
                                        self.logger.info(f"Optimizer LR set to {current_lr} for exploit mode.")
                                    
                                    # Reset CUDA graphs AFTER loading states - graphs will be re-captured with loaded optimizer
                                    if hasattr(self.training_service, 'reset_cuda_graphs'):
                                        self.training_service.reset_cuda_graphs()
                                        self.logger.info("CUDA graphs reset - will be re-captured for exploit phase with loaded optimizer state")
                                elif 'optimizer_state_dict' in checkpoint:
                                    # Standard training: Safe to reload optimizer state
                                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                                    self.logger.info(f"Loaded best model and optimizer state_dict from {best_model_path}")
                                    # Set the learning rate to the exploit_lr value
                                    current_lr = hyperparams["EXPLOIT_LR"]
                                    for param_group in optimizer.param_groups:
                                        param_group['lr'] = current_lr
                                    self.logger.info(f"Optimizer LR updated to {current_lr} for exploit mode.")
                                else:
                                    self.logger.info(f"Loaded best model state_dict from {best_model_path}. Optimizer state not found.")
                                    # Set the learning rate to the exploit_lr value
                                    current_lr = hyperparams["EXPLOIT_LR"]
                                    for param_group in optimizer.param_groups:
                                        param_group['lr'] = current_lr
                                    self.logger.info(f"Optimizer LR set to {current_lr} for exploit mode.")
                            else: # Fallback for older model saves
                                model.load_state_dict(checkpoint)
                                self.logger.info(f"Loaded best model state_dict directly from {best_model_path} (older format).")
                        else:
                            self.logger.warning("Could not find best model to load for exploit mode. Continuing with current model.")

                        # Set the learning rate to the exploit_lr value
                        current_lr = hyperparams["EXPLOIT_LR"]
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = current_lr
                        self.logger.info(f"Optimizer LR set to {current_lr} for exploit mode.")

                        # Re-initialize scheduler for exploit mode with CosineAnnealingLR
                        final_lr = hyperparams["FINAL_LR"]
                        self.logger.info(f"Re-initializing scheduler with CosineAnnealingLR for exploit mode. Epochs={exploit_epochs}, Final LR={final_lr}")
                        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer,
                            T_max=exploit_epochs,
                            eta_min=final_lr
                        )
                        
                        # Re-validate immediately to confirm the model has been restored
                        self.logger.info("Re-validating the restored model to confirm best state.")
                        model.eval() # Ensure model is in evaluation mode for re-validation
                        
                        # Use CUDA Graphs validation if available, same as main training loop
                        if (hasattr(self.training_service, 'validate_epoch_with_graphs') and 
                            device.type == 'cuda' and model_type == 'FNN'):
                            val_loss_norm, epoch_val_preds_norm, epoch_val_trues_norm = self.training_service.validate_epoch_with_graphs(
                                model, val_loader, epoch, device, self.stop_requested, task, verbose=False
                            )
                        else:
                            # Standard validation
                            val_loss_norm, epoch_val_preds_norm, epoch_val_trues_norm = self.training_service.validate_epoch(
                                model, model_type, val_loader, h_s_val, h_c_val, epoch, device, self.stop_requested, task, verbose=False
                            )
                        
                        model.train() # Set model back to training mode
                        self.logger.info(f"Re-validation complete. Loss: {val_loss_norm:.6f}. This should match the best validation loss.")
                        
                        # Update the GUI with the re-validated loss to show the drop
                        progress_data['val_loss'] = val_loss_norm
                        progress_data['val_rmse_scaled'] = best_val_rmse_for_gui # Use the correct, denormalized best value
                        update_progress_callback.emit(progress_data)
                    else:
                        # If patience is reached and all exploit repetitions are used, stop.
                        early_stopping = True
                        self.logger.info(f"Early stopping at epoch {epoch} after exhausting exploit repetitions.")
                        task['results']['early_stopped_reason'] = 'Patience in Exploit Mode'
                        break
            
            # Log to SQLite for non-validation epochs or if not early stopped on a validation epoch
            if not early_stopping:
                if not (epoch == 1 or epoch % valid_freq == 0 or epoch == max_epochs): # If not a validation epoch
                    model_memory_usage = torch.cuda.memory_allocated() if torch.cuda.is_available() else sys.getsizeof(model)
                    model_memory_usage_mb = model_memory_usage / (1024 * 1024)
# This block is for epochs where validation did not run.
                    current_time_train_only = time.time()
                    elapsed_time_train_only = current_time_train_only - loop_start_time # Changed start_time to loop_start_time
                    # model_memory_usage_mb is already calculated at the start of this 'if not (epoch ...)' block
                    
                    # Calculate scaled RMSE for training for GUI if possible
                    train_rmse_for_gui_no_val = float('nan')
                    target_column_no_val = task['data_loader_params']['target_column']
                    # Determine error_unit_label and multiplier for this context
                    # These would have been set during a validation epoch if one occurred,
                    # otherwise, we need defaults or to fetch them.
                    # 'error_unit_label' is defined in the outer scope of the validation block
                    # We need to ensure it's available or use a default if this non-validation epoch occurs before any validation.
                    current_error_unit_label_no_val = "RMS Error" # Default
                    multiplier_no_val = 1.0 # Default
                    if 'error_unit_label' in locals() and 'multiplier' in locals(): # Check if set by validation block
                        current_error_unit_label_no_val = error_unit_label
                        multiplier_no_val = multiplier
                    else: # Recalculate if not set from validation context (e.g. first few epochs before validation)
                        if "voltage" in target_column_no_val.lower():
                            current_error_unit_label_no_val = "RMS Error [mV]"
                            multiplier_no_val = 1000.0
                        elif "soc" in target_column_no_val.lower():
                            current_error_unit_label_no_val = "RMS Error [% SOC]"
                            multiplier_no_val = 100.0
                        elif "soe" in target_column_no_val.lower():
                            current_error_unit_label_no_val = "RMS Error [% SOE]"
                            multiplier_no_val = 100.0
                        elif "sop" in target_column_no_val.lower():
                            current_error_unit_label_no_val = "RMS Error [% SOP]"
                            multiplier_no_val = 100.0
                        elif "temperature" in target_column_no_val.lower() or "temp" in target_column_no_val.lower():
                            current_error_unit_label_no_val = "RMS Error [Deg C]"
                            multiplier_no_val = 1.0

                    if self.loaded_scaler and target_column_no_val in self.scaler_metadata.get('normalized_columns', []):
                        if epoch_train_preds_norm is not None and epoch_train_trues_norm is not None and len(epoch_train_preds_norm) > 0:
                            try:
                                import pandas as pd 
                                import numpy as np
                                from vestim.services.data_processor.src import normalization_service 
                                e_t_p_n_cpu_no_val = epoch_train_preds_norm.cpu().numpy() if epoch_train_preds_norm.is_cuda else epoch_train_preds_norm.numpy()
                                e_t_t_n_cpu_no_val = epoch_train_trues_norm.cpu().numpy() if epoch_train_trues_norm.is_cuda else epoch_train_trues_norm.numpy()
                                temp_df_train_pred_no_val = pd.DataFrame(0, index=np.arange(len(e_t_p_n_cpu_no_val)), columns=self.scaler_metadata['normalized_columns'])
                                temp_df_train_pred_no_val[target_column_no_val] = e_t_p_n_cpu_no_val.flatten()
                                df_train_pred_inv_no_val = normalization_service.inverse_transform_data(temp_df_train_pred_no_val, self.loaded_scaler, self.scaler_metadata['normalized_columns'])
                                train_pred_orig_no_val = df_train_pred_inv_no_val[target_column_no_val].values
                                temp_df_train_true_no_val = pd.DataFrame(0, index=np.arange(len(e_t_t_n_cpu_no_val)), columns=self.scaler_metadata['normalized_columns'])
                                temp_df_train_true_no_val[target_column_no_val] = e_t_t_n_cpu_no_val.flatten()
                                df_train_true_inv_no_val = normalization_service.inverse_transform_data(temp_df_train_true_no_val, self.loaded_scaler, self.scaler_metadata['normalized_columns'])
                                train_true_orig_no_val = df_train_true_inv_no_val[target_column_no_val].values
                                train_mse_orig_no_val = np.mean((train_pred_orig_no_val - train_true_orig_no_val)**2)
                                train_rmse_for_gui_no_val = np.sqrt(train_mse_orig_no_val) * multiplier_no_val
                            except Exception as e_inv_train_no_val:
                                self.logger.error(f"Error during inverse transform for training data (non-val epoch {epoch}): {e_inv_train_no_val}.")
                                if train_loss_norm is not None and not math.isnan(train_loss_norm): train_rmse_for_gui_no_val = math.sqrt(max(0, train_loss_norm)) * multiplier_no_val
                        else: 
                            if train_loss_norm is not None and not math.isnan(train_loss_norm): train_rmse_for_gui_no_val = math.sqrt(max(0, train_loss_norm)) * multiplier_no_val
                    else: 
                        if train_loss_norm is not None and not math.isnan(train_loss_norm): train_rmse_for_gui_no_val = math.sqrt(max(0, train_loss_norm)) * multiplier_no_val

                    # Log to CSV for non-validation epochs
                    if csv_log_file:
                        with open(csv_log_file, 'a', newline='') as f:
                            csv_writer_train_only = csv.writer(f)
                            csv_writer_train_only.writerow([
                                epoch,
                                f"{train_loss_norm:.8f}" if train_loss_norm is not None else 'nan',
                                'nan',
                                f"{best_validation_loss:.8f}" if best_validation_loss is not None else 'nan',
                                f"{current_lr:.6e}" if current_lr is not None else 'nan',
                                f"{elapsed_time_train_only:.4f}" if elapsed_time_train_only is not None else 'nan',
                                f"{avg_batch_time:.6f}" if avg_batch_time is not None else 'nan',
                                patience_counter if patience_counter is not None else 'nan',
                                f"{model_memory_usage_mb:.4f}" if model_memory_usage_mb is not None else 'nan',
                                f"{epoch_duration:.4f}"
                            ])
                    
                    # Update GUI via signal (for non-validation epochs)
                    task_elapsed_time_train_only = time.time() - self.task_start_time if hasattr(self, 'task_start_time') else elapsed_time_train_only
                    
                    progress_data_train_only = {
                        'epoch': epoch,
                        'train_loss_norm': train_loss_norm, 
                        'val_loss_norm': float('nan'),      
                        'train_rmse_scaled': train_rmse_for_gui_no_val, 
                        'val_rmse_scaled': float('nan'),         
                        'best_val_rmse_scaled': getattr(self, f'_task_{task["task_id"]}_best_val_rmse_orig', float('inf')), 
                        'error_unit_label': current_error_unit_label_no_val, 
                        'delta_t_epoch': formatted_epoch_time, 
                        'elapsed_time': format_time(elapsed_time_train_only), 
                        'patience_counter': patience_counter, 
                        'learning_rate': current_lr, 
                        'status': f"Epoch {epoch}/{max_epochs} - Training..."
                    }
                    # DON'T update GUI for non-validation epochs - only during validation
                    # update_progress_callback.emit(progress_data_train_only)
                    self.logger.info(f"Epoch {epoch}/{max_epochs} - Training only (no validation, ValidFreq={valid_freq}) - GUI not updated")
                elif (epoch == 1 or epoch % valid_freq == 0 or epoch == max_epochs): # If it IS a validation epoch but did NOT early stop
                    model_memory_usage = torch.cuda.memory_allocated() if torch.cuda.is_available() else sys.getsizeof(model)
                    model_memory_usage_mb = model_memory_usage / (1024 * 1024)
                    # self.log_to_sqlite(
                    #     task=task, epoch=epoch, train_loss=train_loss_norm,
                    #     val_loss=val_loss_norm,
                    #     best_val_loss=best_validation_loss,
                    #     elapsed_time=elapsed_time,
                    #     avg_batch_time=avg_batch_time, early_stopping=False,
                    #     model_memory_usage=round(model_memory_usage_mb, 3), current_lr=current_lr
                    # )
            
            # Scheduler step - different logic for different schedulers
            scheduler_type = hyperparams.get("SCHEDULER_TYPE", "StepLR")
            
            if scheduler_type == "ReduceLROnPlateau":
                # For ReduceLROnPlateau, pass the validation loss
                scheduler.step(val_loss_norm)
            elif scheduler_type in ["StepLR", "CosineAnnealingWarmRestarts"]:
                # For StepLR and CosineAnnealingWarmRestarts, just step
                scheduler.step()

            # Log data to CSV and SQLite after each epoch (whether validated or not)
            #print(f"Checking log files for the task: {task['task_id']}: task['csv_log_file'], task['db_log_file']")

            # Save log data to CSV and SQLite
            # self.log_to_csv(task, epoch, train_loss, val_loss, elapsed_time, current_lr, best_validation_loss, delta_t_epoch)
            model_memory_usage = torch.cuda.memory_allocated() if torch.cuda.is_available() else sys.getsizeof(model)
            model_memory_usage_mb = model_memory_usage / (1024 * 1024)  # Convert to MB

            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != current_lr:
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
            self.logger.info("Training was stopped early. Exiting...")

        # Final save and cleanup
        # self.save_model(task) # REMOVED: Best model is saved during validation improvement.
        
        # Calculate final task elapsed time before saving summary
        final_task_elapsed_time = time.time() - self.task_start_time if hasattr(self, 'task_start_time') else 0

        # Populate the results dictionary within the task object itself
        task['results'] = {
            'best_train_loss_normalized': best_train_loss_norm,
            'best_train_loss_denormalized': getattr(self, f'_task_{task["task_id"]}_best_train_rmse_orig', best_train_loss_denorm),
            'best_validation_loss_normalized': best_validation_loss,
            'best_validation_loss_denormalized': getattr(self, f'_task_{task["task_id"]}_best_val_rmse_orig', float('inf')),
            'final_train_loss_normalized': train_loss_norm,
            'final_validation_loss_normalized': val_loss_norm,
            'completed_epochs': epoch,
            'early_stopped': early_stopping,
            'early_stopped_reason': task.get('results', {}).get('early_stopped_reason', 'Patience' if early_stopping else 'Completed')
        }

        # Save detailed training task summary for testing GUI integration
        self._save_training_task_summary(task, best_validation_loss, train_loss_norm, val_loss_norm,
                                       epoch, max_epochs, early_stopping, final_task_elapsed_time,
                                       best_train_loss_norm, best_train_loss_denorm,
                                       train_loss_history, val_loss_history)

        # The text summary is redundant now that we have a detailed JSON summary.
        # This section has been removed to clean up the output.

        self.logger.info(f"Training loop finished for task {task['task_id']}. Best model is at: {task.get('training_params', {}).get('best_model_path')}")
        formatted_task_time = format_time(final_task_elapsed_time)
        
        # Calculate final job elapsed time
        final_job_elapsed_time = time.time() - self.job_start_time if hasattr(self, 'job_start_time') and self.job_start_time is not None else 0
        formatted_job_time = format_time_long(final_job_elapsed_time)
        
        update_progress_callback.emit({
            'task_completed': True, 
            'final_task_elapsed_time': final_task_elapsed_time,
            'formatted_task_time': formatted_task_time,
            'final_job_elapsed_time': final_job_elapsed_time,
            'formatted_job_time': formatted_job_time
        })
        self.logger.info(f"Training task completed in {formatted_task_time}. Total job time: {formatted_job_time}")
    # End of _run_training_loop method, convert_hyperparams should be at class level indentation

    def run_optuna_training(self, task, update_progress_callback, train_loader, val_loader, device):
        """Run the training process for a single Optuna trial."""
        # IMPORTANT: Switch to standard training service for Optuna
        # CUDAGraphsTrainingService doesn't have train_epoch() method, only train_epoch_with_graphs()
        # Since we disabled CUDA graphs for Optuna, we need the standard service
        if isinstance(self.training_service, CUDAGraphsTrainingService):
            self.training_service = TrainingTaskService(device=self.device)
            self.logger.info("Switched to standard TrainingTaskService for Optuna optimization.")
        
        try:
            self._run_optuna_training_loop(task, update_progress_callback, train_loader, val_loader, device)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            self.logger.warning(f"Training failed for Optuna trial {task.get('task_id', 'N/A')}: {e}.")
            self.logger.error("Training error occurred during Optuna trial. Cannot recover.")
            raise e

    def _run_optuna_training_loop(self, task, update_progress_callback, train_loader, val_loader, device):
        """
        The actual training loop logic for an Optuna trial, with pruning callbacks.
        """
        import optuna # Ensure Optuna is available
        optuna_trial = task.get('optuna_trial')
        if not optuna_trial:
            raise ValueError("run_optuna_training called without an Optuna trial object in the task.")

        try:
            self.logger.info(f"--- Starting _run_optuna_training_loop for trial: {optuna_trial.number} ---")
            
            # DISABLE CUDA Graphs for Optuna optimization
            # Reason: Dynamic architectures and batch sizes between trials cause persistent CUDA errors
            # "operation failed due to a previous error during capture" cannot be reliably recovered
            # Optuna trials change model architecture/batch size, which breaks CUDA graph assumptions
            # Standard training is fast enough for Optuna's short trial runs
            use_cuda_graphs = False
            model_type = task['hyperparams'].get('MODEL_TYPE', 'LSTM')
            
            if model_type == 'FNN' and device.type == 'cuda':
                self.logger.info("CUDA graphs disabled for Optuna trials (incompatible with dynamic architecture search)")
            
            # Most of this setup is identical to run_training
            setattr(self, f'_task_{task["task_id"]}_best_val_rmse_orig', float('inf'))
            best_train_loss_norm = float('inf')
            best_train_loss_denorm = float('inf')
            hyperparams = self.convert_hyperparams(task['hyperparams'])
            model = task['model'].to(device)
            max_epochs = hyperparams['MAX_EPOCHS']
            valid_freq = hyperparams.get('VALID_FREQUENCY', 1)
            valid_patience = hyperparams['VALID_PATIENCE']
            current_lr = hyperparams['INITIAL_LR']
            lr_drop_period = hyperparams['LR_DROP_PERIOD']
            lr_drop_factor = hyperparams['LR_DROP_FACTOR']
            best_validation_loss = float('inf')
            patience_counter = 0
            loop_start_time = time.time()
            last_validation_time = loop_start_time
            early_stopping = False
            max_training_time_seconds = int(task.get('training_params', {}).get('max_training_time_seconds', 0))
            overall_training_start_time = time.time()

            # Resolve optimizer choice for Optuna path as well
            raw_opt = task.get('hyperparams', {}).get('OPTIMIZER_TYPE', 'Adam')
            optimizer_type = raw_opt.split(',')[0].strip() if isinstance(raw_opt, str) else str(raw_opt)
            try:
                weight_decay = float(hyperparams.get('WEIGHT_DECAY', 0.0))
            except Exception:
                weight_decay = 0.0

            if use_cuda_graphs:
                # CUDA Graphs requires capturable=True
                if optimizer_type.lower() == 'adamw':
                    try:
                        optimizer = torch.optim.AdamW(model.parameters(), lr=current_lr, weight_decay=weight_decay, capturable=True)
                    except TypeError:
                        optimizer = torch.optim.AdamW(model.parameters(), lr=current_lr, weight_decay=weight_decay)
                else:
                    try:
                        optimizer = torch.optim.Adam(model.parameters(), lr=current_lr, weight_decay=weight_decay, capturable=True)
                    except TypeError:
                        optimizer = torch.optim.Adam(model.parameters(), lr=current_lr, weight_decay=weight_decay)
                self.logger.info(f"Created CUDA Graphs-compatible Optuna optimizer ({optimizer_type})")
            else:
                # Standard optimizer for non-CUDA graphs training
                if optimizer_type.lower() == 'adamw':
                    optimizer = torch.optim.AdamW(model.parameters(), lr=current_lr, weight_decay=weight_decay)
                else:
                    optimizer = torch.optim.Adam(model.parameters(), lr=current_lr, weight_decay=weight_decay)
                self.logger.info(f"Created standard Optuna optimizer ({optimizer_type})")
                
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_drop_period, gamma=lr_drop_factor)
            csv_log_file = task.get('csv_log_file')
            if csv_log_file:
                os.makedirs(os.path.dirname(csv_log_file), exist_ok=True)
                with open(csv_log_file, 'w', newline='') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow(["epoch", "train_loss_norm", "val_loss_norm", "best_val_loss_norm", "learning_rate", "elapsed_time_sec", "avg_batch_time_sec", "patience_counter", "model_memory_mb"])

            # Training loop
            for epoch in range(1, max_epochs + 1):
                if self.stop_requested:
                    self.logger.info("Training stopped by user request.")
                    break
                
                # Identical training phase as run_training with CUDA Graphs support
                model.train()
                actual_train_batch_size = train_loader.batch_size or int(hyperparams.get('BATCH_SIZE', 32))
                h_s, h_c = None, None
                model_type = task.get('hyperparams', {}).get('MODEL_TYPE', 'LSTM') # More reliable source
                if model_type in ['LSTM', 'GRU']:
                    h_s = torch.zeros(model.num_layers, actual_train_batch_size, model.hidden_units).to(device)
                    if model_type == 'LSTM':
                        h_c = torch.zeros(model.num_layers, actual_train_batch_size, model.hidden_units).to(device)
                
                verbose = task.get('verbose', True)
                
                # CUDA graphs disabled for Optuna - use standard training for all trials
                # Standard training - works reliably with dynamic architectures
                avg_batch_time, train_loss_norm, _, _ = self.training_service.train_epoch(
                    model, model_type, train_loader, optimizer, None, None, epoch, device, self.stop_requested, task, verbose=verbose
                )
                
                if self.stop_requested:
                    break

                # Validation and Pruning Phase
                if epoch == 1 or epoch % valid_freq == 0 or epoch == max_epochs:
                    # Standard validation for Optuna
                    val_loss_norm, epoch_val_preds_norm, epoch_val_trues_norm = self.training_service.validate_epoch(
                        model, model_type, val_loader, None, None, epoch, device, self.stop_requested, task, verbose=verbose
                    )
                    # --- OPTUNA PRUNING LOGIC ---
                    log_callback = task.get('log_callback')
                    if log_callback:
                        log_callback(f"  Trial {optuna_trial.number} | Epoch {epoch}/{max_epochs} - Reported Val Loss: {val_loss_norm:.6f}")
                    
                    optuna_trial.report(val_loss_norm, epoch)
                    if optuna_trial.should_prune():
                        # This exception is caught by Optuna's study runner and marks the trial as pruned.
                        raise optuna.exceptions.TrialPruned()
                    # --- END OPTUNA PRUNING LOGIC ---

                    # Standard early stopping logic (can run in parallel with pruning)
                    if val_loss_norm < best_validation_loss:
                        best_validation_loss = val_loss_norm
                        patience_counter = 0
                        # No model saving during Optuna trials to save disk space and time
                    else:
                        patience_counter += 1
                    
                    if patience_counter > valid_patience:
                        self.logger.info(f"Optuna trial {optuna_trial.number} stopped early at epoch {epoch} due to validation patience.")
                        early_stopping = True
                        break
                
                scheduler.step()

            # Populate results after the loop finishes (or breaks)
            task['results'] = {
                'best_validation_loss_normalized': best_validation_loss,
                'completed_epochs': epoch,
                'early_stopped': early_stopping,
                'early_stopped_reason': 'Patience' if early_stopping else 'Completed'
            }
            self.logger.info(f"Optuna training loop finished for trial {optuna_trial.number}. Best validation loss: {best_validation_loss:.6f}")

        except optuna.exceptions.TrialPruned:
            # If a TrialPruned exception was raised, re-raise it so Optuna can handle it.
            raise
        except Exception as e:
            self.logger.error(f"Error during Optuna training for trial {optuna_trial.number}: {str(e)}", exc_info=True)
            task['results'] = {
                'best_validation_loss_normalized': float('inf'),
                'error': str(e),
                'completed_epochs': 0
            }
            # Do not raise here, let the Optuna study log the failure and continue.

    def convert_hyperparams(self, hyperparams):
        """Converts all relevant hyperparameters to the correct types based on model type."""
        # Get model type to determine which parameters to convert
        model_type = hyperparams.get('MODEL_TYPE', 'LSTM')
        
        # Common parameters for all model types
        # Handle boundary format parameters before type conversion
        self._convert_boundary_format_params(hyperparams)
        
        hyperparams['BATCH_SIZE'] = int(hyperparams['BATCH_SIZE'])
        hyperparams['MAX_EPOCHS'] = int(hyperparams['MAX_EPOCHS'])
        hyperparams['INITIAL_LR'] = float(hyperparams['INITIAL_LR'])
        
        # Model-specific parameter conversion
        if model_type in ['LSTM', 'GRU']:
            # RNN-specific parameters
            hyperparams['LAYERS'] = int(hyperparams['LAYERS'])
            hyperparams['HIDDEN_UNITS'] = int(hyperparams['HIDDEN_UNITS'])
        elif model_type == 'FNN':
            # FNN-specific parameters - HIDDEN_LAYERS is already a string, no conversion needed
            # FNN doesn't use LAYERS or HIDDEN_UNITS parameters
            pass
        
        # Update scheduler parameter names to match task info
        if hyperparams['SCHEDULER_TYPE'] == 'StepLR':
            hyperparams['LR_DROP_PERIOD'] = int(hyperparams['LR_PERIOD'])  # Map LR_PERIOD to LR_DROP_PERIOD
            hyperparams['LR_DROP_FACTOR'] = float(hyperparams['LR_PARAM'])  # Map LR_PARAM to LR_DROP_FACTOR
        elif hyperparams['SCHEDULER_TYPE'] == 'ReduceLROnPlateau':
            hyperparams['PLATEAU_PATIENCE'] = int(hyperparams['PLATEAU_PATIENCE'])
            hyperparams['PLATEAU_FACTOR'] = float(hyperparams['PLATEAU_FACTOR'])
            # Provide legacy parameters for backward compatibility
            hyperparams['LR_DROP_PERIOD'] = 10  # Default for non-StepLR schedulers
            hyperparams['LR_DROP_FACTOR'] = 0.5  # Default for non-StepLR schedulers
        elif hyperparams['SCHEDULER_TYPE'] == 'CosineAnnealingWarmRestarts':
            # Safely handle CosineAnnealingWarmRestarts parameters with defaults
            hyperparams['COSINE_T0'] = int(hyperparams.get('COSINE_T0', '10'))
            hyperparams['COSINE_T_MULT'] = int(hyperparams.get('COSINE_T_MULT', '2'))
            hyperparams['COSINE_ETA_MIN'] = float(hyperparams.get('COSINE_ETA_MIN', '1e-6'))
            # Provide legacy parameters for backward compatibility
            hyperparams['LR_DROP_PERIOD'] = 10  # Default for non-StepLR schedulers
            hyperparams['LR_DROP_FACTOR'] = 0.5  # Default for non-StepLR schedulers
        else:
            # For any other scheduler, provide defaults for legacy parameters
            hyperparams['LR_DROP_PERIOD'] = 10
            hyperparams['LR_DROP_FACTOR'] = 0.5
        hyperparams['VALID_PATIENCE'] = int(hyperparams['VALID_PATIENCE'])
        hyperparams['VALID_FREQUENCY'] = int(hyperparams.get('VALID_FREQUENCY', 1))
        
        # Handle 'N/A' for LOOKBACK before converting to int
        lookback_val = hyperparams.get('LOOKBACK', 0)
        if lookback_val == 'N/A':
            hyperparams['LOOKBACK'] = 0
        else:
            hyperparams['LOOKBACK'] = int(lookback_val)
            
        hyperparams['REPETITIONS'] = int(hyperparams['REPETITIONS'])

        # Handle exploit-phase hyperparameters with smart defaults
        # First, determine if exploitation is enabled
        exploit_epochs_raw = hyperparams.get('EXPLOIT_EPOCHS')
        
        try:
            if exploit_epochs_raw is None or str(exploit_epochs_raw).strip() == '':
                exploit_epochs = 0
            else:
                exploit_epochs = int(exploit_epochs_raw)
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Invalid EXPLOIT_EPOCHS value '{exploit_epochs_raw}'. Using default: 0 (disabled)")
            exploit_epochs = 0
        
        hyperparams['EXPLOIT_EPOCHS'] = exploit_epochs
        
        # Only require other exploit params if exploit is enabled (epochs > 0)
        if exploit_epochs > 0:
            # EXPLOIT_LR is required when exploit is enabled
            exploit_lr_raw = hyperparams.get('EXPLOIT_LR')
            try:
                if exploit_lr_raw is None or str(exploit_lr_raw).strip() == '':
                    # Use initial LR as fallback
                    fallback_lr = hyperparams.get('INITIAL_LR', 0.001)
                    self.logger.warning(f"EXPLOIT_LR not provided but exploit is enabled (EXPLOIT_EPOCHS={exploit_epochs}). Using INITIAL_LR as default: {fallback_lr}")
                    hyperparams['EXPLOIT_LR'] = float(fallback_lr)
                else:
                    hyperparams['EXPLOIT_LR'] = float(exploit_lr_raw)
            except (ValueError, TypeError) as e:
                fallback_lr = hyperparams.get('INITIAL_LR', 0.001)
                self.logger.warning(f"Invalid EXPLOIT_LR value '{exploit_lr_raw}'. Using INITIAL_LR as default: {fallback_lr}")
                hyperparams['EXPLOIT_LR'] = float(fallback_lr)
            
            # EXPLOIT_REPETITIONS with default
            exploit_reps_raw = hyperparams.get('EXPLOIT_REPETITIONS')
            try:
                if exploit_reps_raw is None or str(exploit_reps_raw).strip() == '':
                    self.logger.info("EXPLOIT_REPETITIONS not provided. Using default: 1")
                    hyperparams['EXPLOIT_REPETITIONS'] = 1
                else:
                    hyperparams['EXPLOIT_REPETITIONS'] = int(exploit_reps_raw)
            except (ValueError, TypeError) as e:
                self.logger.warning(f"Invalid EXPLOIT_REPETITIONS value '{exploit_reps_raw}'. Using default: 1")
                hyperparams['EXPLOIT_REPETITIONS'] = 1
        else:
            # Exploit disabled, set safe defaults
            hyperparams['EXPLOIT_LR'] = 0.0
            hyperparams['EXPLOIT_REPETITIONS'] = 0
        
        # FINAL_LR is optional regardless of exploit status
        final_lr_raw = hyperparams.get('FINAL_LR')
        try:
            if final_lr_raw is not None and str(final_lr_raw).strip() != '':
                hyperparams['FINAL_LR'] = float(final_lr_raw)
            else:
                hyperparams['FINAL_LR'] = None
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Invalid FINAL_LR value '{final_lr_raw}'. Setting to None")
            hyperparams['FINAL_LR'] = None
            
        return hyperparams

    def _convert_boundary_format_params(self, hyperparams):
        """Convert boundary format parameters [min,max] to usable default values."""
        # Define parameters that should be treated as integers vs floats
        integer_params = {
            "LAYERS", "HIDDEN_UNITS", "GRU_LAYERS", "GRU_HIDDEN_UNITS", 
            "MAX_EPOCHS", "VALID_PATIENCE", "VALID_FREQUENCY", "LOOKBACK",
            "BATCH_SIZE", "LR_PERIOD", "PLATEAU_PATIENCE", "REPETITIONS",
            "COSINE_T0", "COSINE_T_MULT"
        }
        # Note: RNN_LAYER_SIZES is NOT an integer param - it's a comma-separated string
        
        for param_name, param_value in hyperparams.items():
            if isinstance(param_value, str) and param_value.startswith('[') and param_value.endswith(']'):
                try:
                    # Parse boundary format [min,max]
                    inner = param_value[1:-1].strip()
                    parts = [part.strip() for part in inner.split(',')]
                    
                    if len(parts) == 2:
                        min_val = float(parts[0])
                        max_val = float(parts[1])
                        
                        # Set a reasonable default value from the range
                        if param_name in integer_params:
                            # For integers, use the middle value rounded to int
                            default_val = int((min_val + max_val) / 2)
                        else:
                            # For floats, use the geometric mean for learning rates, arithmetic mean for others
                            if param_name in ["INITIAL_LR", "LR_PARAM", "PLATEAU_FACTOR", "FNN_DROPOUT_PROB"]:
                                # Use geometric mean for learning rates and probabilities
                                import math
                                default_val = math.sqrt(min_val * max_val)
                            else:
                                # Use arithmetic mean for other float parameters
                                default_val = (min_val + max_val) / 2
                        
                        hyperparams[param_name] = default_val
                        self.logger.info(f"Converted {param_name} boundary format {param_value} to default value: {default_val}")
                        
                except (ValueError, IndexError) as e:
                    self.logger.error(f"Could not parse boundary format for {param_name}: {param_value} - {e}")
                    raise ValueError(f"Invalid {param_name} value: {param_value}")

    def _save_training_task_summary(self, task, best_val_loss_norm, final_train_loss_norm, final_val_loss_norm, final_epoch, max_epochs, early_stopping, elapsed_time, best_train_loss_norm, best_train_loss_denorm, train_loss_history, val_loss_history):
        """Saves a detailed summary of the completed training task and stores it."""
        try:
            task_id = task['task_id']
            # Use task_dir (includes hyperparam subfolder) so each configuration has separate results
            task_dir = task.get('task_dir') or task.get('model_dir')
            if not task_dir:
                self.logger.warning(f"No task_dir or model_dir found for task {task_id}. Skipping summary save.")
                return
            summary_file_path = os.path.join(task_dir, 'training_summary.json')

            # Get the final denormalized losses from the attributes set during training
            final_train_loss_denorm = getattr(self, f'_task_{task_id}_last_train_rmse_orig', float('nan'))
            final_val_loss_denorm = getattr(self, f'_task_{task_id}_last_val_rmse_orig', float('nan'))
            best_validation_loss_denorm = getattr(self, f'_task_{task_id}_best_val_rmse_orig', float('inf'))
            best_train_loss_denorm = getattr(self, f'_task_{task_id}_best_train_rmse_orig', float('inf'))

            # Restructure training history for easier analysis
            history = []
            for i in range(len(train_loss_history)):
                epoch_data = {
                    'epoch': i + 1,
                    'train_loss_normalized': train_loss_history[i],
                    'validation_loss_normalized': val_loss_history[i] if i < len(val_loss_history) else None
                }
                history.append(epoch_data)

            # Get absolute paths for data sources
            job_folder_path = self.job_manager.get_job_folder()
            train_data_path = os.path.abspath(os.path.join(job_folder_path, 'train_data', 'processed_data'))
            val_data_path = os.path.abspath(os.path.join(job_folder_path, 'val_data', 'processed_data'))
            test_data_path = os.path.abspath(os.path.join(job_folder_path, 'test_data', 'processed_data'))

            # Use the centralized filtering method from HyperParamManager
            hp_manager = VEstimHyperParamManager()
            hyperparams_to_save = hp_manager._filter_hyperparams_for_saving(task.get('hyperparams', {}))

            summary_data = {
                'task_id': task_id,
                'model_type': task.get('model_metadata', {}).get('model_type', 'N/A'),
                'data_sources': {
                    'training_data': train_data_path,
                    'validation_data': val_data_path,
                    'testing_data': test_data_path
                },
                'hyperparameters': hyperparams_to_save,
                'best_train_loss_normalized': best_train_loss_norm,
                'best_train_loss_denormalized': best_train_loss_denorm,
                'best_validation_loss_normalized': best_val_loss_norm,
                'best_validation_loss_denormalized': best_validation_loss_denorm,
                'final_train_loss_normalized': final_train_loss_norm,
                'final_train_loss_denormalized': final_train_loss_denorm,
                'final_validation_loss_normalized': final_val_loss_norm,
                'final_validation_loss_denormalized': final_val_loss_denorm,
                'completed_epochs': final_epoch,
                'max_epochs': max_epochs,
                'early_stopped': early_stopping,
                'training_time_seconds': elapsed_time,
                'training_time_formatted': format_time_long(elapsed_time),
                'training_history': history
            }

            with open(summary_file_path, 'w') as f:
                json.dump(summary_data, f, indent=4)

            # Store the results in the manager's dictionary for passing to the testing GUI
            self.training_results[task_id] = {
                'best_train_loss': best_train_loss_denorm,
                'best_validation_loss': best_validation_loss_denorm,  # Match key expected by testing manager
                'completed_epochs': final_epoch,
            }
            self.logger.info(f"Saved training summary for task {task_id} to {summary_file_path}")

        except Exception as e:
            self.logger.error(f"Failed to save training task summary for {task.get('task_id', 'N/A')}: {e}", exc_info=True)

    def save_model(self, task, save_path=None):
        """Save the trained model to disk. Uses save_path if provided, else defaults to task['model_path']."""
        try:
            # Ensure INPUT_SIZE and OUTPUT_SIZE are present in hyperparams for model saving
            if 'INPUT_SIZE' not in task['hyperparams']:
                task['hyperparams']['INPUT_SIZE'] = len(task['data_loader_params']['feature_columns'])
            if 'OUTPUT_SIZE' not in task['hyperparams']:
                task['hyperparams']['OUTPUT_SIZE'] = 1  # Default to 1 if not specified

            path_to_save = save_path if save_path else task.get('model_path')
            
            if path_to_save is None:
                self.logger.error("No valid save path provided or found in task for saving model.")
                raise ValueError("No valid save path for model.")

            model = task['model']
            if model is None:
                self.logger.error("No model instance found in task.")
                raise ValueError("No model instance found in task.")

            # Save model and optimizer state
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, path_to_save)
            self.logger.info(f"Model and optimizer state saved to {path_to_save}")

            # Save portable version (if this is the best model)
            is_best_model_save = save_path and os.path.basename(save_path) == 'best_model.pth'
            if is_best_model_save:
                task_dir = os.path.dirname(path_to_save) # Use path_to_save
                export_path = os.path.join(task_dir, 'best_model_export.pt') # Differentiate export name
                
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

                # Create export dictionary with all necessary information - model type aware
                model_type = task['model_metadata']['model_type']
                export_dict = {
                    'state_dict': model.state_dict(),
                    'model_definition': model_def,
                    'model_metadata': task['model_metadata'],
                    'hyperparams': {
                        'input_size': task['hyperparams']['INPUT_SIZE'],
                        'output_size': task['hyperparams']['OUTPUT_SIZE']
                    },
                    'data_config': {
                        'feature_columns': task['data_loader_params']['feature_columns'],
                        'target_column': task['data_loader_params']['target_column'],
                        'lookback': task['data_loader_params']['lookback']
                    },
                    'model_type': model_type,
                    'export_timestamp': time.strftime("%Y%m%d-%H%M%S")
                }
                
                # Add model-specific hyperparameters
                if model_type in ['LSTM', 'GRU']:
                    export_dict['hyperparams']['hidden_size'] = task['hyperparams']['HIDDEN_UNITS']
                    export_dict['hyperparams']['num_layers'] = task['hyperparams']['LAYERS']
                elif model_type == 'FNN':
                    export_dict['hyperparams']['hidden_layer_sizes'] = task['hyperparams']['HIDDEN_LAYER_SIZES']
                    export_dict['hyperparams']['dropout_prob'] = task['hyperparams']['DROPOUT_PROB']

                # Save the export dictionary
                torch.save(export_dict, export_path)
                self.logger.info(f"Best model exported to {export_path}")

                # Create a README file with multiple loading options for the best model
                readme_path = os.path.join(task_dir, 'BEST_MODEL_LOADING_INSTRUCTIONS.md')
                
                # Model-specific details for README
                if model_type in ['LSTM', 'GRU']:
                    arch_details = f"""- Hidden Units: {task['hyperparams']['HIDDEN_UNITS']}
- Layers: {task['hyperparams']['LAYERS']}"""
                elif model_type == 'FNN':
                    arch_details = f"""- Hidden Layer Sizes: {task['hyperparams']['HIDDEN_LAYER_SIZES']}
- Dropout Probability: {task['hyperparams']['DROPOUT_PROB']}"""
                else:
                    arch_details = "- Architecture details: See model metadata"
                
                readme_content = f"""# Model Loading Instructions

## Model Details
- Model Type: {model_type}
- Input Size: {task['hyperparams']['INPUT_SIZE']}
{arch_details}
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

# Create model instance based on model type
model_type = checkpoint['model_type']
if model_type in ['LSTM', 'GRU']:
    model_service = LSTMModelService()
    model = model_service.create_model(
        input_size=checkpoint['hyperparams']['input_size'],
        hidden_size=checkpoint['hyperparams']['hidden_size'],
        num_layers=checkpoint['hyperparams']['num_layers'],
        output_size=checkpoint['hyperparams']['output_size']
    )
elif model_type == 'FNN':
    from vestim.services.model_training.src.FNN_model_service import FNNModelService
    model_service = FNNModelService()
    model = model_service.create_model(
        input_size=checkpoint['hyperparams']['input_size'],
        hidden_layer_sizes=checkpoint['hyperparams']['hidden_layer_sizes'],
        output_size=checkpoint['hyperparams']['output_size'],
        dropout_prob=checkpoint['hyperparams']['dropout_prob']
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

# Create model instance based on model type
model_type = checkpoint['model_type']
if model_type in ['LSTM', 'GRU']:
    model = LSTMModel(
        input_size=checkpoint['hyperparams']['input_size'],
        hidden_units=checkpoint['hyperparams']['hidden_size'],
        num_layers=checkpoint['hyperparams']['num_layers'],
        output_size=checkpoint['hyperparams']['output_size']
    )
elif model_type == 'FNN':
    # For FNN, you would need to execute the FNN model definition
    # and create an FNN model instance with appropriate parameters
    model = FNNModel(
        input_size=checkpoint['hyperparams']['input_size'],
        hidden_layer_sizes=checkpoint['hyperparams']['hidden_layer_sizes'],
        output_size=checkpoint['hyperparams']['output_size'],
        dropout_prob=checkpoint['hyperparams']['dropout_prob']
    )

# Load state dict
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Example usage:
def predict(model, input_data):
    with torch.no_grad():
        if model_type in ['LSTM', 'GRU']:
            output, _ = model(input_data)
        else:  # FNN
            output = model(input_data)
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
                self.logger.info(f"Best model loading instructions saved to {readme_path}")

        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise

    def get_training_results(self):
        """Return the dictionary of training results."""
        return self.training_results

    def stop_task(self):
        self.stop_requested = True  # Set the flag to request a stop
        if self.training_thread and self.training_thread.isRunning():  # Use isRunning() instead of is_alive()
            self.logger.info("Attempting to gracefully stop the training thread...")
            self.training_thread.quit()  # Gracefully stop the thread
            if self.training_thread.wait(7000):  # Wait for the thread to finish cleanly
                self.logger.info("Training thread has finished after stop request.")
            else:
                self.logger.warning("Training thread did not finish cleanly after stop request and 7s wait.")
