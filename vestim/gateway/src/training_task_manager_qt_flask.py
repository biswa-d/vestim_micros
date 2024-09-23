from flask import Blueprint, jsonify
import logging, sqlite3, os, csv, time, sys, requests
import torch
import threading
# from vestim.gateway.src.job_manager_qt import JobManager
# from vestim.gateway.src.training_setup_manager_qt_flask import VEstimTrainingSetupManager
from vestim.services.model_training.src.data_loader_service_padfil import DataLoaderService
from vestim.services.model_training.src.training_task_service_test import TrainingTaskService

# Training Task Manager Blueprint
training_task_blueprint = Blueprint('training_task_manager', __name__)

class TrainingTaskManager:
    _instance = None  # Singleton instance
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(TrainingTaskManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):  # Ensure the attributes are initialized once
            self.logger = logging.getLogger(__name__)
            self.data_loader_service = DataLoaderService()
            self.training_service = TrainingTaskService()

            self.job_folder = None
            self.job_id = None  
            self.params = None
            self.current_hyper_params = None
            self.task_list = None
            self.current_task = None
            self.current_task_status = {"status": "initialized", "progress_data": {}}  # Initialize task status as a dictionary
            self.stop_requested = False
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.training_thread = None
            self.task_index = 0  # Initialize the task index
            self.initialized = True  # Ensure it's only initialized once
            
    # Get required objects from the relevant singleton manaers
    def fetch_job_folder(self):
        """Fetches and stores the job folder from the Job Manager API."""
        if self.job_folder is None:
            try:
                response_job = requests.get("http://localhost:5000/job_manager/get_job_folder")
                if response_job.status_code == 200:
                    self.job_folder = response_job.json()['job_folder']
                else:
                    raise Exception("Failed to fetch job folder")
            except Exception as e:
                self.logger.error(f"Error fetching job folder: {str(e)}")
                raise e
    
    def fetch_job_id(self):
        """Fetches and stores the job folder from the Job Manager API."""
        if self.job_id is None:
            try:
                response_id = requests.get("http://localhost:5000/job_manager/get_job_id")
                if response_id.status_code == 200:
                    self.job_id = response_id.json()['job_id']
                else:
                    raise Exception("Failed to fetch job ID")
            except Exception as e:
                self.logger.error(f"Error fetching job id: {str(e)}")
                raise e


    def fetch_hyper_params(self):
        """Fetches and stores the hyperparameters from the Hyper Param Manager API."""
        if self.params is None:
            try:
                response_params = requests.get("http://localhost:5000/hyper_param_manager/get_params")
                if response_params.status_code == 200:
                    self.params = response_params.json()
                    self.current_hyper_params = self.params
                else:
                    raise Exception("Failed to fetch hyperparameters")
            except Exception as e:
                self.logger.error(f"Error fetching hyperparameters: {str(e)}")
                raise e

    def fetch_task_list(self):
        """Fetches and stores the task list from the Training Setup Manager API."""
        if self.task_list is None:
            try:
                response_task_list = requests.get("http://localhost:5000/training_setup/get_tasks")
                if response_task_list.status_code == 200:
                    self.task_list = response_task_list.json()
                else:
                    raise Exception("Failed to fetch task list")
            except Exception as e:
                self.logger.error(f"Error fetching task list: {str(e)}")
                raise e

    # Define the start_training method
    def start_training(self):
        """Starts the training in a separate thread."""
        training_thread = threading.Thread(target=self.run_training_tasks)
        training_thread.start()

    def run_training_tasks(self):
        """Run the training tasks sequentially in a separate thread."""
        self.fetch_task_list()
        total_tasks = len(self.task_list)
        self.start_time = time.time()  # Record the start time of the training

        while self.task_index < total_tasks:
            if self.stop_requested:
                self.logger.info("Training stopped by user.")
                break

            task = self.task_list[self.task_index]
            self.logger.info(f"Starting training for task: {task['task_id']}")
            self.process_task(task)
            self.task_index += 1  # Update the task index after each task process

        if not self.stop_requested:
            self.logger.info("All tasks completed.")
        else:
            self.logger.info(f"Training interrupted at task {self.task_index + 1}.")


    def process_task(self, task):
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

            # Configuring DataLoader
            self.logger.info("Configuring DataLoader")

            # Create data loaders for the task
            train_loader, val_loader = self.data_loader_service.create_data_loaders(task)
            self.logger.info(f"DataLoader configured for task: {task['hyperparams']}")

            # Call the training method and pass logging information (task_id, db path)
            self.run_training(task, train_loader, val_loader, self.device)

        except Exception as e:
            self.logger.error(f"Error during task processing: {str(e)}")
            self.current_task_status = {"status": "error", "error": str(e), "task_id": task['task_id']}

    def setup_job_logging(self, task):
        """
        Set up the database and logging environment for the job.
        This ensures that the database tables are created if they do not exist.
        """
        self.fetch_job_id()  # Get the job ID
        model_dir = task.get('model_dir')  # Path where task-related logs are stored

        # Retrieve log file paths from the task
        csv_log_file = task.get('csv_log_file')
        db_log_file = task.get('db_log_file')

        # Log information about the task and log file paths
        print(f"Setting up logging for job: {self.job_id}")
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

    def run_training(self, task, train_loader, val_loader, device):
        """Run the training process for a single task and update task status."""
        try:
            self.logger.info("Starting training loop")
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
            early_stopping = False  # Initialize early stopping flag

            optimizer = self.training_service.get_optimizer(model, lr=hyperparams['INITIAL_LR'])
            scheduler = self.training_service.get_scheduler(optimizer, lr_drop_period)

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
                avg_batch_time, train_loss = self.training_service.train_epoch(
                    model, train_loader, optimizer, h_s, h_c, epoch, device, self.stop_requested, task)

                epoch_end_time = time.time()
                epoch_duration = epoch_end_time - epoch_start_time
                formatted_epoch_time = format_time(epoch_duration)  # Convert epoch time to mm:ss format

                if self.stop_requested:
                    self.logger.info("Training stopped by user")
                    print("Training stopped after training phase.")
                    break

                # Only validate at specified frequency
                if epoch == 1 or epoch % valid_freq == 0 or epoch == max_epochs:
                    h_s = torch.zeros(model.num_layers, hyperparams['BATCH_SIZE'], model.hidden_units).to(device)
                    h_c = torch.zeros(model.num_layers, hyperparams['BATCH_SIZE'], model.hidden_units).to(device)

                    val_loss = self.training_service.validate_epoch(model, val_loader, h_s, h_c, epoch, device, self.stop_requested, task)
                    self.logger.info(f"Epoch {epoch} | Train Loss: {train_loss} | Val Loss: {val_loss} | Epoch Time: {formatted_epoch_time}")

                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    delta_t_epoch = (current_time - last_validation_time) / valid_freq
                    last_validation_time = current_time

                    current_lr = optimizer.param_groups[0]['lr']

                    # Update task status to be served to the GUI
                    task_status = {
                        'current_task_index': self.task_index,
                        'progress_data': {
                            'epoch': epoch,
                            'train_loss': train_loss,
                            'val_loss': val_loss,
                            'elapsed_time': elapsed_time,
                            'delta_t_epoch': formatted_epoch_time,
                            'learning_rate': current_lr,
                            'best_val_loss': best_validation_loss,
                        },
                        'status': 'running'
                    }

                    # Store the task status
                    self.current_task_status = task_status

                    if val_loss < best_validation_loss:
                        best_validation_loss = val_loss
                        patience_counter = 0
                        self.save_model(task)
                    else:
                        patience_counter += 1

                    if patience_counter > valid_patience:
                        early_stopping = True
                        print(f"Early stopping at epoch {epoch} due to no improvement.")
                        break

                # Log data to CSV and SQLite after each epoch (whether validated or not)
                print(f"Checking log files for the task: {task['task_id']}: task['csv_log_file'], task['db_log_file']")

                # Save log data to CSV and SQLite
                self.log_to_csv(task, epoch, train_loss, val_loss, elapsed_time, current_lr, best_validation_loss, delta_t_epoch)
                model_memory_usage = torch.cuda.memory_allocated() if torch.cuda.is_available() else sys.getsizeof(model)
                self.log_to_sqlite(
                    task=task,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    best_val_loss=best_validation_loss,
                    elapsed_time=elapsed_time,
                    avg_batch_time=avg_batch_time,
                    early_stopping=early_stopping,
                    model_memory_usage=model_memory_usage,
                )

                scheduler.step()

            if self.stop_requested:
                print("Training was stopped early. Saving Model...")
                self.save_model(task)

            # Mark task as completed
            self.logger.info("Training task completed")
            self.current_task_status['status'] = 'completed'

        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            self.current_task_status = {"status": "error", "error": str(e), "task_id": task['task_id']}


    def get_current_task_status(self):
        current_time = time.time()
        elapsed_time = current_time - self.start_time if hasattr(self, 'start_time') else 0
        elapsed_hms = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

        # Build task status from the stored current_task_status
        task_status = {
            "current_task_index": self.task_index,
            "progress_data": self.current_task_status.get('progress_data', {}),
            "status": self.current_task_status.get('status', 'running'),
            "elapsed_time": elapsed_hms  # Send elapsed time in hh:mm:ss format
        }
        return task_status


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

# Singleton instance of the TrainingTaskManager
training_task_manager = TrainingTaskManager()

# Flask Endpoints for Training Task Manager

# Define the API endpoint for starting the training
@training_task_blueprint.route('/start_training', methods=['POST'])
def start_training():
    """API endpoint to start training tasks."""
    try:
        training_task_manager.start_training()
        return jsonify({"message": "Training started successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Define the API endpoint for stopping the training task
@training_task_blueprint.route('/stop_task', methods=['POST'])
def stop_task():
    """API endpoint to stop the training task."""
    try:
        # Use the singleton instance
        training_task_manager.stop_task()
        return jsonify({'message': 'Task stopped successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Define the API endpoint for serving the current task status
@training_task_blueprint.route('/task_status', methods=['GET'])
def task_status():
    """API endpoint to serve the current task status to the GUI."""
    try:
        # Directly access the singleton instance to get the current task status
        task_status = training_task_manager.get_current_task_status()
        return jsonify(task_status), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500



