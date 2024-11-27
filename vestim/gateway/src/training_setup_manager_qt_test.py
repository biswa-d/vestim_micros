import os
import time
import json
import logging
import torch
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from vestim.gateway.src.hyper_param_manager_qt_test import VEstimHyperParamManager
from vestim.services.model_training.src.LSTM_model_service import LSTMModelService
from vestim.gateway.src.job_manager_qt import JobManager
from vestim.services.model_training.src.training_task_service_test import TrainingTaskService
from vestim.services.model_training.src.data_loader_service_test_h5 import DataLoaderService

class VEstimTrainingSetupManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(VEstimTrainingSetupManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, progress_signal=None, job_manager=None):
        if not hasattr(self, 'initialized'):
            self.logger = logging.getLogger(__name__)
            self.params = None
            self.training_service = TrainingTaskService()
            self.data_loader_service = DataLoaderService()
            self.hyper_param_manager = VEstimHyperParamManager()
            self.lstm_model_service = LSTMModelService()
            self.job_manager = job_manager or JobManager()
            self.progress_signal = progress_signal
            self.training_tasks = []
            self.models = []  # Store the models created from the best configurations
            self.initialized = True
            self.default_ranges = {
                'HIDDEN_UNITS': "32,256",
                'LAYERS': "1,3",
                'DROPOUT_PROB': "0.0,0.5",
                'INITIAL_LR': "1e-5,1e-2",
                'WEIGHT_DECAY': "1e-6,1e-3",
                'LR_DROP_FACTOR': "0.1,0.9",
                'LR_DROP_PERIOD': "5,20",
                'VALID_PATIENCE': "5,30"
            }

    def setup_training(self):
        """
        Set up the training process, including hyperparameter optimization using Ray Tune.
        """
        self.logger.info("Setting up training with Ray Tune...")
        try:
            self.params = self.hyper_param_manager.get_hyper_params()
            self.logger.info(f"Hyperparameters: {self.params}")

            # Emit progress signal to indicate model building is starting
            if self.progress_signal:
                self.progress_signal.emit("Starting hyperparameter optimization with Ray Tune...", "", 0)

            # Define the hyperparameter search space for Ray Tune
            # Define the hyperparameter search space for Ray Tune using the fetched hyperparameters
            search_space = {
                "HIDDEN_UNITS": tune.randint(*self.parse_range(self.params['HIDDEN_UNITS'])),  # Use validated range
                "LAYERS": tune.randint(*self.parse_range(self.params['LAYERS'])),  # Use validated range
                "DROPOUT_PROB": tune.uniform(*self.parse_range(self.params['DROPOUT_PROB'])),  # Use validated range
                "INITIAL_LR": tune.loguniform(*self.parse_range(self.params['INITIAL_LR'])),  # Log-uniform distribution for learning rate
                "WEIGHT_DECAY": tune.loguniform(*self.parse_range(self.params['WEIGHT_DECAY'])),  # Log-uniform for weight decay
                "BATCH_SIZE": tune.choice([int(b) for b in self.params['BATCH_SIZE'].split(',')]),  # Choice for batch size
                "LR_DROP_FACTOR": tune.uniform(*self.parse_range(self.params['LR_DROP_FACTOR'])),  # Learning rate drop factor
                "LR_DROP_PERIOD": tune.randint(*self.parse_range(self.params['LR_DROP_PERIOD'])),  # Learning rate drop period
                "LOOKBACK": tune.randint(*self.parse_range(self.params['LOOKBACK'])),  # Consistent handling as a range
            }



            # Set up a scheduler for Ray Tune
            scheduler = ASHAScheduler(
                metric="val_loss",
                mode="min",
                max_t=int(self.params['MAX_EPOCHS']),
                grace_period=10,
                reduction_factor=2
            )

            # Run the hyperparameter optimization
            analysis = tune.run(
                self.train_and_evaluate_model,
                config=search_space,
                metric="val_loss",
                mode="min",
                scheduler=scheduler,
                num_samples=100,  # Define the number of hyperparameter trials
            )

            # Get the top configurations
            best_trials = analysis.trials[:4]
            best_configs = [trial.config for trial in best_trials]

            self.logger.info(f"Top hyperparameter configurations found: {best_configs}")

            # Build models for the best configurations
            self.build_models(best_configs)

            # Create training tasks based on the best models
            self.create_training_tasks()

        except Exception as e:
            self.logger.error(f"Error during setup: {str(e)}")
            if self.progress_signal:
                self.progress_signal.emit(f"Error during setup: {str(e)}", "", 0)

    def train_and_evaluate_model(self, config):
        """
        Train and evaluate an LSTM model, and report the validation loss.
        """
        # Initialize the TrainingTaskService for training and validation
        training_service = TrainingTaskService()

        # Initialize the DataLoaderService
        data_loader_service = DataLoaderService()

        # Extract hyperparameters from the config
        model_params = {
            "INPUT_SIZE": 3,
            "HIDDEN_UNITS": config["HIDDEN_UNITS"],
            "LAYERS": config["LAYERS"],
            "DROPOUT_PROB": config["DROPOUT_PROB"]
        }

        # Create the LSTM model with the given configuration
        model = self.lstm_model_service.create_lstm_model(model_params).to(training_service.device)

        # Optimizer setup
        optimizer = training_service.get_optimizer(model, config["INITIAL_LR"], weight_decay=config["WEIGHT_DECAY"])

        # Scheduler setup with tuned step_size and gamma
        scheduler = training_service.get_scheduler(optimizer, step_size=config["step_size"], gamma=config["gamma"])

        # Create DataLoaders using DataLoaderService with the configuration from Ray Tune
        folder_path = config.get('DATA_FOLDER_PATH', './data')  # Assume a default folder path
        lookback = config["LOOKBACK"]
        batch_size = config["BATCH_SIZE"]
        num_workers = config.get('NUM_WORKERS', 4)  # Default to 4 workers if not specified

        train_loader, val_loader = data_loader_service.create_data_loaders(
            folder_path=folder_path,
            lookback=lookback,
            batch_size=batch_size,
            num_workers=num_workers,
            train_split=0.7  # Train-validation split can also be tuned if needed
        )

        # Initialize hidden states for LSTM
        h_s, h_c = (
            torch.zeros(config['LAYERS'], batch_size, config['HIDDEN_UNITS']).to(training_service.device),
            torch.zeros(config['LAYERS'], batch_size, config['HIDDEN_UNITS']).to(training_service.device)
        )

        # Training Loop (for a certain number of epochs)
        max_epochs = int(config.get('MAX_EPOCHS', 100))
        stop_requested = False
        for epoch in range(max_epochs):
            # Train for one epoch
            avg_batch_time, train_loss = training_service.train_epoch(
                model, train_loader, optimizer, h_s, h_c, epoch, training_service.device, stop_requested, config)

            # Validate after each epoch
            val_loss = training_service.validate_epoch(
                model, val_loader, h_s, h_c, epoch, training_service.device, stop_requested, config)

            # Adjust learning rate with the scheduler
            scheduler.step()

            # Report validation loss for Ray Tune to optimize
            tune.report(val_loss=val_loss)

            # Optional: Save the model at specific intervals if needed
            if epoch % 10 == 0:
                model_path = os.path.join(self.job_manager.get_job_folder(), f"checkpoint_epoch_{epoch}.pth")
                training_service.save_model(model, model_path)


    def train_model(self, model, config):
        """
        Placeholder training function.
        """
        # Extract training-related hyperparameters
        initial_lr = config["INITIAL_LR"]
        weight_decay = config["WEIGHT_DECAY"]
        batch_size = config["BATCH_SIZE"]
        lr_drop_factor = config["LR_DROP_FACTOR"]
        lr_drop_period = config["LR_DROP_PERIOD"]
        valid_patience = config["VALID_PATIENCE"]

        # Add your actual training logic here (training loop, optimizer setup, learning rate schedule, etc.)
        val_loss = 0.05  # Replace with actual validation loss after training
        return val_loss

    def build_models(self, best_configs):
        """
        Build and store the LSTM models based on the best configurations found by Ray Tune.
        """
        self.logger.info("Building models for the best configurations...")
        for config in best_configs:
            # Create model directory based on hyperparameters
            model_dir = os.path.join(
                self.job_manager.get_job_folder(),
                'models',
                f'model_lstm_hu_{config["HIDDEN_UNITS"]}_layers_{config["LAYERS"]}_dropout_{config["DROPOUT_PROB"]}'
            )
            os.makedirs(model_dir, exist_ok=True)

            model_name = f"model_lstm_hu_{config['HIDDEN_UNITS']}_layers_{config['LAYERS']}_dropout_{config['DROPOUT_PROB']}.pth"
            model_path = os.path.join(model_dir, model_name)

            # Create and save the LSTM model
            model = self.lstm_model_service.build_lstm_model({
                "INPUT_SIZE": 3,
                "HIDDEN_UNITS": config["HIDDEN_UNITS"],
                "LAYERS": config["LAYERS"],
                "DROPOUT_PROB": config["DROPOUT_PROB"]
            }, model_path)

            # Store model information
            self.models.append({
                'model': model,
                'model_dir': model_dir,
                'hyperparams': config,
                'model_path': model_path
            })

        self.logger.info("Model building complete.")

    def create_training_tasks(self):
        """
        Create training tasks by combining the best models and training hyperparameters.
        """
        self.logger.info("Creating training tasks from the best models...")
        task_list = []

        timestamp = time.strftime("%Y%m%d%H%M%S")
        task_counter = 1

        for model_task in self.models:
            model = model_task['model']
            model_metadata = model_task['hyperparams']
            model_metadata.update({
                'model_type': 'LSTMModel',
                'input_size': model.input_size,
                'num_layers': model.num_layers,
                'dropout_prob': model.dropout_prob
            })
            num_learnable_params = self.calculate_learnable_parameters(
                model.num_layers, model.input_size, model_metadata['HIDDEN_UNITS']
            )

            task_id = f"task_{timestamp}_{task_counter}"
            task_counter += 1

            # Create a unique directory for each task
            task_dir = os.path.join(model_task['model_dir'], f'task_{task_id}')
            os.makedirs(task_dir, exist_ok=True)

            csv_log_file = os.path.join(task_dir, f"{task_id}_train_log.csv")
            db_log_file = os.path.join(task_dir, f"{task_id}_train_log.db")

            # Task information
            task_info = {
                'task_id': task_id,
                "task_dir": task_dir,
                'model': model,
                'model_metadata': model_metadata,
                'data_loader_params': {
                    'lookback': model_metadata['LOOKBACK'],
                    'batch_size': model_metadata['BATCH_SIZE'],
                },
                'model_dir': task_dir,
                'model_path': os.path.join(task_dir, 'model.pth'),
                'hyperparams': {
                    **model_metadata,
                    'MAX_EPOCHS': self.params['MAX_EPOCHS'],
                    'NUM_LEARNABLE_PARAMS': num_learnable_params,
                },
                'csv_log_file': csv_log_file,
                'db_log_file': db_log_file
            }

            # Append task to the list
            task_list.append(task_info)

            # Save task info to disk
            task_info_file = os.path.join(task_dir, 'task_info.json')
            with open(task_info_file, 'w') as f:
                json.dump({k: v for k, v in task_info.items() if k != 'model'}, f, indent=4)

        self.training_tasks = task_list

        # Save task summary
        tasks_summary_file = os.path.join(self.job_manager.get_job_folder(), 'training_tasks_summary.json')
        with open(tasks_summary_file, 'w') as f:
            json.dump([{k: v for k, v in task.items() if k != 'model'} for task in self.training_tasks], f, indent=4)

        # Emit progress signal
        task_count = len(self.training_tasks)
        if self.progress_signal:
            self.logger.info(f"Created {task_count} training tasks.")
            self.progress_signal.emit(f"Created {task_count} training tasks and saved to disk.", self.job_manager.get_job_folder(), task_count)

    def parse_range(self, range_str):
        """
        Parse a range string (e.g., "16,128") into a tuple (16, 128).

        :param range_str: A string containing the range in "min,max" format.
        :return: A tuple containing (min, max) as floats or integers.
        """
        try:
            min_val, max_val = map(float, range_str.split(","))
            if min_val >= max_val:
                raise ValueError("Minimum value must be less than the maximum value.")
            return min_val, max_val
        except ValueError as e:
            self.logger.error(f"Invalid range format for '{range_str}': {e}")
            raise


    def calculate_learnable_parameters(self, layers, input_size, hidden_units):
        """
        Calculate the number of learnable parameters for an LSTM model.
        """
        learnable_params = (4 * (input_size + hidden_units) * hidden_units) + (4 * hidden_units)
        for _ in range(1, layers):
            learnable_params += (4 * (hidden_units + hidden_units) * hidden_units) + (4 * hidden_units)
        learnable_params += hidden_units * 1 + 1  # Output layer params
        return learnable_params
