import os, uuid, time
import json
from vestim.gateway.src.hyper_param_manager_qt_test import VEstimHyperParamManager
from vestim.services.model_training.src.LSTM_model_service import LSTMModelService
from vestim.gateway.src.job_manager_qt import JobManager
import logging

class VEstimTrainingSetupManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(VEstimTrainingSetupManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, progress_signal=None, job_manager=None):
        if not hasattr(self, 'initialized'):  # Ensure initialization only happens once
            self.logger = logging.getLogger(__name__)  # Initialize logger
            self.params = None
            self.current_hyper_params = None
            self.hyper_param_manager = VEstimHyperParamManager()  # Initialize your hyperparameter manager here
            self.lstm_model_service = LSTMModelService()  # Initialize your model service here
            self.job_manager = job_manager  # JobManager should be passed in or initialized separately
            self.models = []  # Store model information
            self.training_tasks = []  # Store created tasks
            self.progress_signal = progress_signal  # Signal to communicate progress with the GUI
            self.initialized = True  # Mark as initialized

    def setup_training(self):
        print("Setting up training by the manager...")
        self.logger.info("Setting up training...")
        self.logger.info("Fetching hyperparameters...")
        """Set up the training process, including building models and creating training tasks."""
        try:
            print("Fetching hyperparameters...")
            self.params = self.hyper_param_manager.get_hyper_params()
            self.current_hyper_params = self.params
            self.logger.info(f"Params after updating: {self.current_hyper_params}")

            # Emit progress signal to indicate model building is starting
            if self.progress_signal:
                self.progress_signal.emit("Building models...", "", 0)

            # Build models
            self.build_models()

            # Emit progress signal to indicate training task creation is starting
            if self.progress_signal:
                self.progress_signal.emit("Creating training tasks...", "", 0)

            # Create training tasks
            self.create_training_tasks()

            # Emit final progress signal after tasks are created
            task_count = len(self.training_tasks)
            if self.progress_signal:
                self.progress_signal.emit(
                    f"Setup complete! Task info saved in {self.job_manager.get_job_folder()}.",
                    self.job_manager.get_job_folder(),
                    task_count
                )

        except Exception as e:
            self.logger.error(f"Error during setup: {str(e)}")
            # Handle any error during setup and pass it to the GUI
            if self.progress_signal:
                self.progress_signal.emit(f"Error during setup: {str(e)}", "", 0)

    def build_models(self):
        """Build and store the LSTM models based on hyperparameters."""
        # Parse comma-separated values for hidden_units, layers, and dropout_prob
        hidden_units_list = [int(h) for h in self.params['HIDDEN_UNITS'].split(',')]
        layers_list = [int(l) for l in self.params['LAYERS'].split(',')]  # Allow multiple layers
        dropout_prob_list = [float(d) for d in self.params.get('DROPOUT_PROB', '0.5').split(',')]  # Parse multiple dropout values

        # Iterate over all combinations of hidden_units, layers, and dropout_prob
        for hidden_units in hidden_units_list:
            for layers in layers_list:
                for dropout_prob in dropout_prob_list:  # Iterate through dropout probabilities as well
                    self.logger.info(f"Creating model with hidden_units: {hidden_units}, layers: {layers}, dropout_prob: {dropout_prob}")
                    
                    # Create model directory based on hidden_units, layers, and dropout_prob
                    model_dir = os.path.join(
                        self.job_manager.get_job_folder(), 
                        'models', 
                        f'model_lstm_hu_{hidden_units}_layers_{layers}_dropout_{dropout_prob}'
                    )
                    os.makedirs(model_dir, exist_ok=True)

                    model_name = f"model_lstm_hu_{hidden_units}_layers_{layers}_dropout_{dropout_prob}.pth"
                    model_path = os.path.join(model_dir, model_name)

                    # Model parameters
                    model_params = {
                        "INPUT_SIZE": 3,  # Modify as needed
                        "HIDDEN_UNITS": hidden_units,
                        "LAYERS": layers,
                        "DROPOUT_PROB": dropout_prob  # Pass the dropout probability to the model
                    }

                    # Create and save the LSTM model
                    model = self.lstm_model_service.create_and_save_lstm_model(model_params, model_path)

                    # Store model information
                    self.models.append({
                        'model': model,
                        'model_dir': model_dir,
                        'hyperparams': {
                            'LAYERS': layers,
                            'HIDDEN_UNITS': hidden_units,
                            'DROPOUT_PROB': dropout_prob,
                            'model_path': model_path
                        }
                    })
            
        self.logger.info("Model building complete.")



    def create_training_tasks(self):
        """
        Create a list of training tasks by combining models and relevant training hyperparameters.
        This method will also save the task information to disk for future use.
        """
        self.logger.info("Creating training tasks...")
        task_list = []  # Initialize a list to store tasks

        # Retrieve relevant hyperparameters for training
        learning_rates = [float(lr) for lr in self.current_hyper_params['INITIAL_LR'].split(',')]
        lr_drop_factors = [float(drop_factor) for drop_factor in self.current_hyper_params['LR_DROP_FACTOR'].split(',')]
        weight_decays = [float(wd) for wd in self.current_hyper_params['WEIGHT_DECAY'].split(',')]
        lr_drop_periods = [int(drop) for drop in self.current_hyper_params['LR_DROP_PERIOD'].split(',')]
        valid_patience_values = [int(vp) for vp in self.current_hyper_params['VALID_PATIENCE'].split(',')]
        repetitions = int(self.current_hyper_params['REPETITIONS'])
        lookbacks = [int(lb) for lb in self.current_hyper_params['LOOKBACK'].split(',')]
        batch_sizes = [int(bs) for bs in self.current_hyper_params['BATCH_SIZE'].split(',')]
        max_epochs = int(self.current_hyper_params['MAX_EPOCHS'])  # Ensure MAX_EPOCHS is included

        #set the logic for task_id
        timestamp = time.strftime("%Y%m%d%H%M%S")  # Format timestamp as YYYYMMDDHHMMSS
        task_counter = 1
        # Iterate through each model and create tasks
        for model_task in self.models:
            model = model_task['model']
            model_metadata = {
                'model_type': 'LSTMModel',
                'input_size': model.input_size,
                'hidden_units': model.hidden_units,
                'num_layers': model.num_layers,
                'dropout_prob': model.dropout_prob
            }
            num_learnable_params = self.calculate_learnable_parameters(model.num_layers, model.input_size, model.hidden_units)

            # Iterate through hyperparameters
            for lr in learning_rates:
                for drop_factor in lr_drop_factors:
                    for drop_period in lr_drop_periods:
                        for weight_decay in weight_decays:
                            for patience in valid_patience_values:
                                for lookback in lookbacks:
                                    for batch_size in batch_sizes:
                                        for rep in range(1, repetitions + 1):

                                            #create an unique task_id
                                            task_id = f"task_{timestamp}_{task_counter}"
                                            task_counter += 1  # Increment for each subsequent task
                                            
                                            # Create a unique directory for each task based on all parameters
                                            task_dir = os.path.join(
                                                model_task['model_dir'],
                                                f'lr_{lr}_drop_{drop_period}_factor_{drop_factor}_patience_{patience}_rep_{rep}_lookback_{lookback}_batch_{batch_size}'
                                            )
                                            os.makedirs(task_dir, exist_ok=True)
                                            
                                            # Create the log files at this stage
                                            csv_log_file = os.path.join(task_dir, f"{task_id}_train_log.csv")
                                            db_log_file = os.path.join(task_dir, f"{task_id}_train_log.db")

                                            # Define task information
                                            task_info = {
                                                'task_id': task_id,
                                                "task_dir": task_dir,
                                                'model': model,
                                                'model_metadata': model_metadata,  # Use metadata instead of the full model
                                                'data_loader_params': {
                                                    'lookback': lookback,
                                                    'batch_size': batch_size,
                                                },
                                                'model_dir': task_dir,
                                                'model_path': os.path.join(task_dir, 'model.pth'),
                                                'hyperparams': {
                                                    'LAYERS': model_metadata['num_layers'],
                                                    'HIDDEN_UNITS': model_metadata['hidden_units'],
                                                    'DROPOUT_PROB': model_metadata['dropout_prob'],
                                                    'WEIGHT_DECAY': weight_decay,
                                                    'BATCH_SIZE': batch_size,
                                                    'LOOKBACK': lookback,
                                                    'INITIAL_LR': lr,
                                                    'LR_DROP_FACTOR': drop_factor,
                                                    'LR_DROP_PERIOD': drop_period,
                                                    'VALID_PATIENCE': patience,
                                                    'ValidFrequency': self.current_hyper_params['ValidFrequency'],
                                                    'REPETITIONS': rep,
                                                    'MAX_EPOCHS': max_epochs,  # Include MAX_EPOCHS here
                                                    'NUM_LEARNABLE_PARAMS': num_learnable_params,
                                                },
                                                'csv_log_file': csv_log_file,
                                                'db_log_file': db_log_file  # Set log files here
                                            }

                                            # Append the task to the task list
                                            task_list.append(task_info)

                                            # Save the task info to disk as a JSON file
                                            task_info_file = os.path.join(task_dir, 'task_info.json')
                                            with open(task_info_file, 'w') as f:
                                                json.dump({k: v for k, v in task_info.items() if k != 'model'}, f, indent=4)

        # Replace the existing training tasks with the newly created task list
        self.training_tasks = task_list

        # Optionally, save the entire task list for future reference at the root level
        tasks_summary_file = os.path.join(self.job_manager.get_job_folder(), 'training_tasks_summary.json')
        with open(tasks_summary_file, 'w') as f:
            json.dump([{k: v for k, v in task.items() if k != 'model'} for task in self.training_tasks], f, indent=4)

        # Emit progress signal to notify the GUI
        task_count = len(self.training_tasks)
        if self.progress_signal:
            self.logger.info(f"Created {len(self.training_tasks)} training tasks.")
            print(f"Created {len(self.training_tasks)} training tasks.")
            self.progress_signal.emit(f"Created {task_count} training tasks and saved to disk.", self.job_manager.get_job_folder(), task_count)

    def calculate_learnable_parameters(self, layers, input_size, hidden_units):
        """
        Calculate the number of learnable parameters for an LSTM model.

        :param layers: Number of layers (an integer representing the number of LSTM layers)
        :param input_size: The size of the input features (e.g., 3 for [SOC, Current, Temp])
        :param hidden_units: An integer representing the number of hidden units in each layer
        :return: Total number of learnable parameters
        """

        # Initialize the number of parameters
        learnable_params = 0

        # Input-to-hidden weights for the first layer (4 * hidden_units * (input_size + hidden_units))
        # We account for 4 gates (input, forget, output, and candidate gates)
        input_layer_params = 4 * (input_size + hidden_units) * hidden_units

        # Add bias terms for each gate in the first layer
        input_layer_bias = 4 * hidden_units

        learnable_params += input_layer_params + input_layer_bias

        # For each additional LSTM layer, it's hidden_units -> hidden_units
        for i in range(1, layers):
            hidden_layer_params = 4 * (hidden_units + hidden_units) * hidden_units
            hidden_layer_bias = 4 * hidden_units
            learnable_params += hidden_layer_params + hidden_layer_bias

        # Output layer (assuming 1 output)
        output_size = 1
        output_layer_params = hidden_units * output_size
        output_layer_bias = output_size  # 1 bias for the output layer
        learnable_params += output_layer_params + output_layer_bias

        return learnable_params

    def update_task(self, task_id, db_log_file=None, csv_log_file=None):
        """Update a specific task in the manager."""
        for task in self.training_tasks:
            if task['task_id'] == task_id:
                if db_log_file:
                    task['db_log_file'] = db_log_file
                if csv_log_file:
                    task['csv_log_file'] = csv_log_file
                # Additional fields can be updated similarly
                break

    def get_task_list(self):
        """Returns the list of training tasks."""
        return self.training_tasks


