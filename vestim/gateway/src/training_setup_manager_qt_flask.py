from flask import Blueprint, jsonify, request
import os, uuid, json
from vestim.gateway.src.hyper_param_manager_qt_flask import VEstimHyperParamManager
from vestim.services.model_training.src.LSTM_model_service import LSTMModelService
from vestim.gateway.src.job_manager_qt_flask import JobManager
import logging

# Create a Flask Blueprint
training_setup_blueprint = Blueprint('training_setup', __name__)

class VEstimTrainingSetupManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(VEstimTrainingSetupManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, job_manager=None):
        if not hasattr(self, 'initialized'):  # Ensure initialization only happens once
            self.logger = logging.getLogger(__name__)  # Initialize logger
            self.params = None
            self.current_hyper_params = None
            self.hyper_param_manager = VEstimHyperParamManager()  # Initialize hyperparameter manager
            self.lstm_model_service = LSTMModelService()  # Initialize model service
            self.job_manager = job_manager or JobManager()  # JobManager should be passed in or initialized separately
            self.models = []  # Store model information
            self.training_tasks = []  # Store created tasks
            self.initialized = True  # Mark as initialized

    def setup_training(self):
        """Set up training by fetching hyperparameters, building models, and creating training tasks."""
        self.logger.info("Setting up training...")
        try:
            self.params = self.hyper_param_manager.get_hyper_params()
            self.current_hyper_params = self.params
            self.build_models()
            self.create_training_tasks()

            task_count = len(self.training_tasks)
            return {
                "message": f"Setup complete! {task_count} tasks created.",
                "job_folder": self.job_manager.get_job_folder(),
                "task_count": task_count
            }, 200

        except Exception as e:
            self.logger.error(f"Error during setup: {str(e)}")
            return {"error": str(e)}, 500

    def build_models(self):
        """Build and store the LSTM models based on hyperparameters."""
        hidden_units_list = [int(h) for h in self.params['HIDDEN_UNITS'].split(',')]
        layers_list = [int(l) for l in self.params['LAYERS'].split(',')]

        for hidden_units in hidden_units_list:
            for layers in layers_list:
                self.logger.info(f"Creating model with hidden_units: {hidden_units}, layers: {layers}")
                model_dir = os.path.join(self.job_manager.get_job_folder(), 'models', f'model_lstm_hu_{hidden_units}_layers_{layers}')
                os.makedirs(model_dir, exist_ok=True)

                model_name = f"model_lstm_hu_{hidden_units}_layers_{layers}.pth"
                model_path = os.path.join(model_dir, model_name)

                model_params = {
                    "INPUT_SIZE": 3,  # Modify as needed
                    "HIDDEN_UNITS": hidden_units,
                    "LAYERS": layers
                }

                model = self.lstm_model_service.create_and_save_lstm_model(model_params, model_path)

                self.models.append({
                    'model': model,
                    'model_dir': model_dir,
                    'hyperparams': {
                        'LAYERS': layers,
                        'HIDDEN_UNITS': hidden_units,
                        'model_path': model_path
                    }
                })

        self.logger.info("Model building complete.")

    def create_training_tasks(self):
        """Create a list of training tasks by combining models and relevant training hyperparameters."""
        self.logger.info("Creating training tasks...")
        task_list = []
        learning_rates = [float(lr) for lr in self.current_hyper_params['INITIAL_LR'].split(',')]
        lr_drop_factors = [float(drop_factor) for drop_factor in self.current_hyper_params['LR_DROP_FACTOR'].split(',')]
        lr_drop_periods = [int(drop) for drop in self.current_hyper_params['LR_DROP_PERIOD'].split(',')]
        valid_patience_values = [int(vp) for vp in self.current_hyper_params['VALID_PATIENCE'].split(',')]
        repetitions = int(self.current_hyper_params['REPETITIONS'])
        lookbacks = [int(lb) for lb in self.current_hyper_params['LOOKBACK'].split(',')]
        batch_sizes = [int(bs) for bs in self.current_hyper_params['BATCH_SIZE'].split(',')]
        max_epochs = int(self.current_hyper_params['MAX_EPOCHS'])

        for model_task in self.models:
            model = model_task['model']
            model_metadata = {
                'model_type': 'LSTMModel',
                'input_size': model.input_size,
                'hidden_units': model.hidden_units,
                'num_layers': model.num_layers,
            }
            num_learnable_params = self.calculate_learnable_parameters(model.num_layers, model.input_size, model.hidden_units)

            for lr in learning_rates:
                for drop_factor in lr_drop_factors:
                    for drop_period in lr_drop_periods:
                        for patience in valid_patience_values:
                            for lookback in lookbacks:
                                for batch_size in batch_sizes:
                                    for rep in range(1, repetitions + 1):
                                        task_id = str(uuid.uuid4())
                                        task_dir = os.path.join(
                                            model_task['model_dir'],
                                            f'lr_{lr}_drop_{drop_period}_factor_{drop_factor}_patience_{patience}_rep_{rep}_lookback_{lookback}_batch_{batch_size}'
                                        )
                                        os.makedirs(task_dir, exist_ok=True)
                                        task_info = {
                                            'task_id': task_id,
                                            'model': model,
                                            'model_metadata': model_metadata,
                                            'data_loader_params': {'lookback': lookback, 'batch_size': batch_size},
                                            'model_dir': task_dir,
                                            'model_path': os.path.join(task_dir, 'model.pth'),
                                            'hyperparams': {
                                                'LAYERS': model_metadata['num_layers'],
                                                'HIDDEN_UNITS': model_metadata['hidden_units'],
                                                'BATCH_SIZE': batch_size,
                                                'LOOKBACK': lookback,
                                                'INITIAL_LR': lr,
                                                'LR_DROP_FACTOR': drop_factor,
                                                'LR_DROP_PERIOD': drop_period,
                                                'VALID_PATIENCE': patience,
                                                'ValidFrequency': self.current_hyper_params['ValidFrequency'],
                                                'REPETITIONS': rep,
                                                'MAX_EPOCHS': max_epochs,
                                                'NUM_LEARNABLE_PARAMS': num_learnable_params,
                                            }
                                        }
                                        task_list.append(task_info)
                                        with open(os.path.join(task_dir, 'task_info.json'), 'w') as f:
                                            json.dump({k: v for k, v in task_info.items() if k != 'model'}, f, indent=4)

        self.training_tasks = task_list
        self.logger.info(f"Created {len(self.training_tasks)} training tasks.")

    def calculate_learnable_parameters(self, layers, input_size, hidden_units):
        """Calculate the number of learnable parameters for an LSTM model."""
        learnable_params = 4 * (input_size + 1) * hidden_units
        for i in range(1, layers):
            learnable_params += 4 * (hidden_units + 1) * hidden_units
        learnable_params += hidden_units * 1  # Last hidden layer to output
        return learnable_params

    def get_task_list(self):
        """Returns the list of training tasks."""
        return self.training_tasks

# Initialize the manager
training_setup_manager = VEstimTrainingSetupManager()

# Flask Endpoints for Training Setup Manager
@training_setup_blueprint.route('/setup_training', methods=['POST'])
def setup_training():
    """Endpoint to set up training and create tasks."""
    result, status = training_setup_manager.setup_training()
    return jsonify(result), status

@training_setup_blueprint.route('/get_tasks', methods=['GET'])
def get_tasks():
    """Endpoint to get the list of created training tasks."""
    return jsonify(training_setup_manager.get_task_list()), 200

@training_setup_blueprint.route('/update_task', methods=['POST'])
def update_task():
    """Endpoint to update a specific task with new log files."""
    task_id = request.json.get('task_id')
    db_log_file = request.json.get('db_log_file')
    csv_log_file = request.json.get('csv_log_file')
    training_setup_manager.update_task(task_id, db_log_file, csv_log_file)
    return jsonify({'message': 'Task updated successfully'}), 200
