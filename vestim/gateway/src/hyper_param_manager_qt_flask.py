from flask import Flask, jsonify, request, Blueprint, requests
import os
import json
from vestim.gateway.src.job_manager_qt_flask import JobManager  # Assuming job_manager_flask has the job_manager instance
import logging

hyper_param_manager_blueprint = Blueprint('hyper_param_manager', __name__)
logger = logging.getLogger(__name__)


class VEstimHyperParamManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(VEstimHyperParamManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.logger = logger
            self.current_params = {}  # Initialize current_params
            self.initialized = True
            self.job_folder = None
            self.logger.info("VEstimHyperParamManager initialized.")
    
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
            
    def load_params(self, filepath):
        """Load and validate parameters from a JSON file."""
        self.logger.info(f"Loading parameters from {filepath}")
        with open(filepath, 'r') as file:
            params = json.load(file)
            validated_params = self.validate_and_normalize_params(params)
            self.current_params = validated_params
            self.logger.info("Parameters successfully loaded and validated.")
        return validated_params

    def validate_and_normalize_params(self, params):
        """Validate the parameter values without altering them."""
        validated_params = {}
        
        for key, value in params.items():
            if isinstance(value, str):
                value_list = [v.strip() for v in value.replace(',', ' ').split() if v]
                if key in ['LAYERS', 'HIDDEN_UNITS', 'BATCH_SIZE', 'MAX_EPOCHS', 'LR_DROP_PERIOD', 'VALID_PATIENCE', 'ValidFrequency', 'LOOKBACK', 'REPETITIONS']:
                    if not all(v.isdigit() for v in value_list):
                        self.logger.error(f"Invalid value for {key}: {value_list} (expected integers)")
                        raise ValueError(f"Invalid value for {key}: Expected integers, got {value_list}")
                    validated_params[key] = value

                elif key in ['INITIAL_LR', 'LR_DROP_FACTOR']:
                    try:
                        [float(v) for v in value_list]
                    except ValueError:
                        self.logger.error(f"Invalid value for {key}: {value_list} (expected floats)")
                        raise ValueError(f"Invalid value for {key}: Expected floats, got {value_list}")
                    validated_params[key] = value

                else:
                    validated_params[key] = value
            elif isinstance(value, list):
                validated_params[key] = value
            else:
                validated_params[key] = value
        self.logger.info("Parameter validation complete.")
        return validated_params

    def save_params(self):
        """Save the current parameters to the job folder."""
        self.fetch_job_folder()
        job_folder = self.job_folder
        if job_folder and self.current_params:
            params_file = os.path.join(job_folder, 'hyperparams.json')
            with open(params_file, 'w') as file:
                json.dump(self.current_params, file, indent=4)
                self.logger.info("Parameters successfully saved.")
        else:
            self.logger.error("Failed to save parameters: Job folder or current parameters are not set.")
            raise ValueError("Job folder is not set or current parameters are not available.")

    def update_params(self, new_params):
        """Update the current parameters with new values."""
        validated_params = self.validate_and_normalize_params(new_params)
        self.current_params.update(validated_params)
        self.logger.info("Parameters successfully updated.")

    def get_current_params(self):
        """Load the parameters from the saved JSON file in the job folder."""
        job_folder = self.job_folder
        params_file = os.path.join(job_folder, 'hyperparams.json')
        
        if os.path.exists(params_file):
            with open(params_file, 'r') as file:
                current_params = json.load(file)
                self.current_params = current_params
                return current_params
        else:
            self.logger.error(f"Hyperparameters file not found in {job_folder}")
            raise FileNotFoundError("Hyperparameters JSON file not found in the job folder.")

    def get_hyper_params(self):
        """Return the current hyperparameters stored in memory."""
        if self.current_params:
            self.logger.info("Returning current hyperparameters.")
            return self.current_params
        else:
            self.logger.error("No current parameters available in memory.")
            raise ValueError("No current parameters are available in memory.")

# Initialize the manager
hyper_param_manager = VEstimHyperParamManager()

@hyper_param_manager_blueprint.route('/load_params', methods=['POST'])
def load_params():
    filepath = request.json.get('filepath')
    if not filepath:
        return jsonify({'error': 'Filepath is required'}), 400
    
    try:
        params = hyper_param_manager.load_params(filepath)
        return jsonify(params), 200
    except Exception as e:
        logger.error(f"Error loading parameters: {str(e)}")
        return jsonify({'error': str(e)}), 500

@hyper_param_manager_blueprint.route('/save_params', methods=['POST'])
def save_params():
    try:
        hyper_param_manager.save_params()
        return jsonify({'message': 'Parameters saved successfully'}), 200
    except Exception as e:
        logger.error(f"Error saving parameters: {str(e)}")
        return jsonify({'error': str(e)}), 500

@hyper_param_manager_blueprint.route('/update_params', methods=['POST'])
def update_params():
    new_params = request.json.get('params')
    if not new_params:
        return jsonify({'error': 'Params are required'}), 400
    
    try:
        hyper_param_manager.update_params(new_params)
        return jsonify({'message': 'Parameters updated successfully'}), 200
    except Exception as e:
        logger.error(f"Error updating parameters: {str(e)}")
        return jsonify({'error': str(e)}), 500

@hyper_param_manager_blueprint.route('/get_params', methods=['GET'])
def get_params():
    try:
        current_params = hyper_param_manager.get_current_params()
        return jsonify(current_params), 200
    except Exception as e:
        logger.error(f"Error getting parameters: {str(e)}")
        return jsonify({'error': str(e)}), 500

