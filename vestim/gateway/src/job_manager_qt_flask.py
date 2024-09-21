from flask import Flask, jsonify, request, Blueprint
import os
from datetime import datetime
from vestim.config import OUTPUT_DIR

job_manager_blueprint = Blueprint('job_manager', __name__)

class JobManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(JobManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'job_id'):  # Ensure the attributes are initialized once
            self.job_id = None

    def create_new_job(self):
        """Generates a new job ID based on the current timestamp and initializes job directories."""
        self.job_id = f"job_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        job_folder = os.path.join(OUTPUT_DIR, self.job_id)
        os.makedirs(job_folder, exist_ok=True)
        return self.job_id, job_folder

    def get_job_id(self):
        """Returns the current job ID."""
        return self.job_id

    def get_job_folder(self):
        """Returns the path to the current job folder."""
        if self.job_id:
            return os.path.join(OUTPUT_DIR, self.job_id)
        return None
    
    def get_train_folder(self):
        """Returns the path to the train processed data folder."""
        if self.job_id:
            return os.path.join(self.get_job_folder(), 'train', 'processed_data')
        return None

    def get_test_folder(self):
        """Returns the path to the test processed data folder."""
        if self.job_id:
            return os.path.join(self.get_job_folder(), 'test', 'processed_data')
        return None
    
    def get_test_results_folder(self):
        """Returns the path to the test results folder."""
        if self.job_id:
            results_folder = os.path.join(self.get_job_folder(), 'test', 'results')
            os.makedirs(results_folder, exist_ok=True)
            return results_folder
        return None

# Create the JobManager instance
job_manager = JobManager()

### Flask Endpoints ###

@job_manager_blueprint.route('/create_job', methods=['POST'])
def create_job():
    """Endpoint to create a new job."""
    job_id, job_folder = job_manager.create_new_job()
    return jsonify({"job_id": job_id, "job_folder": job_folder}), 200

@job_manager_blueprint.route('/get_job_id', methods=['GET'])
def get_job_id():
    """Endpoint to get the current job ID."""
    job_id = job_manager.get_job_id()
    if job_id:
        return jsonify({"job_id": job_id}), 200
    return jsonify({"error": "No job created yet"}), 404

@job_manager_blueprint.route('/get_job_folder', methods=['GET'])
def get_job_folder():
    """Endpoint to get the current job folder."""
    job_folder = job_manager.get_job_folder()
    if job_folder:
        return jsonify({"job_folder": job_folder}), 200
    return jsonify({"error": "No job created yet"}), 404

@job_manager_blueprint.route('/get_train_folder', methods=['GET'])
def get_train_folder():
    """Endpoint to get the train processed data folder."""
    train_folder = job_manager.get_train_folder()
    if train_folder:
        return jsonify({"train_folder": train_folder}), 200
    return jsonify({"error": "No job created yet"}), 404

@job_manager_blueprint.route('/get_test_folder', methods=['GET'])
def get_test_folder():
    """Endpoint to get the test processed data folder."""
    test_folder = job_manager.get_test_folder()
    if test_folder:
        return jsonify({"test_folder": test_folder}), 200
    return jsonify({"error": "No job created yet"}), 404

@job_manager_blueprint.route('/get_test_results_folder', methods=['GET'])
def get_test_results_folder():
    """Endpoint to get the test results folder."""
    results_folder = job_manager.get_test_results_folder()
    if results_folder:
        return jsonify({"results_folder": results_folder}), 200
    return jsonify({"error": "No job created yet"}), 404
