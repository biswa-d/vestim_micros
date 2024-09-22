from flask import Flask, jsonify, request, Blueprint
import os
from datetime import datetime
from threading import Thread
import json
from queue import Queue
from vestim.config import OUTPUT_DIR
from vestim.services.data_processor.src.data_processor_qt_digatron import DataProcessorDigatron
from vestim.services.data_processor.src.data_processor_qt_tesla import DataProcessorTesla
from vestim.services.data_processor.src.data_processor_qt_pouch import DataProcessorPouch

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
        self.processing_status = {"status": "Not Started", "progress": 0}  # Status tracker
        # self.data_processor_digatron = DataProcessorDigatron()  # Initialize DataProcessor
        # self.data_processor_tesla = DataProcessorTesla()  # Initialize DataProcessor
        # self.data_processor_pouch = DataProcessorPouch()

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

    def organize_files(self, train_files, test_files, data_processor, job_folder):
        """Organizes and converts files into the job folder."""
        print(f"Starting file organization using {data_processor}...")
        self.processing_status = {"status_message": "Processing...", "progress": 0}  # Reset the status

        # Implement threading to process files asynchronously
        def process_files():
            try:
                if data_processor == "Digatron":
                    processor = DataProcessorDigatron()
                elif data_processor == "Tesla":
                    processor = DataProcessorTesla()
                elif data_processor == "Pouch":
                    processor = DataProcessorPouch()
                else:
                    raise ValueError(f"Invalid data processor: {data_processor}")

                total_files = len(train_files) + len(test_files)
                processed_files = 0

                def update_progress():
                    nonlocal processed_files
                    processed_files += 1
                    self.processing_status["progress"] = int((processed_files / total_files) * 100)
                    print(f"Progress: {self.processing_status['progress']}%")

                # Call organize and convert method of the processor
                processor.organize_and_convert_files(train_files, test_files, job_folder, update_progress)
                self.processing_status["status_message"] = "Completed"
                self.processing_status["progress"] = 100
                print("Files processed successfully.")
            except Exception as e:
                self.processing_status["status_message"] = f"Error: {str(e)}"
                self.processing_status["progress"] = 100
                print(f"Error occurred during file organization: {e}")

        # Start a new thread for processing files
        self.processing_thread = Thread(target=process_files)
        self.processing_thread.start()


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

@job_manager_blueprint.route('/organize_files', methods=['POST'])
def organize_files():
    """Endpoint to organize and convert files into the job folder."""
    try:
        data = request.get_json()
        train_files = data.get('train_files', [])
        test_files = data.get('test_files', [])
        data_processor = data.get('data_processor', '')
        job_folder = data.get('job_folder', '')

        # Validate input
        if not train_files or not test_files or not data_processor or not job_folder:
            return jsonify({"error": "Invalid input data"}), 400

        # Call the organize files method of the JobManager
        job_manager.organize_files(train_files, test_files, data_processor, job_folder)

        return jsonify({"message": "File processing started"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@job_manager_blueprint.route('/processing_status', methods=['GET'])
def processing_status():
    """Endpoint to check the processing status."""
    status = job_manager.check_processing_status()
    return jsonify(status), 200

