import os
import json
import time
import uuid
from datetime import datetime
from multiprocessing import Process, Queue
import threading
from vestim.config import OUTPUT_DIR
from vestim.logger_config import configure_job_specific_logging
import logging
import shutil
from vestim.backend.src.managers.hyper_param_manager_qt import VEstimHyperParamManager
from vestim.backend.src.managers.training_setup_manager_qt import VEstimTrainingSetupManager

class JobManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(JobManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.jobs = {}  # In-memory job registry
            self.job_managers = {} # For non-serializable manager objects
            self.status_queue = Queue()
            self.logger = logging.getLogger(__name__)
            
            # Start a thread to listen for status updates from job processes
            self.status_listener_thread = threading.Thread(target=self._listen_for_status_updates, daemon=True)
            self.status_listener_thread.start()

            # Ensure output directory exists
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            self.job_registry_file = os.path.join(OUTPUT_DIR, 'job_registry.json')
            self._load_jobs_from_registry()

    def _listen_for_status_updates(self):
        """Listens for status updates from the queue and updates the job registry."""
        while True:
            try:
                job_id, status, data = self.status_queue.get()
                if job_id in self.jobs:
                    self.logger.info(f"Received status update for job {job_id}: {status}")
                    self.jobs[job_id]['status'] = status
                    self.jobs[job_id]['updated_at'] = datetime.now().isoformat()
                    if data:
                        self.jobs[job_id]['details'] = data
                    self._persist_job(job_id)
                else:
                    self.logger.warning(f"Received status for unknown job_id: {job_id}")
            except Exception as e:
                self.logger.error(f"Error in status listener: {e}", exc_info=True)

    def _load_jobs_from_registry(self):
        """Load jobs from the registry file on startup."""
        if os.path.exists(self.job_registry_file):
            try:
                with open(self.job_registry_file, 'r') as f:
                    persisted_jobs = json.load(f)
                for job_id, job_info in persisted_jobs.items():
                    if isinstance(job_info, dict):
                        # We don't try to resurrect the process, just load the metadata
                        job_info['process'] = None
                        self.jobs[job_id] = job_info
                    else:
                        self.logger.warning(f"Skipping malformed job entry in registry for job_id '{job_id}'. Expected a dictionary, got {type(job_info)}.")
                self.logger.info(f"Loaded {len(self.jobs)} jobs from registry.")
            except Exception as e:
                self.logger.error(f"Failed to load job registry: {e}", exc_info=True)

    def _persist_job(self, job_id):
        """Persist a single job's state to the registry file."""
        if os.path.exists(self.job_registry_file):
            try:
                with open(self.job_registry_file, 'r') as f:
                    persisted_jobs = json.load(f)
            except (IOError, json.JSONDecodeError):
                persisted_jobs = {}
        else:
            persisted_jobs = {}

        job_to_persist = self.jobs.get(job_id)
        if job_to_persist:
            # Don't persist the process or manager objects
            persisted_jobs[job_id] = {k: v for k, v in job_to_persist.items() if k not in ['process', 'managers']}
        
        with open(self.job_registry_file, 'w') as f:
            json.dump(persisted_jobs, f, indent=4)

    def create_job(self, selections: dict):
        """Creates a new job and adds it to the in-memory registry."""
        job_id = f"job_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        job_folder = os.path.join(OUTPUT_DIR, job_id)
        os.makedirs(job_folder, exist_ok=True)

        # Create job-specific manager instances
        hyper_param_manager = VEstimHyperParamManager()
        training_setup_manager = VEstimTrainingSetupManager(job_id, job_folder, {}) # Initially empty hyperparams

        job_info = {
            "job_id": job_id,
            "status": "created",
            "selections": selections,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "job_folder": job_folder,
            "process": None,
            "details": {}
        }
        self.jobs[job_id] = job_info
        self.job_managers[job_id] = {
            "hyper_param_manager": hyper_param_manager,
            "training_setup_manager": training_setup_manager,
        }
        self._persist_job(job_id)
        self.logger.info(f"Created new job {job_id}")
        return job_id

    def update_job_details(self, job_id: str, details: dict):
        """Updates the details of a specific job."""
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found.")
        
        self.jobs[job_id]['details'].update(details)
        self.jobs[job_id]['updated_at'] = datetime.now().isoformat()
        self._persist_job(job_id)
        self.logger.info(f"Updated details for job {job_id}")

    def start_job(self, job_id: str, target_func, task_info: dict):
        """Starts a job in a new process."""
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found.")
        if self.jobs[job_id]['process'] and self.jobs[job_id]['process'].is_alive():
            self.logger.warning(f"Job {job_id} is already running.")
            return

        self.logger.info(f"Starting job {job_id}")
        
        # Inject job-specific details into the task_info
        task_info['job_folder'] = self.jobs[job_id]['job_folder']
        
        process = Process(target=target_func, args=(self.status_queue, task_info))
        self.jobs[job_id]['process'] = process
        self.jobs[job_id]['status'] = 'running'
        self.jobs[job_id]['updated_at'] = datetime.now().isoformat()
        process.start()
        self._persist_job(job_id)
        return True

    def stop_job(self, job_id: str):
        """Stops a running job."""
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found.")
        
        process = self.jobs[job_id].get('process')
        if process and process.is_alive():
            self.logger.info(f"Stopping job {job_id}")
            process.terminate()
            process.join()
            self.jobs[job_id]['status'] = 'stopped'
            self.jobs[job_id]['updated_at'] = datetime.now().isoformat()
            self._persist_job(job_id)
            return True
        self.logger.warning(f"Job {job_id} is not running.")
        return False

    def get_job(self, job_id: str):
        """
        Gets a job's details from the in-memory registry, ensuring it's serializable.
        """
        job_data = self.jobs.get(job_id)
        if not job_data:
            return None
        
        # Explicitly build the response to ensure no non-serializable objects are included.
        serializable_job = {
            "job_id": job_data.get("job_id"),
            "status": job_data.get("status"),
            "created_at": job_data.get("created_at"),
            "updated_at": job_data.get("updated_at"),
            "selections": job_data.get("selections"),
            "job_folder": job_data.get("job_folder"),
            "details": job_data.get("details", {}),
        }
        return serializable_job

    def get_all_jobs(self):
        """
        Gets all jobs from the in-memory registry, ensuring they are serializable.
        """
        serializable_jobs = []
        for job_id, job_data in self.jobs.items():
            serializable_jobs.append({
                "job_id": job_data.get("job_id"),
                "status": job_data.get("status"),
                "created_at": job_data.get("created_at"),
                "updated_at": job_data.get("updated_at"),
                "selections": job_data.get("selections"),
                "job_folder": job_data.get("job_folder"),
                "details": job_data.get("details", {}),
            })
        return serializable_jobs

    def delete_job(self, job_id: str):
        """Deletes a job from the registry and cleans up its directory."""
        if job_id in self.jobs:
            self.stop_job(job_id)  # Ensure the process is stopped before deleting
            job_folder = self.jobs[job_id].get("job_folder")
            if job_folder and os.path.exists(job_folder):
                shutil.rmtree(job_folder, ignore_errors=True)
                self.logger.info(f"Removed job folder: {job_folder}")
            
            del self.jobs[job_id]
            if job_id in self.job_managers:
                del self.job_managers[job_id]
            
            # Update registry file
            if os.path.exists(self.job_registry_file):
                try:
                    with open(self.job_registry_file, 'r') as f:
                        persisted_jobs = json.load(f)
                    if job_id in persisted_jobs:
                        del persisted_jobs[job_id]
                    with open(self.job_registry_file, 'w') as f:
                        json.dump(persisted_jobs, f, indent=4)
                except (IOError, json.JSONDecodeError) as e:
                    self.logger.error(f"Error updating registry file on delete: {e}")

            self.logger.info(f"Job {job_id} deleted.")
            return True
        return False