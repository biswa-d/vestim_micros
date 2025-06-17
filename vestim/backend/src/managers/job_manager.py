import os
import json
import time
import uuid
from datetime import datetime
from multiprocessing import Process, Queue, Event
import threading
from vestim.config import OUTPUT_DIR
from vestim.logger_config import configure_job_specific_logging
import logging
import shutil
import traceback
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
            self.job_locks = {}  # Locks for thread-safe job operations
            self.status_queue = Queue()
            self.logger = logging.getLogger(__name__)
            
            # Start a thread to listen for status updates from job processes
            self.status_listener_thread = threading.Thread(target=self._listen_for_status_updates, daemon=True)
            self.status_listener_thread.start()

            # Ensure output directory exists
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            self.job_registry_file = os.path.join(OUTPUT_DIR, 'job_registry.json')
            self._load_jobs_from_registry()
            
            # Signal flags for all jobs
            self.stop_flags = {}

    def _listen_for_status_updates(self):
        """Listens for status updates from the queue and updates the job registry."""
        while True:
            try:
                job_id, status, data = self.status_queue.get()
                if job_id in self.jobs:
                    self.logger.info(f"Received status update for job {job_id}: {status}")
                    with self._get_job_lock(job_id):
                        self.jobs[job_id]['status'] = status
                        self.jobs[job_id]['updated_at'] = datetime.now().isoformat()
                        if data:
                            # Merge details rather than replace to preserve any fields
                            if 'details' not in self.jobs[job_id]:
                                self.jobs[job_id]['details'] = {}
                            self.jobs[job_id]['details'].update(data)
                        self._persist_job(job_id)
                else:
                    self.logger.warning(f"Received status for unknown job_id: {job_id}")
            except Exception as e:
                self.logger.error(f"Error in status listener: {e}", exc_info=True)
                # Don't let an exception break the listener loop
                time.sleep(1)  # Prevent tight loop if persistent error

    def _get_job_lock(self, job_id):
        """Get a thread lock for the specific job."""
        if job_id not in self.job_locks:
            self.job_locks[job_id] = threading.Lock()
        return self.job_locks[job_id]

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
                        # Initialize a stop flag for each job
                        self.stop_flags[job_id] = Event()
                    else:
                        self.logger.warning(f"Skipping malformed job entry in registry for job_id '{job_id}'. Expected a dictionary, got {type(job_info)}.")
                self.logger.info(f"Loaded {len(self.jobs)} jobs from registry.")
            except Exception as e:
                self.logger.error(f"Failed to load job registry: {e}", exc_info=True)
                # Create an empty registry if loading fails
                self.jobs = {}

    def _persist_job(self, job_id):
        """Persist a single job's state to the registry file."""
        try:
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
        except Exception as e:
            self.logger.error(f"Error persisting job {job_id}: {e}", exc_info=True)

    def create_job(self, selections: dict):
        """Creates a new job and adds it to the in-memory registry."""
        try:
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
            
            with self._get_job_lock(job_id):
                self.jobs[job_id] = job_info
                self.job_managers[job_id] = {
                    "hyper_param_manager": hyper_param_manager,
                    "training_setup_manager": training_setup_manager,
                }
                # Create a stop flag for the job
                self.stop_flags[job_id] = Event()
                self._persist_job(job_id)
            
            self.logger.info(f"Created new job {job_id}")
            return job_id
        except Exception as e:
            self.logger.error(f"Error creating job: {e}", exc_info=True)
            # Clean up any partially created job resources
            if 'job_folder' in locals() and os.path.exists(job_folder):
                try:
                    shutil.rmtree(job_folder)
                except Exception as cleanup_error:
                    self.logger.error(f"Error cleaning up job folder after creation failure: {cleanup_error}")
            raise

    def update_job_details(self, job_id: str, details: dict):
        """Updates the details of a specific job."""
        try:
            if job_id not in self.jobs:
                raise ValueError(f"Job {job_id} not found.")
            
            with self._get_job_lock(job_id):
                if 'details' not in self.jobs[job_id]:
                    self.jobs[job_id]['details'] = {}
                self.jobs[job_id]['details'].update(details)
                self.jobs[job_id]['updated_at'] = datetime.now().isoformat()
                self._persist_job(job_id)
            
            self.logger.info(f"Updated details for job {job_id}")
        except Exception as e:
            self.logger.error(f"Error updating job details for {job_id}: {e}", exc_info=True)
            raise

    def start_job(self, job_id: str, target_func, task_info: dict):
        """Starts a job in a new process."""
        try:
            if job_id not in self.jobs:
                raise ValueError(f"Job {job_id} not found.")
                
            with self._get_job_lock(job_id):
                if self.jobs[job_id].get('process') and self.jobs[job_id]['process'].is_alive():
                    self.logger.warning(f"Job {job_id} is already running.")
                    return False

                self.logger.info(f"Starting job {job_id}")
                
                # Reset the stop flag
                if job_id in self.stop_flags:
                    self.stop_flags[job_id].clear()
                else:
                    self.stop_flags[job_id] = Event()
                
                # Inject job-specific details into the task_info
                task_info['job_folder'] = self.jobs[job_id]['job_folder']
                task_info['stop_flag'] = self.stop_flags[job_id]
                
                # Start the process
                process = Process(target=self._run_job_with_error_handling, 
                                 args=(job_id, target_func, self.status_queue, task_info))
                self.jobs[job_id]['process'] = process
                self.jobs[job_id]['status'] = 'running'
                self.jobs[job_id]['updated_at'] = datetime.now().isoformat()
                process.start()
                self._persist_job(job_id)
                
            return True
        except Exception as e:
            self.logger.error(f"Error starting job {job_id}: {e}", exc_info=True)
            # Update job status to indicate error
            with self._get_job_lock(job_id):
                self.jobs[job_id]['status'] = 'error'
                self.jobs[job_id]['updated_at'] = datetime.now().isoformat()
                self.jobs[job_id]['details']['error'] = str(e)
                self._persist_job(job_id)
            return False

    def _run_job_with_error_handling(self, job_id, target_func, status_queue, task_info):
        """Wrapper function to run the job with error handling."""
        try:
            self.logger.info(f"Starting job process for {job_id}")
            target_func(status_queue, task_info)
        except Exception as e:
            self.logger.error(f"Error in job process {job_id}: {e}", exc_info=True)
            # Send error status back to main process
            status_queue.put((job_id, 'error', {"error": str(e), "message": "Job failed with error"}))

    def stop_job(self, job_id: str):
        """Stops a running job."""
        try:
            if job_id not in self.jobs:
                raise ValueError(f"Job {job_id} not found.")
            
            # Set the stop flag to signal the process to exit gracefully
            if job_id in self.stop_flags:
                self.stop_flags[job_id].set()
                self.logger.info(f"Set stop flag for job {job_id}")
            
            with self._get_job_lock(job_id):
                process = self.jobs[job_id].get('process')
                if process and process.is_alive():
                    self.logger.info(f"Stopping job {job_id}")
                    # Give the process some time to shut down gracefully
                    time.sleep(2)
                    # If still alive, terminate it
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=5)
                        if process.is_alive():
                            # If still alive, kill it forcefully
                            self.logger.warning(f"Job {job_id} did not terminate gracefully, forcefully killing")
                            process.kill()
                    
                    self.jobs[job_id]['status'] = 'stopped'
                    self.jobs[job_id]['updated_at'] = datetime.now().isoformat()
                    self._persist_job(job_id)
                    return True
                else:
                    self.logger.warning(f"Job {job_id} is not running.")
                    return False
        except Exception as e:
            self.logger.error(f"Error stopping job {job_id}: {e}", exc_info=True)
            return False

    def get_job(self, job_id: str):
        """
        Gets a job's details from the in-memory registry, ensuring it's serializable.
        """
        try:
            with self._get_job_lock(job_id):
                job_data = self.jobs.get(job_id)
                if not job_data:
                    return None
                
                # Explicitly build the response to ensure no non-serializable objects are included.
                # Only include basic data types (str, int, float, bool, dict, list)
                def make_serializable(obj):
                    """Recursively ensure an object is JSON serializable"""
                    if obj is None or isinstance(obj, (str, int, float, bool)):
                        return obj
                    elif isinstance(obj, dict):
                        return {k: make_serializable(v) for k, v in obj.items() 
                               if isinstance(k, str) and not k.startswith('_')}
                    elif isinstance(obj, (list, tuple)):
                        return [make_serializable(item) for item in obj]
                    else:
                        # For any other type, convert to string or skip
                        try:
                            return str(obj) if obj is not None else None
                        except:
                            return None
                
                serializable_job = {
                    "job_id": str(job_data.get("job_id", "")),
                    "status": str(job_data.get("status", "")),
                    "created_at": str(job_data.get("created_at", "")),
                    "updated_at": str(job_data.get("updated_at", "")),
                    "selections": make_serializable(job_data.get("selections", {})),
                    "job_folder": str(job_data.get("job_folder", "")),
                    "details": make_serializable(job_data.get("details", {})),
                }
                return serializable_job
        except Exception as e:
            self.logger.error(f"Error getting job {job_id}: {e}", exc_info=True)
            return None

    def get_all_jobs(self):
        """
        Gets all jobs from the in-memory registry, ensuring they are serializable.
        """
        try:
            serializable_jobs = {}
            job_ids = list(self.jobs.keys())
            
            for job_id in job_ids:
                job_data = self.get_job(job_id)
                if job_data:
                    serializable_jobs[job_id] = job_data
            
            return list(serializable_jobs.values())
        except Exception as e:
            self.logger.error(f"Error getting all jobs: {e}", exc_info=True)
            return []

    def delete_job(self, job_id: str):
        """Deletes a job from the registry and cleans up its directory."""
        try:
            if job_id in self.jobs:
                # Stop the job if it's running
                self.stop_job(job_id)
                
                # Get the job folder path before deleting the job info
                with self._get_job_lock(job_id):
                    job_folder = self.jobs[job_id].get("job_folder")
                    
                    # Clean up resources
                    if job_folder and os.path.exists(job_folder):
                        shutil.rmtree(job_folder, ignore_errors=True)
                        self.logger.info(f"Removed job folder: {job_folder}")
                    
                    # Remove job from dictionaries
                    del self.jobs[job_id]
                    if job_id in self.job_managers:
                        del self.job_managers[job_id]
                    if job_id in self.stop_flags:
                        del self.stop_flags[job_id]
                    if job_id in self.job_locks:
                        del self.job_locks[job_id]
                
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
        except Exception as e:
            self.logger.error(f"Error deleting job {job_id}: {e}", exc_info=True)
            return False

    def ensure_job_exists(self, job_id: str, job_folder: str = None):
        """
        Ensures that a job exists in the manager. If it doesn't exist, creates it.
        
        Args:
            job_id (str): The ID of the job to ensure exists
            job_folder (str, optional): The job folder path. If not provided, it will be constructed from the job_id
            
        Returns:
            bool: True if the job exists or was created successfully, False otherwise
        """
        try:
            if job_id in self.jobs:
                self.logger.info(f"Job {job_id} already exists in JobManager")
                return True
                
            self.logger.info(f"Job {job_id} not found in JobManager, attempting to create registry entry")
            
            if not job_folder:
                job_folder = os.path.join(OUTPUT_DIR, job_id)
                
            if not os.path.exists(job_folder):
                self.logger.warning(f"Job folder {job_folder} does not exist for {job_id}")
                os.makedirs(job_folder, exist_ok=True)
                self.logger.info(f"Created job folder {job_folder} for {job_id}")
            
            # Create job-specific manager instances
            hyper_param_manager = VEstimHyperParamManager()
            training_setup_manager = VEstimTrainingSetupManager(job_id, job_folder, {})  # Empty hyperparams initially
            
            job_info = {
                "job_id": job_id,
                "status": "imported",  # Using a special status to indicate it was imported
                "selections": {},  # Empty selections since we don't know the original
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "job_folder": job_folder,
                "process": None,
                "details": {"imported": True}
            }
            
            with self._get_job_lock(job_id):
                self.jobs[job_id] = job_info
                self.job_managers[job_id] = {
                    "hyper_param_manager": hyper_param_manager,
                    "training_setup_manager": training_setup_manager,
                }
                # Create a stop flag for the job
                self.stop_flags[job_id] = Event()
                self._persist_job(job_id)
            
            self.logger.info(f"Successfully created registry entry for existing job {job_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error ensuring job {job_id} exists: {e}", exc_info=True)
            return False