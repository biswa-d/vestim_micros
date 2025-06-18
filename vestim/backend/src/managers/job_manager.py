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
from vestim.backend.src.managers.job_container import JobContainer

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
            self.job_containers = {}  # Job containers registry (main data structure)
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

    def _listen_for_status_updates(self):
        """Listens for status updates from the queue and updates job containers."""
        while True:
            try:
                job_id, status, data = self.status_queue.get()
                if job_id in self.job_containers:
                    self.logger.info(f"Received status update for job {job_id}: {status}")
                    with self._get_job_lock(job_id):
                        job_container = self.job_containers[job_id]
                        
                        # Update job container status
                        progress_percent = data.get('progress_percent') if data else None
                        message = data.get('message', f'Status: {status}') if data else f'Status: {status}'
                        job_container.update_status(status, message, progress_percent)
                        
                        # Update task-specific progress if provided
                        if data and 'task_id' in data:
                            task_id = data['task_id']
                            task_data = {k: v for k, v in data.items() if k != 'task_id'}
                            job_container.update_task_progress(task_id, task_data)
                        
                        self._persist_job_container(job_id)
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
        """Load jobs from the registry file on startup and create job containers."""
        if os.path.exists(self.job_registry_file):
            try:
                with open(self.job_registry_file, 'r') as f:
                    persisted_jobs = json.load(f)
                for job_id, job_info in persisted_jobs.items():
                    if isinstance(job_info, dict):
                        # Create job container from persisted data
                        job_folder = job_info.get('job_folder', os.path.join(OUTPUT_DIR, job_id))
                        selections = job_info.get('selections', {})
                        
                        job_container = JobContainer(job_id, job_folder, selections)
                        job_container.status = job_info.get('status', 'loaded')
                        job_container.created_at = job_info.get('created_at', datetime.now().isoformat())
                        job_container.updated_at = job_info.get('updated_at', datetime.now().isoformat())
                        job_container.details = job_info.get('details', {})
                        job_container.task_progress = job_info.get('task_progress', {})
                        
                        # Reset process state - processes can't be persisted
                        job_container.process = None
                        job_container.stop_flag.clear()
                        
                        self.job_containers[job_id] = job_container
                    else:
                        self.logger.warning(f"Skipping malformed job entry in registry for job_id '{job_id}'. Expected a dictionary, got {type(job_info)}.")
                self.logger.info(f"Loaded {len(self.job_containers)} job containers from registry.")
            except Exception as e:
                self.logger.error(f"Failed to load job registry: {e}", exc_info=True)
                # Create an empty registry if loading fails
                self.job_containers = {}

    def _persist_job_container(self, job_id):
        """Persist a job container's state to the registry file."""
        try:
            if os.path.exists(self.job_registry_file):
                try:
                    with open(self.job_registry_file, 'r') as f:
                        persisted_jobs = json.load(f)
                except (IOError, json.JSONDecodeError):
                    persisted_jobs = {}
            else:
                persisted_jobs = {}

            job_container = self.job_containers.get(job_id)
            if job_container:
                # Get serializable state from job container
                job_state = job_container.get_detailed_state()
                # Remove non-serializable fields
                job_state.pop('is_running', None)  # This is computed
                job_state.pop('active_managers', None)  # This is computed
                persisted_jobs[job_id] = job_state
            
            with open(self.job_registry_file, 'w') as f:
                json.dump(persisted_jobs, f, indent=4)
        except Exception as e:
            self.logger.error(f"Error persisting job container {job_id}: {e}", exc_info=True)

    def create_job(self, selections: dict):
        """Creates a new job and adds it to the in-memory registry using JobContainer."""
        try:
            print(f"JobManager: Creating job with selections: {selections}")
            job_id = f"job_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            job_folder = os.path.join(OUTPUT_DIR, job_id)
            print(f"JobManager: Generated job_id: {job_id}")
            print(f"JobManager: Job folder will be: {job_folder}")
            
            os.makedirs(job_folder, exist_ok=True)
            print(f"JobManager: Created job folder: {job_folder}")

            # Create JobContainer which will manage all job-specific resources
            print(f"JobManager: Creating JobContainer...")
            job_container = JobContainer(job_id, job_folder, selections)
            print(f"JobManager: JobContainer created successfully")
            
            with self._get_job_lock(job_id):
                self.job_containers[job_id] = job_container
                print(f"JobManager: Added job container to registry")
                self._persist_job_container(job_id)
                print(f"JobManager: Persisted job container")
            
            self.logger.info(f"Created new job container {job_id}")
            print(f"JobManager: Successfully created job {job_id}")
            return job_id
        except Exception as e:
            self.logger.error(f"Error creating job: {e}", exc_info=True)
            print(f"JobManager ERROR: Exception in create_job: {e}")
            import traceback
            traceback.print_exc()
            # Clean up any partially created job resources
            if 'job_folder' in locals() and os.path.exists(job_folder):
                try:
                    shutil.rmtree(job_folder)
                    print(f"JobManager: Cleaned up job folder after error: {job_folder}")
                except Exception as cleanup_error:
                    self.logger.error(f"Error cleaning up job folder after creation failure: {cleanup_error}")
                    print(f"JobManager: Error cleaning up job folder: {cleanup_error}")
            raise

    def update_job_details(self, job_id: str, details: dict):
        """Updates the details of a specific job container."""
        try:
            if job_id not in self.job_containers:
                raise ValueError(f"Job {job_id} not found.")
            
            with self._get_job_lock(job_id):
                job_container = self.job_containers[job_id]
                job_container.details.update(details)
                job_container.updated_at = datetime.now().isoformat()
                self._persist_job_container(job_id)
            
            self.logger.info(f"Updated details for job {job_id}")
        except Exception as e:
            self.logger.error(f"Error updating job details for {job_id}: {e}", exc_info=True)
            raise

    def start_job(self, job_id: str, target_func, task_info: dict):
        """Starts a job in a new process using job container."""
        try:
            if job_id not in self.job_containers:
                raise ValueError(f"Job {job_id} not found.")
                
            with self._get_job_lock(job_id):
                job_container = self.job_containers[job_id]
                
                if job_container.is_running():
                    self.logger.warning(f"Job {job_id} is already running.")
                    return False

                self.logger.info(f"Starting job {job_id}")
                
                # Reset the stop flag
                job_container.stop_flag.clear()
                
                # Inject job-specific details into the task_info
                task_info['job_folder'] = job_container.job_folder
                task_info['stop_flag'] = job_container.stop_flag
                
                # Start the process
                process = Process(target=self._run_job_with_error_handling, 
                                 args=(job_id, target_func, self.status_queue, task_info))
                job_container.set_process(process)
                job_container.update_status('running', 'Job started')
                process.start()
                self._persist_job_container(job_id)
                
            return True
        except Exception as e:
            self.logger.error(f"Error starting job {job_id}: {e}", exc_info=True)
            # Update job container status to indicate error
            if job_id in self.job_containers:
                job_container = self.job_containers[job_id]
                job_container.update_status('error', f'Failed to start: {str(e)}')
                self._persist_job_container(job_id)
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
        """Stops a running job using job container."""
        try:
            if job_id not in self.job_containers:
                raise ValueError(f"Job {job_id} not found.")
            
            job_container = self.job_containers[job_id]
            
            with self._get_job_lock(job_id):
                if job_container.is_running():
                    self.logger.info(f"Stopping job {job_id}")
                    job_container.stop()
                    job_container.update_status('stopped', 'Job stopped by user')
                    self._persist_job_container(job_id)
                    return True
                else:
                    self.logger.warning(f"Job {job_id} is not running.")
                    return False
        except Exception as e:
            self.logger.error(f"Error stopping job {job_id}: {e}", exc_info=True)
            return False

    def get_job(self, job_id: str):
        """
        Gets a job's details from job container, ensuring it's serializable.
        """
        try:
            with self._get_job_lock(job_id):
                job_container = self.job_containers.get(job_id)
                if not job_container:
                    return None
                
                # Get serializable state from job container
                return job_container.get_detailed_state()
        except Exception as e:
            self.logger.error(f"Error getting job {job_id}: {e}", exc_info=True)
            return None

    def get_all_jobs(self):
        """
        Gets all jobs from job containers, ensuring they are serializable.
        """
        try:
            jobs_list = []
            job_ids = list(self.job_containers.keys())
            
            for job_id in job_ids:
                job_data = self.get_job(job_id)
                if job_data:
                    jobs_list.append(job_data)
            
            return jobs_list
        except Exception as e:
            self.logger.error(f"Error getting all jobs: {e}", exc_info=True)
            return []

    def delete_job(self, job_id: str):
        """Deletes a job container and cleans up its directory."""
        try:
            if job_id in self.job_containers:
                job_container = self.job_containers[job_id]
                
                # Stop the job if it's running
                self.stop_job(job_id)
                
                with self._get_job_lock(job_id):
                    job_folder = job_container.job_folder
                    
                    # Clean up managers
                    job_container.cleanup_managers()
                    
                    # Clean up resources
                    if job_folder and os.path.exists(job_folder):
                        shutil.rmtree(job_folder, ignore_errors=True)
                        self.logger.info(f"Removed job folder: {job_folder}")
                    
                    # Remove job container
                    del self.job_containers[job_id]
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

    def get_job_container(self, job_id: str):
        """Get the job container for a specific job."""
        return self.job_containers.get(job_id)
    
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
            if job_id in self.job_containers:
                self.logger.info(f"Job container {job_id} already exists")
                return True
                
            self.logger.info(f"Job container {job_id} not found, creating new one")
            
            if not job_folder:
                job_folder = os.path.join(OUTPUT_DIR, job_id)
                
            if not os.path.exists(job_folder):
                self.logger.warning(f"Job folder {job_folder} does not exist for {job_id}")
                os.makedirs(job_folder, exist_ok=True)
                self.logger.info(f"Created job folder {job_folder} for {job_id}")
            
            # Create new job container
            job_container = JobContainer(job_id, job_folder, {})
            job_container.status = "imported"  # Special status for imported jobs
            job_container.details["imported"] = True
            
            with self._get_job_lock(job_id):
                self.job_containers[job_id] = job_container
                self._persist_job_container(job_id)
            
            self.logger.info(f"Successfully created job container for existing job {job_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error ensuring job {job_id} exists: {e}", exc_info=True)
            return False

    def get_training_task_manager(self, job_id: str):
        """Get the job-specific training task manager instance."""
        try:
            if job_id not in self.job_containers:
                raise ValueError(f"Job {job_id} not found.")
            
            job_container = self.job_containers[job_id]
            return job_container.get_training_task_manager()
            
        except Exception as e:
            self.logger.error(f"Error getting training task manager for job {job_id}: {e}", exc_info=True)
            raise

    def get_job_managers(self, job_id: str):
        """Get all managers for a specific job."""
        if job_id not in self.job_containers:
            raise ValueError(f"No job container found for job {job_id}.")
        
        job_container = self.job_containers[job_id]
        return job_container.managers