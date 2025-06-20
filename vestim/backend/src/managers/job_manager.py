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
from vestim.backend.src.managers.hyper_param_manager import VEstimHyperParamManager
from vestim.backend.src.managers.training_setup_manager import VEstimTrainingSetupManager
from vestim.backend.src.managers.job_container import JobContainer


def _run_job_process(job_id, target_func, status_queue, task_info):
    """Wrapper function to run the job with error handling. Must be a top-level function."""
    logger = logging.getLogger(__name__)
    print(f"[DEBUG] _run_job_process started for job {job_id}")
    try:
        logger.info(f"Starting job process for {job_id}")
        target_func(status_queue, task_info)
        print(f"[DEBUG] target_func completed successfully for job {job_id}")
    except Exception as e:
        print(f"[DEBUG] Exception in _run_job_process for job {job_id}: {e}")
        tb_str = traceback.format_exc()
        print(f"[DEBUG] Exception traceback: {tb_str}")
        logger.error(f"Error in job process {job_id}: {e}", exc_info=True)
        status_queue.put((job_id, 'error', {"error": str(e), "message": "Job failed with error", "traceback": tb_str}))


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
            
            # Resource management - JobManager orchestrates all resources
            self._initialize_resource_management()
            
            # Start a thread to listen for status updates from job processes            self.status_listener_thread = threading.Thread(target=self._listen_for_status_updates, daemon=True)            self.status_listener_thread.start()
            
            # Ensure output directory exists
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            self.job_registry_file = os.path.join(OUTPUT_DIR, 'job_registry.json')
            self._load_jobs_from_registry()
    
    def _initialize_resource_management(self):
        """Initialize resource management based on available hardware"""
        # Detect available hardware
        try:
            import torch
            self.torch_available = True
            self.gpu_available = torch.cuda.is_available()
            self.gpu_count = torch.cuda.device_count() if self.gpu_available else 0
        except ImportError:
            self.torch_available = False
            self.gpu_available = False
            self.gpu_count = 0
        
        # CPU resource management
        import psutil
        cpu_cores = psutil.cpu_count()
        
        # Resource limits based on available hardware
        if self.gpu_available:
            # With GPU: Allow more concurrent jobs since GPU handles heavy lifting
            self.max_concurrent_cpu_jobs = min(cpu_cores, 8)  # Max 8 CPU jobs
            self.max_concurrent_gpu_jobs = self.gpu_count  # One job per GPU
        else:
            # CPU only: Be more conservative to prevent system overload
            self.max_concurrent_cpu_jobs = min(cpu_cores // 2, 4)  # Max 4 CPU jobs
            self.max_concurrent_gpu_jobs = 0
        
        # Job tracking - prioritize older jobs (FIFO)
        self.active_cpu_jobs = []  # List to maintain creation order
        self.active_gpu_jobs = {}  # {gpu_id: job_id}
        self.queued_jobs = []  # FIFO queue for jobs waiting for resources
        self.resource_lock = threading.Lock()
        
        self.logger.info(f"Resource management initialized:")
        self.logger.info(f"  CPU cores: {cpu_cores}")
        self.logger.info(f"  GPU available: {self.gpu_available}")
        self.logger.info(f"  GPU count: {self.gpu_count}")
        self.logger.info(f"  Max concurrent CPU jobs: {self.max_concurrent_cpu_jobs}")
        self.logger.info(f"  Max concurrent GPU jobs: {self.max_concurrent_gpu_jobs}")
    
    def request_training_resources(self, job_id: str, device_preference: str = "cpu"):
        """
        Request training resources for a job based on device preference from hyperparameters.
        Returns: (can_start: bool, allocated_device: str, queue_position: int)
        """
        with self.resource_lock:
            # Parse device preference (e.g., "cuda:0", "cuda:1", "cpu")
            wants_gpu = device_preference.startswith("cuda") if device_preference else False
            
            if wants_gpu and self.gpu_available:
                # Try to allocate GPU
                requested_gpu_id = int(device_preference.split(":")[1]) if ":" in device_preference else 0
                
                # Check if requested GPU is available
                if requested_gpu_id < self.gpu_count:
                    if requested_gpu_id not in self.active_gpu_jobs:
                        # GPU is available
                        self.active_gpu_jobs[requested_gpu_id] = job_id
                        self.logger.info(f"Allocated GPU {requested_gpu_id} to job {job_id}")
                        return True, f"cuda:{requested_gpu_id}", 0
                    else:
                        # Requested GPU is busy, try other GPUs
                        for gpu_id in range(self.gpu_count):
                            if gpu_id not in self.active_gpu_jobs:
                                self.active_gpu_jobs[gpu_id] = job_id
                                self.logger.info(f"Allocated GPU {gpu_id} to job {job_id} (requested {requested_gpu_id} was busy)")
                                return True, f"cuda:{gpu_id}", 0
                
                # All GPUs busy, queue the job
                if job_id not in self.queued_jobs:
                    self.queued_jobs.append(job_id)
                queue_pos = self.queued_jobs.index(job_id) + 1
                self.logger.info(f"All GPUs busy, queued job {job_id} for GPU (position: {queue_pos})")
                return False, "cpu", queue_pos  # Fallback to CPU while waiting
            
            # CPU training (either requested or fallback from GPU)
            if len(self.active_cpu_jobs) < self.max_concurrent_cpu_jobs:
                self.active_cpu_jobs.append(job_id)
                device = device_preference if not wants_gpu else "cpu"
                self.logger.info(f"Allocated CPU to job {job_id} (active CPU jobs: {len(self.active_cpu_jobs)})")
                return True, device, 0
            else:
                # CPU slots full, queue the job
                if job_id not in self.queued_jobs:
                    self.queued_jobs.append(job_id)
                queue_pos = self.queued_jobs.index(job_id) + 1
                self.logger.info(f"All CPU slots busy, queued job {job_id} (position: {queue_pos})")
                return False, "cpu", queue_pos
    
    def release_training_resources(self, job_id: str):
        """Release training resources and start next queued job"""
        with self.resource_lock:
            released_resource = None
            
            # Check if job was using GPU
            for gpu_id, active_job_id in list(self.active_gpu_jobs.items()):
                if active_job_id == job_id:
                    del self.active_gpu_jobs[gpu_id]
                    released_resource = f"cuda:{gpu_id}"
                    self.logger.info(f"Released GPU {gpu_id} from job {job_id}")
                    break
            
            # Check if job was using CPU
            if job_id in self.active_cpu_jobs:
                self.active_cpu_jobs.remove(job_id)
                released_resource = "cpu"
                self.logger.info(f"Released CPU from job {job_id} (active CPU jobs: {len(self.active_cpu_jobs)})")
            
            # Start next queued job (FIFO - oldest jobs get priority)
            if self.queued_jobs and released_resource:
                next_job_id = self.queued_jobs.pop(0)  # FIFO: take oldest queued job
                self.logger.info(f"Starting queued job {next_job_id} on {released_resource}")
                
                # Notify the job container that resources are available
                if next_job_id in self.job_containers:
                    job_container = self.job_containers[next_job_id]
                    # Allocate the resource immediately
                    if released_resource.startswith("cuda"):
                        gpu_id = int(released_resource.split(":")[1])
                        self.active_gpu_jobs[gpu_id] = next_job_id
                    else:
                        self.active_cpu_jobs.append(next_job_id)
                    
                    # TODO: Notify job container to start training
                    # job_container.on_training_resources_available(released_resource)
    
    def get_resource_status(self):
        """Get current resource usage for dashboard display"""
        with self.resource_lock:
            return {
                "gpu_available": self.gpu_available,
                "gpu_count": self.gpu_count, 
                "active_gpu_jobs": dict(self.active_gpu_jobs),
                "active_cpu_jobs": self.active_cpu_jobs.copy(),
                "queued_jobs": self.queued_jobs.copy(),
                "max_concurrent_cpu": self.max_concurrent_cpu_jobs,
                "max_concurrent_gpu": self.max_concurrent_gpu_jobs,
                "current_cpu_usage": len(self.active_cpu_jobs),
                "current_gpu_usage": len(self.active_gpu_jobs)
            }
            self.logger.info("PyTorch not available, using CPU only")
            
    def allocate_training_resources(self, job_id: str):
        """
        Allocate training resources (GPU/CPU) for a job.
        Returns resource allocation info or None if resources not available.
        """
        with self.resource_lock:
            # Check if we can start another training job
            if len(self.active_training_jobs) >= self.max_concurrent_training_jobs:
                self.training_queue.append(job_id)
                self.logger.info(f"Job {job_id} queued for training - resource limit reached")
                return None
            
            # Allocate GPU if available
            allocated_device = "cpu"  # Default fallback
            
            with self.gpu_lock:
                # Find available GPU
                for gpu_id in self.available_gpus:
                    if gpu_id not in self.gpu_allocation.values():
                        self.gpu_allocation[job_id] = gpu_id
                        allocated_device = f"cuda:{gpu_id}"
                        break
            
            # Mark job as active
            self.active_training_jobs.add(job_id)
            
            resource_info = {
                "device": allocated_device,
                "max_concurrent_tasks": 1 if allocated_device.startswith("cuda") else 2,
                "allocated_at": datetime.now().isoformat()
            }
            
            self.logger.info(f"Allocated resources for job {job_id}: {resource_info}")
            return resource_info
    
    def release_training_resources(self, job_id: str):
        """Release training resources when job completes"""
        with self.resource_lock:
            # Remove from active jobs
            self.active_training_jobs.discard(job_id)
            
            # Release GPU allocation
            with self.gpu_lock:
                if job_id in self.gpu_allocation:
                    released_gpu = self.gpu_allocation.pop(job_id)
                    self.logger.info(f"Released GPU {released_gpu} from job {job_id}")
            
            # Start next job in queue if any
            if self.training_queue:
                next_job_id = self.training_queue.pop(0)
                self.logger.info(f"Starting queued job {next_job_id}")
                
                # Get the next job container and trigger training
                if next_job_id in self.job_containers:
                    next_job_container = self.job_containers[next_job_id]
                    # Notify the job container that resources are now available
                    next_job_container.on_training_resources_available()
                    
    def get_resource_status(self):
        """Get current resource allocation status for monitoring"""
        with self.resource_lock:
            return {                "active_training_jobs": len(self.active_training_jobs),
                "max_concurrent_jobs": self.max_concurrent_training_jobs,
                "queued_jobs": len(self.training_queue),
                "gpu_allocations": dict(self.gpu_allocation),
                "available_gpus": self.available_gpus
            }
    
    def _listen_for_status_updates(self):
        """Listens for status updates from the queue and updates job containers with detailed phase progress."""
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
                        
                        # Enhanced: Handle detailed phase progress based on status
                        if data and status == 'training_progress':
                            # Training phase detailed progress
                            job_container.update_phase_progress(
                                phase="training",
                                status="in_progress",
                                progress_percent=data.get('progress_percent', 0),
                                message=message,
                                current_epoch=data.get('current_epoch', 0),
                                total_epochs=data.get('total_epochs', 0),
                                training_loss=data.get('train_loss', 0.0),
                                validation_loss=data.get('val_loss', 0.0),
                                epoch_time=data.get('epoch_time', 0.0)
                            )
                            
                            # Store training history in job details
                            if 'training_history' in data:
                                job_container.details['training_history'] = data['training_history']
                                
                        elif data and status == 'training_completed':
                            # Training completion
                            job_container.update_phase_progress(
                                phase="training",
                                status="completed",
                                progress_percent=100,
                                message="Training completed successfully"
                            )
                            if 'training_history' in data:
                                job_container.details['training_history'] = data['training_history']
                                
                        elif data and status == 'training_error':
                            # Training error
                            job_container.update_phase_progress(
                                phase="training",
                                status="error",
                                progress_percent=0,
                                message=f"Training failed: {data.get('message', 'Unknown error')}"
                            )
                            
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
        print(f"[DEBUG] JobManager.start_job called for job {job_id}")
        print(f"[DEBUG] target_func: {target_func}")
        print(f"[DEBUG] task_info type: {type(task_info)}")
        
        try:
            print(f"[DEBUG] Checking if job {job_id} exists in job_containers")
            if job_id not in self.job_containers:
                print(f"[DEBUG] ERROR: Job {job_id} not found in job_containers")
                print(f"[DEBUG] Available job_containers: {list(self.job_containers.keys())}")
                raise ValueError(f"Job {job_id} not found.")
            print(f"[DEBUG] Job {job_id} found in job_containers")
                
            print(f"[DEBUG] Acquiring lock for job {job_id}")
            with self._get_job_lock(job_id):
                print(f"[DEBUG] Lock acquired for job {job_id}")
                job_container = self.job_containers[job_id]
                print(f"[DEBUG] Job container retrieved: {type(job_container)}")
                
                print(f"[DEBUG] Checking if job {job_id} is already running")
                if job_container.is_running():
                    print(f"[DEBUG] WARNING: Job {job_id} is already running")
                    self.logger.warning(f"Job {job_id} is already running.")
                    return False
                print(f"[DEBUG] Job {job_id} is not currently running")

                self.logger.info(f"Starting job {job_id}")
                print(f"[DEBUG] Starting job {job_id}")
                
                # Reset the stop flag
                print(f"[DEBUG] Resetting stop flag for job {job_id}")
                job_container.stop_flag.clear()
                
                # Inject job-specific, serializable details into the task_info
                print(f"[DEBUG] Injecting job-specific details into task_info")
                task_info['job_id'] = job_id
                task_info['job_folder'] = job_container.job_folder
                task_info['stop_flag'] = job_container.stop_flag
                task_info['training_tasks'] = job_container.get_training_tasks() # Pass tasks directly
                
                # Ensure job_container itself is not in task_info
                task_info.pop('job_container', None)
                
                print(f"[DEBUG] task_info now contains: {list(task_info.keys())}")
                
                # Start the process
                print(f"[DEBUG] Creating process for job {job_id}")
                print(f"[DEBUG] Process target: {_run_job_process}")
                print(f"[DEBUG] Process args: job_id={job_id}, target_func={target_func}")
                
                # The target function must not be a bound method.
                # We get the manager and extract the function to be called in the process.
                training_manager = job_container.get_training_task_manager()
                
                process = Process(target=_run_job_process,
                                 args=(job_id, training_manager.process_all_tasks_in_background, self.status_queue, task_info))
                print(f"[DEBUG] Process created: {process}")
                
                print(f"[DEBUG] Setting process in job container")
                job_container.set_process(process)
                
                print(f"[DEBUG] Updating job container status to 'running'")
                job_container.update_status('running', 'Job started')
                
                print(f"[DEBUG] Starting process for job {job_id}")
                process.start()
                print(f"[DEBUG] Process started for job {job_id}")
                
                print(f"[DEBUG] Persisting job container for job {job_id}")
                self._persist_job_container(job_id)
                print(f"[DEBUG] Job container persisted for job {job_id}")
                
            print(f"[DEBUG] Job {job_id} successfully started")
            return True
        except Exception as e:
            print(f"[DEBUG] Exception in JobManager.start_job for job {job_id}: {e}")
            import traceback
            print(f"[DEBUG] Exception traceback: {traceback.format_exc()}")
            self.logger.error(f"Error starting job {job_id}: {e}", exc_info=True)
            # Update job container status to indicate error
            if job_id in self.job_containers:
                job_container = self.job_containers[job_id]
                job_container.update_status('error', f'Failed to start: {str(e)}')
                self._persist_job_container(job_id)
            return False

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