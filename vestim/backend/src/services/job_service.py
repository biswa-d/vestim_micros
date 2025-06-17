import os
import itertools
import logging
from vestim.backend.src.managers.job_manager import JobManager

class JobService:
    """
    A service layer that acts as a client to the JobManager.
    This service is responsible for handling job-related operations by delegating them
    to the central JobManager, which manages job state and execution.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.job_manager = JobManager()
        self.current_job_id = None
    def create_new_job(self, selections: dict):
        """
        Creates a new job via the JobManager.
        """
        self.logger.info(f"Creating new job with selections: {selections}")
        try:
            job_id = self.job_manager.create_job(selections)
            job_info = self.job_manager.get_job(job_id)
            return job_id, job_info.get("job_folder")
        except Exception as e:
            self.logger.error(f"Failed to create new job: {e}", exc_info=True)
            return None, None

    def start_job(self, job_id: str, target_func, task_info: dict):
        """
        Starts a job process via the JobManager.
        """
        self.logger.info(f"Attempting to start job: {job_id}")
        try:
            # Update job status to indicate training has started
            self.update_job_status(job_id, "training")
            self.logger.info(f"Updated job {job_id} status to 'training'")
            
            return self.job_manager.start_job(job_id, target_func, task_info)
        except Exception as e:
            self.logger.error(f"Failed to start job {job_id}: {e}", exc_info=True)
            return False

    def stop_job(self, job_id: str):
        """
        Stops a job process via the JobManager.
        """
        self.logger.info(f"Attempting to stop job: {job_id}")
        try:
            return self.job_manager.stop_job(job_id)
        except Exception as e:
            self.logger.error(f"Failed to stop job {job_id}: {e}", exc_info=True)
            return False

    def get_job_by_id(self, job_id: str):
        """
        Retrieves job details from the JobManager.
        """
        return self.job_manager.get_job(job_id)

    def get_all_jobs(self):
        """
        Retrieves all jobs from the JobManager.
        """
        return self.job_manager.get_all_jobs()

    def delete_job(self, job_id: str):
        """
        Deletes a job via the JobManager.
        """
        self.logger.info(f"Attempting to delete job: {job_id}")
        try:
            return self.job_manager.delete_job(job_id)
        except Exception as e:
            self.logger.error(f"Failed to delete job {job_id}: {e}", exc_info=True)
            return False

    def save_hyperparameters(self, job_id: str, params: dict):
        """Saves hyperparameters for a job via the JobManager."""
        self.logger.info(f"Attempting to save hyperparameters for job: {job_id}")
        try:
            job_managers = self.job_manager.job_managers.get(job_id)
            if not job_managers:
                raise ValueError(f"Managers for job with ID {job_id} not found.")

            job_info = self.job_manager.get_job(job_id)
            if not job_info:
                raise ValueError(f"Job with ID {job_id} not found.")

            hyper_param_manager = job_managers["hyper_param_manager"]
            hyper_param_manager.update_params(params)
            hyper_param_manager.save_params(job_info["job_folder"])
            
            self.job_manager.update_job_details(job_id, {"hyperparameters": hyper_param_manager.get_hyper_params()})
            
            # Update job status to indicate hyperparameters have been set
            self.update_job_status(job_id, "hyperparameters_set")
            self.logger.info(f"Updated job {job_id} status to 'hyperparameters_set'")
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to save hyperparameters for job {job_id}: {e}", exc_info=True)
            return False

    def set_job_id(self, job_id: str):
        """Sets the current job context for the service."""
        job = self.get_job_by_id(job_id)
        if job:
            self.current_job_id = job_id
            self.logger.info(f"JobService context set to job_id: {job_id}")
            return

        # If we got here, the job wasn't found in the regular way - try to ensure it exists
        job_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 
                                 'output', job_id)
                                 
        if os.path.isdir(job_folder):
            self.logger.info(f"Job folder exists at {job_folder}, attempting to register job {job_id}")
            if self.job_manager.ensure_job_exists(job_id, job_folder):
                self.current_job_id = job_id
                self.logger.info(f"JobService context set to newly registered job_id: {job_id}")
                return
                  # If we reach here, the job couldn't be found or created
        raise ValueError(f"Job with ID '{job_id}' not found in JobManager and could not be created.")
        
    def get_job_id(self):
        """Gets the current job context ID."""
        return self.current_job_id
        
    def setup_training_tasks(self, job_id: str):
        """
        Initializes the training setup manager and runs the setup process.
        
        Returns:
            dict: Information about the training tasks that were set up
        """
        self.logger.info(f"Setting up training tasks for job {job_id}")
        
        # Check job_managers first
        job_managers = self.job_manager.job_managers.get(job_id)
        if not job_managers:
            self.logger.error(f"Job managers for {job_id} not found. Available job managers: {list(self.job_manager.job_managers.keys())}")
            
            # Try to recreate job managers if possible
            job_info = self.job_manager.get_job(job_id)
            if job_info and "job_folder" in job_info:
                self.logger.info(f"Attempting to recreate job managers for {job_id}")
                try:
                    self.job_manager.ensure_job_exists(job_id, job_info["job_folder"])
                    job_managers = self.job_manager.job_managers.get(job_id)
                    if not job_managers:
                        raise ValueError(f"Failed to recreate job managers for {job_id}")
                except Exception as e:
                    self.logger.error(f"Error recreating job managers: {e}")
                    raise ValueError(f"Managers for job with ID {job_id} could not be created: {str(e)}")
            else:
                raise ValueError(f"Job with ID {job_id} not found or missing folder information")
                
        # Get job info
        job_info = self.job_manager.get_job(job_id)
        if not job_info:
            self.logger.error(f"Job {job_id} not found in JobManager")
            raise ValueError(f"Job with ID {job_id} not found.")
            
        # Print job info for debugging
        self.logger.info(f"Job info for {job_id}: status={job_info.get('status')}, folder={job_info.get('job_folder')}")
        self.logger.info(f"Job details: {job_info.get('details', {})}")

        # Get hyperparameters
        hyperparams = job_info.get("details", {}).get("hyperparameters", {})
        if not hyperparams:
            self.logger.error(f"No hyperparameters found for job {job_id}")
            raise ValueError(f"No hyperparameters found for job {job_id}.")
        
        self.logger.info(f"Hyperparameters for job {job_id}: {hyperparams}")

        # Get training setup manager
        training_setup_manager = job_managers["training_setup_manager"]
        training_setup_manager.params = hyperparams
        training_setup_manager.current_hyper_params = hyperparams
        
        # Run the training setup process
        self.logger.info(f"Starting training setup process for job {job_id}")
        try:
            training_setup_manager.setup_training()
            self.logger.info(f"Training setup process completed successfully for job {job_id}")
        except Exception as e:
            self.logger.error(f"Error during training setup process for job {job_id}: {e}", exc_info=True)
            raise ValueError(f"Training setup failed: {str(e)}")
        
        # Get the task list that was generated
        try:
            task_list = training_setup_manager.get_task_list()
            self.logger.info(f"Retrieved task list for job {job_id}: {[task.get('task_id') for task in task_list]}")
        except Exception as e:
            self.logger.error(f"Error getting task list for job {job_id}: {e}", exc_info=True)
            raise ValueError(f"Failed to get task list: {str(e)}")
            
        task_count = len(task_list)
        job_folder = job_info.get("job_folder")

        self.logger.info(f"Generated {task_count} training tasks for job {job_id}")        # Persist the generated tasks
        try:
            # Add an empty history array for the training progress
            task_details = {
                "training_tasks": task_list,
                "history": []  # Initialize empty history for training progress
            }
            
            # Add some sample history data for testing the GUI
            # The sample history was for testing and should be removed.
            
            self.job_manager.update_job_details(job_id, task_details)
            self.logger.info(f"Updated job details with training tasks for job {job_id}")
        except Exception as e:
            self.logger.error(f"Error updating job details for job {job_id}: {e}", exc_info=True)
            raise ValueError(f"Failed to update job details: {str(e)}")
        
        # Update job status to indicate training setup is completed
        self.update_job_status(job_id, "training_setup_completed")
        self.logger.info(f"Updated job {job_id} status to 'training_setup_completed'")
        
        # Return information about the tasks
        return {
            "task_count": task_count,
            "tasks": task_list,
            "job_folder": job_folder
        }

    def _get_current_job_info(self):
        """Helper to get the full info dict for the current job."""
        if not self.current_job_id:
            self.logger.error("No current job ID is set in JobService.")
            return None
        return self.get_job_by_id(self.current_job_id)

    def get_job_folder(self):
        """Gets the current job folder path."""
        if not self.current_job_id:
            return None
        
        job = self.get_job_by_id(self.current_job_id)
        if job:
            return job.get("job_folder")
        return None

    def get_train_folder(self, subfolder: str = "processed_data"):
        """Gets the path to the training data folder for the current job."""
        job_folder = self.get_job_folder()
        if not job_folder:
            return None
        return os.path.join(job_folder, "train_data", subfolder)

    def get_test_folder(self, subfolder: str = "processed_data"):
        """Gets the path to the testing data folder for the current job."""
        job_folder = self.get_job_folder()
        if not job_folder:
            return None
        return os.path.join(job_folder, "test_data", subfolder)

    def update_job_status(self, job_id: str, status: str, data: dict = None):
        """Updates the status of a job via the JobManager's queue."""
        self.logger.info(f"Queueing status update for job {job_id}: {status}")
        self.job_manager.status_queue.put((job_id, status, data))

    def generate_task_configs(self, hyperparams: dict):
        """
        Generates individual task configurations from a dictionary of hyperparameters,
        some of which may contain comma-separated values for tuning.
        """
        self.logger.info("Generating task configurations from hyperparameters.")
        
        fixed_params = {}
        tuning_params = {}
        
        for key, value in hyperparams.items():
            if isinstance(value, str) and ',' in value:
                options = [v.strip() for v in value.split(',') if v.strip()]
                if options:
                    tuning_params[key] = options
                else:
                    self.logger.warning(f"Hyperparameter '{key}' contained commas but no valid values. Ignoring.")
            else:
                fixed_params[key] = value

        if not tuning_params:
            self.logger.info("No hyperparameter tuning detected. Creating a single task.")
            return [hyperparams]

        keys, values = zip(*tuning_params.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        self.logger.info(f"Generated {len(param_combinations)} unique parameter combinations for tuning.")

        task_configs = []
        for combo in param_combinations:
            config = fixed_params.copy()
            config.update(combo)
            task_configs.append(config)
            
        return task_configs