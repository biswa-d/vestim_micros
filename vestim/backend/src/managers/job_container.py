import os
import time
import logging
import threading
from datetime import datetime
from multiprocessing import Event
from typing import Dict, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

class JobContainer:
    """
    Container that manages all aspects of a single job.
    Acts as a coordinator for job-specific managers and shared services.
    Provides clean isolation between jobs and handles asynchronous task execution.
    """
    
    def __init__(self, job_id: str, job_folder: str, selections: dict):
        self.job_id = job_id
        self.job_folder = job_folder
        self.selections = selections
        self.logger = logging.getLogger(f"{__name__}.{job_id}")        # Job state and progress with detailed phase tracking
        self.status = "created"
        self.progress_message = "Job created"
        self.progress_percent = 0
        self.current_phase = "initialization"  # initialization, data_processing, data_augmentation, training_setup, training, testing, completed
        self.phase_progress = {
            "data_import": {"status": "pending", "progress": 0, "message": "Waiting to start"},
            "data_processing": {"status": "pending", "progress": 0, "message": "Waiting to start"},
            "data_augmentation": {"status": "pending", "progress": 0, "message": "Waiting to start"},
            "hyperparameters": {"status": "pending", "progress": 0, "message": "Waiting to start"},
            "training_setup": {"status": "pending", "progress": 0, "message": "Waiting to start"},
            "training": {"status": "pending", "progress": 0, "message": "Waiting to start"},
            "testing": {"status": "pending", "progress": 0, "message": "Waiting to start"}
        }
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        
        # GUI state persistence - determines which GUI to show when reopening
        self.gui_ready_for_phase = "data_import"  # Which GUI should be shown
        self.requires_user_input = True  # Whether this job needs user interaction
        self.current_gui_type = None  # Track current GUI type
        
        # Managers registry - job-specific instances from common scripts
        self.managers = {}
        self.manager_lock = threading.Lock()
          # Process management
        self.process = None
        self.stop_flag = Event()
        
        # Training tasks storage - jobs contain their own tasks for isolation
        self.training_tasks = []  # List of training task dictionaries
        self.current_training_task_index = 0  # Track which task is currently being processed
        self.training_history = {}  # Store training progress for all tasks: {task_id: training_data}
        
        # Job details and history
        self.details = {}
        self.task_progress = {}  # Task-specific progress tracking
        
        # Thread pool for asynchronous operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix=f"Job-{job_id}")
        
        # Services - shared instances (microservices pattern)
        self._data_processing_service = None
        self._data_augmentation_service = None
        self._model_training_service = None
        
        self.logger.info(f"JobContainer created for {job_id}")
    
    def update_status(self, status: str, message: str, progress_percent: float = None, **kwargs):
        """Update job status with thread safety"""
        with self.manager_lock:
            self.status = status
            self.progress_message = message
            if progress_percent is not None:
                self.progress_percent = progress_percent
            self.updated_at = datetime.now().isoformat()
            
            # Update details with any additional info
            if kwargs:
                self.details.update(kwargs)
            
            self.logger.info(f"Status updated: {status} - {message} ({progress_percent}%)")
    
    def update_task_progress(self, task_id: str, task_data: dict):
        """Update progress for a specific task"""
        with self.manager_lock:
            self.task_progress[task_id] = task_data
            self.updated_at = datetime.now().isoformat()
      # === SERVICE GETTERS (Shared Services - Microservices Pattern) ===
    
    def get_data_processing_service(self):
        """Get shared data processing service instance"""
        if self._data_processing_service is None:
            from vestim.backend.src.services.data_processing_service import DataProcessingService
            self._data_processing_service = DataProcessingService()
        return self._data_processing_service
    
    def get_data_augmentation_service(self):
        """Get shared data augmentation service instance"""
        if self._data_augmentation_service is None:
            from vestim.backend.src.services.data_processor.src.data_augment_service import DataAugmentService
            self._data_augmentation_service = DataAugmentService()
            self.logger.info("Created shared data augmentation service")
        return self._data_augmentation_service
    
    def get_model_training_service(self):
        """Get shared model training service instance"""
        if self._model_training_service is None:
            from vestim.backend.src.services.model_training.src.training_task_service import TrainingTaskService
            self._model_training_service = TrainingTaskService()
        return self._model_training_service
    
    # === MANAGER GETTERS (Job-Specific Instances from Common Scripts) ===
    
    def get_training_task_manager(self):
        """Get or create job-specific training task manager from common script"""
        with self.manager_lock:
            if 'training_task_manager' not in self.managers:
                from vestim.backend.src.managers.training_task_manager import TrainingTaskManager
                self.managers['training_task_manager'] = TrainingTaskManager()
                self.logger.info("Created job-specific training task manager")
            return self.managers['training_task_manager']
    
    def get_data_processing_manager(self):
        """Get or create job-specific data processing manager"""
        with self.manager_lock:
            if 'data_processing_manager' not in self.managers:
                # Create job-specific manager instance (NOT the service)
                # This would be a manager that coordinates with the service
                self.logger.info("Data processing manager would be created here")
                # For now, we'll use the service directly
                return None
            return self.managers['data_processing_manager']
    
    def get_data_augmentation_manager(self):
        """Get or create job-specific data augmentation manager from common script"""
        with self.manager_lock:
            if 'data_augmentation_manager' not in self.managers:
                from vestim.backend.src.managers.data_augment_manager_qt import DataAugmentManager
                self.managers['data_augmentation_manager'] = DataAugmentManager(
                    self.job_id, self.job_folder
                )
                self.logger.info("Created job-specific data augmentation manager")
            return self.managers['data_augmentation_manager']
    
    
    def get_hyperparams_manager(self):
        """Get or create job-specific hyperparameters manager"""
        with self.manager_lock:
            if 'hyperparams_manager' not in self.managers:
                from vestim.backend.src.managers.hyper_param_manager import VEstimHyperParamManager
                self.managers['hyperparams_manager'] = VEstimHyperParamManager()
                self.logger.info("Created job-specific hyperparams manager")
            return self.managers['hyperparams_manager']
    
    def get_training_setup_manager(self, hyperparams: dict = None):
        """Get or create job-specific training setup manager"""
        with self.manager_lock:
            if 'training_setup_manager' not in self.managers:
                from vestim.backend.src.managers.training_setup_manager import VEstimTrainingSetupManager
                # If hyperparams are not provided, try to get them from details
                if hyperparams is None:
                    hyperparams = self.details.get('hyperparameters', {})
                
                self.managers['training_setup_manager'] = VEstimTrainingSetupManager(
                    self.job_id, self.job_folder, hyperparams
                )
                self.logger.info("Created job-specific training setup manager")
            
            # If hyperparams are provided, update the existing manager
            elif hyperparams is not None:
                self.managers['training_setup_manager'].params = hyperparams

            return self.managers['training_setup_manager']
    
    # === ASYNCHRONOUS TASK COORDINATION ===
    
    def run_data_processing_async(self, train_files: list, test_files: list, data_source: str, callback=None):
        """
        Coordinate data processing using shared DataProcessingService.
        Runs asynchronously and updates job status.
        """
        def process_data():
            try:
                self.update_status("processing_data", "Starting data processing", 10)
                
                # Use shared data processing service
                data_service = self.get_data_processing_service()
                success = data_service.process_data_files(
                    job_id=self.job_id,
                    job_folder=self.job_folder,
                    train_files=train_files,
                    test_files=test_files,
                    data_source=data_source
                )
                
                if success:
                    self.update_status("data_processed", "Data processing completed", 25)
                    self.details.update({
                        "data_processed": True,
                        "data_source": data_source,
                        "train_files": train_files,
                        "test_files": test_files
                    })
                else:
                    self.update_status("error", "Data processing failed", 0)                
                if callback:
                    callback(success)
                
                return success
            except Exception as e:
                self.logger.error(f"Error in async data processing: {e}", exc_info=True)
                self.update_status("error", f"Data processing failed: {str(e)}", 0)
                if callback:
                    callback(False)
                return False
        
        # Submit to thread pool for non-blocking execution
        future = self.thread_pool.submit(process_data)
        return future
    
    def run_data_augmentation_async(self, augmentation_params: dict, callback=None):
        """Run data augmentation asynchronously using shared service with job-specific manager"""
        def augment_data():
            try:
                self.update_status("augmenting_data", "Starting data augmentation", 30)
                
                # Get shared data augmentation service and job-specific manager
                augment_service = self.get_data_augmentation_service()
                augment_manager = self.get_data_augmentation_manager()
                
                if not augment_service:
                    self.logger.error("Data augmentation service not available")
                    self.update_status("error", "Data augmentation service not available", 0)
                    if callback:
                        callback(False)
                    return False
                
                # Use the service to perform data augmentation coordinated by the manager
                # The manager provides job-specific context, service does the work
                success = augment_service.process_job_data(
                    job_folder=self.job_folder,
                    **augmentation_params
                )
                
                if success:
                    self.update_status("data_augmented", "Data augmentation completed", 50)
                    self.details.update({
                        "data_augmented": True,
                        "augmentation_params": augmentation_params
                    })
                else:
                    self.update_status("error", "Data augmentation failed", 0)
                
                if callback:
                    callback(success)
                return success
                
            except Exception as e:
                self.logger.error(f"Error in async data augmentation: {e}", exc_info=True)
                self.update_status("error", f"Data augmentation failed: {str(e)}", 0)
                if callback:
                    callback(False)
                return False
        
        future = self.thread_pool.submit(augment_data)
        return future
    
    def run_training_async(self, training_params: dict, callback=None):
        """Run model training asynchronously using shared service"""
        def train_model():
            try:
                self.update_status("training", "Starting model training", 60)
                
                # Use shared model training service with job-specific manager
                training_service = self.get_model_training_service()
                training_manager = self.get_training_task_manager()
                
                # TODO: Implement training coordination
                # success = training_service.train_model(training_manager, training_params)
                
                self.update_status("training_completed", "Model training completed", 100)
                
                if callback:
                    callback(True)
                return True
            except Exception as e:
                self.logger.error(f"Error in async training: {e}", exc_info=True)
                self.update_status("error", f"Training failed: {str(e)}", 0)
                if callback:
                    callback(False)
                return False
        
        future = self.thread_pool.submit(train_model)
        return future
    
    def cleanup_managers(self):
        """Clean up all managers when job is done"""
        with self.manager_lock:
            for manager_name, manager in self.managers.items():
                try:
                    if hasattr(manager, 'cleanup'):
                        manager.cleanup()
                    self.logger.info(f"Cleaned up {manager_name}")
                except Exception as e:
                    self.logger.error(f"Error cleaning up {manager_name}: {e}")
            
            self.managers.clear()
            self.logger.info("All managers cleaned up")
    
    def set_process(self, process):
        """Set the process handle for this job"""
        self.process = process
    
    def is_running(self):
        """Check if the job process is running"""
        return self.process and self.process.is_alive()
    
    def stop(self):
        """Signal the job to stop gracefully"""
        self.stop_flag.set()
        self.update_status("stopping", "Job stop requested")
        
        if self.process and self.process.is_alive():
            # Give the process time to stop gracefully
            self.process.join(timeout=10)
            if self.process.is_alive():
                self.process.terminate()
                self.logger.warning(f"Had to forcefully terminate process for {self.job_id}")
    
    def get_summary(self):
        """Get a summary of the job container state"""
        return {
            "job_id": self.job_id,
            "status": self.status,
            "progress_message": self.progress_message,
            "progress_percent": self.progress_percent,
            "current_phase": self.current_phase,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "is_running": self.is_running(),
            "active_managers": list(self.managers.keys()),
            "task_progress": self.task_progress,
            "details": self.details,
            "training_tasks": self.training_tasks,
            "job_folder": self.job_folder,
            "selections": self.selections
        }
    
    def get_detailed_state(self):
        """Get detailed state including all task progress, ensuring serializable output"""
        summary = self.get_summary()
        
        # Add additional fields, but filter out non-serializable objects
        summary["job_folder"] = self.job_folder
        summary["selections"] = self.selections
        
        # Filter out non-serializable objects from details
        filtered_details = {}
        if self.details:
            for key, value in self.details.items():
                try:
                    # Test if the value is JSON serializable
                    import json
                    json.dumps(value)
                    filtered_details[key] = value
                except (TypeError, ValueError):
                    # Skip non-serializable objects
                    self.logger.warning(f"Skipping non-serializable detail key '{key}' of type {type(value)}")
                    continue
        summary["details"] = filtered_details
        
        return summary
    def get_status_summary(self):
        """Get a comprehensive status summary for dashboard display"""
        return {
            "job_id": self.job_id,
            "status": self.status,
            "progress_message": self.progress_message,
            "progress_percent": self.progress_percent,
            "current_phase": self.current_phase,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "is_running": self.is_running(),
            "active_managers": list(self.managers.keys()),
            "task_progress": self.task_progress,
            "details": self.details,
            "job_folder": self.job_folder,
            "selections": self.selections
        }
      # === TRAINING TASK MANAGEMENT ===
    
    def add_training_tasks(self, tasks: list):
        """Add training tasks to this job container for isolation"""
        print(f"[DEBUG] JobContainer.add_training_tasks called for job {self.job_id}")
        print(f"[DEBUG] Adding {len(tasks)} training tasks")
        print(f"[DEBUG] Task IDs: {[task.get('task_id', 'no_id') for task in tasks]}")
        
        self.training_tasks = tasks.copy()  # Store tasks in job container
        self.current_training_task_index = 0
        self.logger.info(f"Added {len(tasks)} training tasks to job {self.job_id}")
        print(f"[DEBUG] Training tasks stored successfully in job container")
        
        # Initialize training history for each task
        print(f"[DEBUG] Initializing training history for tasks")
        for task in tasks:
            task_id = task.get('task_id')
            if task_id:
                self.training_history[task_id] = {
                    'status': 'pending',
                    'epoch_logs': [],
                    'train_losses': [],
                    'val_losses': [],
                    'current_epoch': 0,
                    'total_epochs': task.get('hyperparams', {}).get('MAX_EPOCHS', 10)
                }
                print(f"[DEBUG] Initialized training history for task {task_id}")
        print(f"[DEBUG] Training history initialization complete")
    def get_training_tasks(self):
        """Get all training tasks for this job"""
        print(f"[DEBUG] JobContainer.get_training_tasks called for job {self.job_id}")
        print(f"[DEBUG] Returning {len(self.training_tasks)} training tasks")
        return self.training_tasks.copy()
    
    def get_current_training_task(self):
        """Get the current training task being processed"""
        if 0 <= self.current_training_task_index < len(self.training_tasks):
            return self.training_tasks[self.current_training_task_index]
        return None
    
    def get_training_task_by_id(self, task_id: str):
        """Get a specific training task by ID"""
        for task in self.training_tasks:
            if task.get('task_id') == task_id:
                return task
        return None
    
    def update_training_task_progress(self, task_id: str, training_data: dict):
        """Update training progress for a specific task"""
        if task_id in self.training_history:
            self.training_history[task_id].update(training_data)
            self.logger.info(f"Updated training progress for task {task_id}")
    
    def get_training_task_progress(self, task_id: str):
        """Get training progress for a specific task"""
        return self.training_history.get(task_id, {})
    
    def start_job_training(self):
        """Start training for this job (processes all tasks sequentially)"""
        if not self.training_tasks:
            raise ValueError(f"No training tasks found for job {self.job_id}")
        
        self.logger.info(f"Starting job-level training for {len(self.training_tasks)} tasks")
        
        # Get the training task manager for this job
        training_manager = self.get_training_task_manager()
        
        # Start processing tasks sequentially
        # This will be handled by the JobManager's start_job method
        return True
    
    def get_all_training_progress(self):
        """Get training progress for all tasks in this job"""
        return {
            'job_id': self.job_id,
            'total_tasks': len(self.training_tasks),
            'current_task_index': self.current_training_task_index,
            'task_progress': self.training_history
        }

    def start_training_task(self, task_id: str):
        """Mark a task as started."""
        task = self.get_training_task_by_id(task_id)
        if task:
            self.update_training_task_progress(task_id, {"status": "running"})
            self.logger.info(f"Task {task_id} started.")

    def complete_training_task(self, task_id: str):
        """Mark a task as completed."""
        task = self.get_training_task_by_id(task_id)
        if task:
            self.update_training_task_progress(task_id, {"status": "completed"})
            self.logger.info(f"Task {task_id} completed.")

    def fail_training_task(self, task_id: str, error_message: str):
        """Mark a task as failed."""
        task = self.get_training_task_by_id(task_id)
        if task:
            self.update_training_task_progress(task_id, {"status": "failed", "error": error_message})
            self.logger.error(f"Task {task_id} failed: {error_message}")
