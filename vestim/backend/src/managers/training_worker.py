"""
Simple training worker module for multiprocessing.
Contains only essential training logic without complex state or locks.
"""

import os
import sys
import logging
import traceback
from typing import Dict, List, Any


def setup_logging(job_id: str):
    """Setup logging for the training worker"""
    logger = logging.getLogger(f"training_worker_{job_id}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def process_training_tasks(status_queue, task_info: Dict[str, Any]):
    """
    Simple module-level function that can be pickled for multiprocessing.
    Processes all training tasks for a job sequentially.
    
    Args:
        status_queue: Queue to send status updates
        task_info: Dictionary containing job_id, training_tasks, job_folder
    """
    job_id = None
    logger = None
    
    try:
        print(f"[DEBUG] process_training_tasks called")
        print(f"[DEBUG] task_info keys: {list(task_info.keys()) if isinstance(task_info, dict) else 'not a dict'}")
        
        # Extract basic info
        job_id = task_info.get('job_id')
        training_tasks = task_info.get('training_tasks', [])
        job_folder = task_info.get('job_folder')
        
        if not job_id:
            raise ValueError("No job_id provided in task_info")
        
        if not training_tasks:
            raise ValueError(f"No training tasks found for job {job_id}")
        
        # Setup logging
        logger = setup_logging(job_id)
        logger.info(f"Starting training worker for job {job_id}")
        print(f"[DEBUG] Training worker started for job {job_id}")
        print(f"[DEBUG] Processing {len(training_tasks)} tasks")
        
        # Send initial status
        status_queue.put((job_id, 'training_started', {
            "message": f"Training started for {len(training_tasks)} tasks",
            "total_tasks": len(training_tasks)
        }))
        
        # Process each task
        for i, task in enumerate(training_tasks):
            if not task:
                logger.warning(f"Skipping empty task at index {i}")
                continue
                
            task_id = task.get('task_id', f'task_{i}')
            logger.info(f"Processing task {i+1}/{len(training_tasks)}: {task_id}")
            print(f"[DEBUG] Processing task {task_id}")
            
            # Send progress update
            progress = ((i) / len(training_tasks)) * 100
            status_queue.put((job_id, 'training_progress', {
                "message": f"Processing task {i+1}/{len(training_tasks)}: {task_id}",
                "progress": progress,
                "task_id": task_id,
                "task_index": i,
                "total_tasks": len(training_tasks)
            }))
            
            # Process the individual task
            try:
                process_single_task(status_queue, task, logger)
                logger.info(f"Completed task {task_id}")
                print(f"[DEBUG] Completed task {task_id}")
                
            except Exception as e:
                error_msg = f"Error in task {task_id}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                print(f"[DEBUG] Error in task {task_id}: {e}")
                
                status_queue.put((job_id, 'task_error', {
                    "message": error_msg,
                    "task_id": task_id,
                    "error": str(e)
                }))
                # Continue with next task
                continue
        
        # All tasks completed
        logger.info(f"All {len(training_tasks)} tasks completed successfully")
        print(f"[DEBUG] All tasks completed for job {job_id}")
        
        status_queue.put((job_id, 'training_completed', {
            "message": f"All {len(training_tasks)} tasks completed successfully",
            "total_tasks": len(training_tasks),
            "progress": 100
        }))
        
    except Exception as e:
        error_msg = f"Error in training worker: {str(e)}"
        print(f"[DEBUG] Exception in process_training_tasks: {e}")
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        
        if logger:
            logger.error(error_msg, exc_info=True)
        
        if job_id:
            status_queue.put((job_id, 'training_error', {
                "message": error_msg,
                "error": str(e),
                "traceback": traceback.format_exc()
            }))


def process_single_task(status_queue, task: Dict[str, Any], logger):
    """
    Process a single training task.
    
    Args:
        status_queue: Queue to send status updates
        task: Task dictionary containing all task information
        logger: Logger instance
    """
    task_id = task.get('task_id', 'unknown_task')
    
    try:
        logger.info(f"Starting training for task {task_id}")
        print(f"[DEBUG] Starting training for task {task_id}")
        
        # Import training services (only when needed)
        try:
            from vestim.backend.src.services.model_training.src.data_loader_service import DataLoaderService
            from vestim.backend.src.services.model_training.src.training_task_service import TrainingTaskService
            from vestim.backend.src.services.model_training.src.LSTM_model_service import LSTMModelService
        except ImportError as e:
            raise ImportError(f"Failed to import training services: {e}")
        
        # Initialize services
        data_loader_service = DataLoaderService()
        training_service = TrainingTaskService()
        model_service = LSTMModelService()
        
        logger.info(f"Training services initialized for task {task_id}")
        print(f"[DEBUG] Training services initialized for task {task_id}")
        
        # Send task started status
        status_queue.put((task.get('job_id', 'unknown'), 'task_started', {
            "message": f"Started training task {task_id}",
            "task_id": task_id
        }))
        
        # Load data
        logger.info(f"Loading data for task {task_id}")
        print(f"[DEBUG] Loading data for task {task_id}")
        
        data_loader_params = task.get('data_loader_params', {})
        job_folder = task.get('job_folder')
        
        # Prepare data loading
        train_data, val_data = data_loader_service.load_and_prepare_data(
            job_folder=job_folder,
            **data_loader_params
        )
        
        logger.info(f"Data loaded for task {task_id}")
        print(f"[DEBUG] Data loaded for task {task_id}")
        
        # Create model
        logger.info(f"Creating model for task {task_id}")
        print(f"[DEBUG] Creating model for task {task_id}")
        
        model_metadata = task.get('model_metadata', {})
        model = model_service.create_model(**model_metadata)
        
        logger.info(f"Model created for task {task_id}")
        print(f"[DEBUG] Model created for task {task_id}")
        
        # Train model
        logger.info(f"Starting training for task {task_id}")
        print(f"[DEBUG] Starting actual training for task {task_id}")
        
        training_params = task.get('training_params', {})
        hyperparams = task.get('hyperparams', {})
        
        # Start training with progress callback
        def progress_callback(epoch, train_loss, val_loss, **kwargs):
            """Callback to send training progress"""
            progress_data = {
                "task_id": task_id,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "message": f"Task {task_id} - Epoch {epoch}"
            }
            progress_data.update(kwargs)
            
            status_queue.put((task.get('job_id', 'unknown'), 'epoch_progress', progress_data))
        
        # Execute training
        results = training_service.train_model(
            model=model,
            train_data=train_data,
            val_data=val_data,
            hyperparams=hyperparams,
            training_params=training_params,
            progress_callback=progress_callback
        )
        
        logger.info(f"Training completed for task {task_id}")
        print(f"[DEBUG] Training completed for task {task_id}")
        
        # Send completion status
        status_queue.put((task.get('job_id', 'unknown'), 'task_completed', {
            "message": f"Task {task_id} completed successfully",
            "task_id": task_id,
            "results": results
        }))
        
    except Exception as e:
        error_msg = f"Error in task {task_id}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        print(f"[DEBUG] Error in process_single_task: {e}")
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        
        # Send error status
        status_queue.put((task.get('job_id', 'unknown'), 'task_error', {
            "message": error_msg,
            "task_id": task_id,
            "error": str(e)
        }))
        
        raise  # Re-raise to let caller handle
