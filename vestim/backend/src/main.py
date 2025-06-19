# ---------------------------------------------------------------------------------
# Author: Biswanath Dehury
# Date: 2025-06-10
# Version: 1.0.0
# Description: Main entrypoint for the VEstim backend service.
# ---------------------------------------------------------------------------------
#
# saving commit


import sys
import os
# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import uvicorn
import json
import time
import os
import traceback

from vestim.backend.src.managers.job_manager import JobManager
from vestim.backend.src.services.job_service import JobService
from vestim.backend.src.managers.training_task_manager_qt import TrainingTaskManager
from vestim.backend.src.managers.training_setup_manager_qt import VEstimTrainingSetupManager
from vestim.backend.src.managers.hyper_param_manager_qt import VEstimHyperParamManager
# from vestim.backend.src.managers.data_augment_manager_qt import DataAugmentManager
from vestim.backend.src.services.training_service import TrainingService
from vestim.backend.src.services.data_processing_service import DataProcessingService

# --- Pydantic Models ---
class JobPayload(BaseModel):
    selections: Dict[str, Any]

class JobResponse(BaseModel):
    job_id: str
    status: str
    created_at: str
    updated_at: str
    selections: Dict[str, Any]
    job_folder: str
    details: Optional[Dict[str, Any]] = None

class SimpleJobResponse(BaseModel):
    status: str
    job_id: str
    message: str
    job_folder: str

class TaskInfoPayload(BaseModel):
    task_info: Dict[str, Any]

class HyperparametersPayload(BaseModel):
    hyperparameters: Dict[str, Any]

# --- FastAPI App ---
app = FastAPI(
    title="VEstim Backend Service",
    description="Handles all computational tasks for the VEstim application.",
    version="2.0.0"
)

# --- Global Exception Handler ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler to prevent server crashes"""
    error_detail = f"Internal Server Error: {str(exc)}"
    print(f"EXCEPTION: {error_detail}")
    print(f"TRACEBACK: {traceback.format_exc()}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": error_detail}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Validation error handler for API request validation failures"""
    print(f"VALIDATION ERROR: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(exc)}
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log every incoming request."""
    print(f"Incoming request: {request.method} {request.url}")
    print(f"Headers: {request.headers}")
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        print(f"Middleware caught exception: {e}")
        print(traceback.format_exc())
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": f"Internal Server Error: {str(e)}"}
        )

# --- Dependency Injection ---
def get_job_manager():
    try:
        from vestim.backend.src.managers.job_manager import JobManager
        return JobManager()
    except Exception as e:
        print(f"Error initializing JobManager: {e}")
        print(traceback.format_exc())
        raise

def get_job_service():
    try:
        from vestim.backend.src.services.job_service import JobService
        return JobService()
    except Exception as e:
        print(f"Error initializing JobService: {e}")
        print(traceback.format_exc())
        raise

def get_training_task_manager():
    try:
        from vestim.backend.src.managers.training_task_manager_qt import TrainingTaskManager
        return TrainingTaskManager()
    except Exception as e:
        print(f"Error initializing TrainingTaskManager: {e}")
        print(traceback.format_exc())
        raise

def get_data_processing_service():
    try:
        from vestim.backend.src.services.data_processing_service import DataProcessingService
        return DataProcessingService()
    except Exception as e:
        print(f"Error initializing DataProcessingService: {e}")
        print(traceback.format_exc())
        raise

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the VEstim Backend API", "status": "online", "timestamp": time.time()}

@app.get("/health")
def health_check():
    """A simple health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/jobs", response_model=JobResponse)
def create_new_job(payload: JobPayload, job_service: JobService = Depends(get_job_service)):
    try:
        job_id, _ = job_service.create_new_job(payload.selections)
        if not job_id:
            raise HTTPException(status_code=500, detail="Failed to create job.")
        job = job_service.get_job_by_id(job_id)
        return job
    except Exception as e:
        print(f"Error creating new job: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to create job: {str(e)}")

@app.get("/jobs", response_model=List[JobResponse])
def get_all_jobs(job_service: JobService = Depends(get_job_service)):
    try:
        jobs = job_service.get_all_jobs()
        return jobs
    except Exception as e:
        print(f"Error getting all jobs: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to retrieve jobs: {str(e)}")

@app.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str, job_service: JobService = Depends(get_job_service)):
    try:
        job = job_service.get_job_by_id(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
        
        # Add debugging information for development/testing
        print(f"Serving job details for {job_id}: {job}")
        
        # Make sure the job has necessary training data structure
        if 'details' not in job:
            job['details'] = {}
        return job
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting job {job_id}: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to retrieve job: {str(e)}")

@app.get("/jobs/{job_id}/detailed-status")
def get_job_detailed_status(job_id: str, job_service: JobService = Depends(get_job_service)):
    """Get detailed status including phase progress for GUI synchronization"""
    try:
        job_container = job_service.job_manager.get_job_container(job_id)
        if not job_container:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
        
        # Return detailed status for GUI synchronization
        return {
            "job_id": job_id,
            "status": job_container.status,
            "current_phase": job_container.current_phase,
            "progress_percent": job_container.progress_percent,
            "progress_message": job_container.progress_message,
            "phase_progress": job_container.phase_progress,
            "updated_at": job_container.updated_at,
            "job_folder": job_container.job_folder
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting detailed status for job {job_id}: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")

class JobStatusUpdatePayload(BaseModel):
    status: str
    message: str
    progress_percent: Optional[int] = None

@app.post("/jobs/{job_id}/update-status")
def update_job_status(job_id: str, payload: JobStatusUpdatePayload, job_service: JobService = Depends(get_job_service)):
    """Update job status from GUI when phases complete"""
    try:
        job_container = job_service.job_manager.get_job_container(job_id)
        if not job_container:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
        
        # Update the job status
        job_container.update_status(
            status=payload.status,
            message=payload.message,
            progress_percent=payload.progress_percent
        )
        
        return {"status": "success", "message": f"Job {job_id} status updated to {payload.status}"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error updating status for job {job_id}: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to update job status: {str(e)}")
        print(f"Error getting job {job_id}: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to retrieve job: {str(e)}")

@app.post("/jobs/{job_id}/hyperparameters")
def save_hyperparameters(job_id: str, payload: HyperparametersPayload, job_service: JobService = Depends(get_job_service)):
    try:
        success = job_service.save_hyperparameters(job_id, payload.hyperparameters)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save hyperparameters.")
        return {"status": "success", "message": "Hyperparameters saved successfully."}
    except Exception as e:
        print(f"Error saving hyperparameters for job {job_id}: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to save hyperparameters: {str(e)}")


@app.post("/jobs/{job_id}/stop")
def stop_job(job_id: str, job_service: JobService = Depends(get_job_service)):
    try:
        success = job_service.stop_job(job_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to stop job.")
        return {"status": "success", "message": f"Job {job_id} stopped successfully."}
    except Exception as e:
        print(f"Error stopping job {job_id}: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to stop job: {str(e)}")

@app.post("/jobs/{job_id}/stop-training")
def stop_training(job_id: str, job_service: JobService = Depends(get_job_service)):
    """Gracefully stop training for a specific job after current epoch"""
    try:
        job_container = job_service.job_manager.get_job_container(job_id)
        if not job_container:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
        
        # Call the stop training method on the job container
        success = job_container.stop_training()
        
        if success:
            return {
                "status": "success", 
                "message": f"Training stop signal sent to job {job_id}. Training will stop after current epoch."
            }
        else:
            return {
                "status": "error", 
                "message": f"Job {job_id} is not currently training or no training process found."
            }
            
    except Exception as e:
        print(f"Error stopping training for job {job_id}: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to stop training: {str(e)}")

@app.delete("/jobs/{job_id}")
def delete_job(job_id: str, job_service: JobService = Depends(get_job_service)):
    try:
        success = job_service.delete_job(job_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
        return {"status": "success", "message": f"Job {job_id} deleted successfully."}
    except Exception as e:
        print(f"Error deleting job {job_id}: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to delete job: {str(e)}")

@app.post("/jobs/process-and-create", response_model=SimpleJobResponse)
def process_and_create_job(
    payload: JobPayload,
    job_service: JobService = Depends(get_job_service)
):
    """
    Complete job creation workflow:
    1. Create JobContainer with folder structure
    2. Process and organize data files (train/test folders)
    3. Perform initial data processing based on data source
    4. Return job ID and success status
    """
    try:
        selections = payload.selections
        train_files = selections.get('train_files', [])
        test_files = selections.get('test_files', [])
        data_source = selections.get('data_source', 'Unknown')
        
        print(f"Starting complete job creation workflow")
        print(f"Data source: {data_source}")
        print(f"Train files: {train_files}")
        print(f"Test files: {test_files}")
        
        # Step 1: Create JobContainer with basic structure
        print("Step 1: Creating JobContainer...")
        job_id = job_service.job_manager.create_job(selections)
        if not job_id:
            print("ERROR: Failed to create JobContainer")
            raise HTTPException(status_code=500, detail="Failed to create job container")
        
        print(f"JobContainer created successfully: {job_id}")
        
        # Step 2: Get the JobContainer and start async data processing
        job_container = job_service.job_manager.get_job_container(job_id)
        if not job_container:
            print(f"ERROR: Could not retrieve JobContainer for {job_id}")
            raise HTTPException(status_code=500, detail=f"Could not retrieve job container for {job_id}")
        
        print("Step 2: Starting asynchronous data processing...")
        
        def on_data_processing_complete(success):
            if success:
                print(f"Background data processing completed successfully for job {job_id}")
            else:
                print(f"Background data processing failed for job {job_id}")
        
        # Use the job container's async data processing
        future = job_container.run_data_processing_async(
            train_files=train_files,
            test_files=test_files,
            data_source=data_source,
            callback=on_data_processing_complete        )
        
        print(f"Data processing started asynchronously for job {job_id}")
        
        # Step 3: Return immediately with job ID and folder - processing continues in background
        return SimpleJobResponse(
            status="success",
            job_id=job_id,
            job_folder=job_container.job_folder,
            message=f"Job {job_id} created successfully. Data processing started in background."
        )
    except Exception as e:
        print(f"Error processing and creating job: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to process and create job: {str(e)}")

@app.post("/jobs/{job_id}/ensure")
def ensure_job_exists(job_id: str, job_service: JobService = Depends(get_job_service)):
    """
    Ensures a job exists in the JobManager. If not, creates it from an existing job folder.
    This is useful when opening GUIs in the workflow where the job might exist on disk
    but not be registered in the JobManager.
    """
    try:
        job = job_service.get_job_by_id(job_id)
        if job:
            return {"status": "success", "message": f"Job {job_id} already exists", "job": job}
        
        # Job not found, try to set it as current (which will create it if the folder exists)
        try:
            job_service.set_job_id(job_id)
            job = job_service.get_job_by_id(job_id)
            return {"status": "success", "message": f"Job {job_id} created from existing folder", "job": job}
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Job folder for {job_id} not found: {str(e)}")
            
    except Exception as e:
        print(f"Error ensuring job {job_id} exists: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to ensure job exists: {str(e)}")

@app.post("/server/shutdown")
def shutdown():
    """Shutdown the server gracefully."""
    print("Shutdown signal received. Server will stop within 5 seconds.")
    # Create a task to shutdown the server after sending the response
    import threading
    def shutdown_server():
        import time
        time.sleep(1)  # Give time for the response to be sent
        import os, signal
        # Send SIGTERM to self
        os.kill(os.getpid(), signal.SIGTERM)
    
    threading.Thread(target=shutdown_server).start()
    return {"message": "Shutdown signal received. Server will stop."}

@app.post("/jobs/{job_id}/setup-training")
async def setup_training_tasks(
    job_id: str, 
    job_service: JobService = Depends(get_job_service)
):
    """
    Sets up training tasks for the specified job.
    This endpoint is called after hyperparameters are set and before actual training begins.
    It analyzes the job data, creates the necessary training tasks, and prepares the model architecture.
    """
    try:
        print(f"Setting up training tasks for job {job_id}")
        
        # Get job info before setup to confirm it exists
        job_info_before = job_service.get_job_by_id(job_id)
        if not job_info_before:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
            
        # Setup training tasks
        try:
            result = job_service.setup_training_tasks(job_id)
            print(f"Setup training tasks result: {result}")
        except Exception as setup_error:
            print(f"Error in setup_training_tasks: {setup_error}")
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Training setup failed: {str(setup_error)}")
            
        # Get updated job info with training tasks
        job_info = job_service.get_job_by_id(job_id)
        if not job_info:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found after setup.")
        
        # Extract task information for the response
        tasks = job_info.get("details", {}).get("training_tasks", [])
        task_count = len(tasks)
        
        # Update job status to indicate training is ready
        job_service.job_manager.update_job_details(
            job_id, 
            {"status": "training_setup_complete"}
        )
        
        return {
            "status": "success", 
            "message": f"Training setup completed successfully with {task_count} tasks.",
            "task_count": task_count,
            "job_id": job_id
        }
    except ValueError as ve:
        print(f"Value error during training setup for job {job_id}: {ve}")
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        print(f"Error setting up training tasks for job {job_id}: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to set up training: {str(e)}")

@app.post("/jobs/{job_id}/tasks/{task_id}/start_training")
def start_training_task(
    job_id: str,
    task_id: str,
    job_service: JobService = Depends(get_job_service)
):
    """
    Start a specific training task for a job using JobContainer architecture.
    This endpoint gets the job-specific training task manager from the job container.
    """
    try:
        print(f"Starting training task {task_id} for job {job_id}")
        
        # Get job container (which contains job-specific managers)
        job_container = job_service.job_manager.get_job_container(job_id)
        if not job_container:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
        
        # Get the job-specific training task manager
        training_task_manager = job_container.get_training_task_manager()
        
        # Get job information
        job_info = job_service.get_job_by_id(job_id)
        if not job_info:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
            
        # Extract task information
        task_list = job_info.get("details", {}).get("training_tasks", [])
        matching_tasks = [task for task in task_list if task.get("task_id") == task_id]
        
        if not matching_tasks:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found for job {job_id}")
            
        task_info = matching_tasks[0]
        task_info["job_id"] = job_id
        
        # Update job container status
        job_container.update_status("starting_training", f"Starting training task {task_id}", 0)
        
        # Start the training process using the job-specific manager
        success = job_service.start_job(
            job_id, 
            training_task_manager.process_task_in_background, 
            task_info
        )
        
        if not success:
            job_container.update_status("error", f"Failed to start training task {task_id}")
            raise HTTPException(status_code=500, detail=f"Failed to start training task {task_id}")
            
        # Update job container status to indicate training is in progress
        job_container.update_status("training", f"Training task {task_id} started", 1)
        
        return {
            "status": "success",
            "message": f"Training task {task_id} started for job {job_id}",
            "task_id": task_id,
            "job_id": job_id
        }
    except ValueError as ve:
        print(f"Value error starting task {task_id} for job {job_id}: {ve}")
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Error starting training task {task_id} for job {job_id}: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to start training task: {str(e)}")

@app.get("/jobs/{job_id}/tasks/{task_id}/status")
def get_training_task_status(
    job_id: str,
    task_id: str,
    job_service: JobService = Depends(get_job_service)
):
    """
    Get the status and progress of a specific training task using JobContainer.
    """
    try:
        # Get job container
        job_container = job_service.job_manager.get_job_container(job_id)
        if not job_container:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
            
        # Get job information
        job_info = job_service.get_job_by_id(job_id)
        if not job_info:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
            
        # Check if task exists
        task_list = job_info.get("details", {}).get("training_tasks", [])
        matching_tasks = [task for task in task_list if task.get("task_id") == task_id]
        
        if not matching_tasks:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found for job {job_id}")
        
        # Get task-specific progress from job container
        task_progress = job_container.task_progress.get(task_id, {})
        
        # If no task-specific progress, create basic response from job container status
        if not task_progress:
            task_progress = {
                "status": job_container.status,
                "message": job_container.progress_message,
                "current_epoch": 0,
                "total_epochs": 0,
                "progress_percent": job_container.progress_percent            }
        
        return {
            "status": "success",
            "task_id": task_id,
            "job_id": job_id,
            "job_status": job_container.status,
            "task_progress": task_progress
        }
    except Exception as e:
        print(f"Error getting status for task {task_id} of job {job_id}: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {str(e)}")

# --- Server Startup ---
def start_server(host="0.0.0.0", port=8001):
    """Start the FastAPI server with the specified host and port."""
    print(f"Starting VEstim backend server on {host}:{port}...")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    # If this file is run directly, start the server
    print("Starting VEstim backend server...")
    start_server()