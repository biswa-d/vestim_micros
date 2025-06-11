# ---------------------------------------------------------------------------------
# Author: Biswanath Dehury
# Date: 2025-06-10
# Version: 1.0.0
# Description: Main entrypoint for the VEstim backend service.
# ---------------------------------------------------------------------------------

import sys
import os
# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from fastapi import FastAPI, Depends, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import uvicorn
import json
import time


from vestim.backend.src.services.job_service import JobService
from vestim.backend.src.managers.training_task_manager_qt import TrainingTaskManager
from vestim.backend.src.managers.training_setup_manager_qt import VEstimTrainingSetupManager
from vestim.backend.src.managers.hyper_param_manager_qt import VEstimHyperParamManager
# from vestim.backend.src.managers.data_augment_manager_qt import DataAugmentManager  # Temporarily commented out due to syntax errors
from vestim.backend.src.services.training_service import TrainingService

# --- Pydantic Models for Request Bodies ---
class HyperparameterPayload(BaseModel):
    """Defines the expected structure for the hyperparameter configuration."""
    params: Dict[str, Any] = Field(..., description="A dictionary containing all hyperparameter settings.")

class TaskInfoPayload(BaseModel):
    """Defines the expected structure for starting a training task."""
    task_info: Dict[str, Any]
    global_params: Dict[str, Any]

class JobPayload(BaseModel):
    """Defines the expected structure for creating a new job."""
    selections: Dict[str, Any]

class JobResponse(BaseModel):
    """Defines the structure for job response data."""
    job_id: str
    status: str
    created_at: str
    updated_at: str
    selections: Dict[str, Any]
    message: Optional[str] = None

class TaskStatusResponse(BaseModel):
    """Defines the structure for task status response."""
    task_id: str
    status: str
    progress: Optional[float] = None
    message: Optional[str] = None
    error: Optional[str] = None

# Create the FastAPI application instance
app = FastAPI(
    title="VEstim Backend Service",
    description="Handles all computational tasks for the VEstim application.",
    version="1.0.0"
)

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# A simple in-memory dictionary to store running background tasks
# In a production scenario, a more robust solution like Redis or a database would be used.
running_tasks: Dict[str, TrainingService] = {}

# --- Dependency Injection ---
def get_job_service():
    """Dependency injector for the JobService."""
    return JobService()

def get_training_task_manager():
    """Dependency injector for the TrainingTaskManager."""
    return TrainingTaskManager()


def get_training_setup_manager():
    """Dependency injector for the VEstimTrainingSetupManager."""
    return VEstimTrainingSetupManager()

def get_hyper_param_manager():
    """Dependency injector for the VEstimHyperParamManager."""
    return VEstimHyperParamManager()

def get_data_augment_manager():
    """Dependency injector for the DataAugmentManager."""
    # return DataAugmentManager()  # Temporarily commented out due to syntax errors
    return None

@app.get("/")
def read_root():
    """
    Root endpoint for the API. Provides a simple welcome message.
    """
    return {"message": "Welcome to the VEstim Backend API", "status": "online", "timestamp": time.time()}

@app.post("/jobs", response_model=JobResponse)
def create_new_job(payload: JobPayload, job_service: JobService = Depends(get_job_service)):
    """
    Creates a new job and returns the job details.
    """
    job_id, _ = job_service.create_new_job(payload.selections)
    job = job_service.get_job_by_id(job_id)
    if not job:
        raise HTTPException(status_code=500, detail="Failed to create job.")
    return job

@app.get("/jobs", response_model=List[JobResponse])
def get_jobs(job_service: JobService = Depends(get_job_service)):
    """
    Returns a list of all jobs.
    """
    return job_service.get_all_jobs()

@app.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str, job_service: JobService = Depends(get_job_service)):
    """
    Returns details of a specific job.
    """
    job = job_service.get_job_by_id(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job with ID {job_id} not found")
    return job

@app.post("/jobs/{job_id}/train", response_model=Dict[str, str])
def train_job(job_id: str, background_tasks: BackgroundTasks, training_task_manager: TrainingTaskManager = Depends(get_training_task_manager)):
    """
    Starts the training process for a specific job.
    """
    # This is a simplified implementation. In a real application, you would
    # retrieve the job details and pass them to the training task manager.
    task_info = {"job_id": job_id, "task_id": f"task_{job_id}"}
    background_tasks.add_task(training_task_manager.process_task, task_info, None)
    return {"message": "Training started", "job_id": job_id}

@app.get("/jobs/{job_id}/status", response_model=TaskStatusResponse)
def get_job_status(job_id: str):
    """
    Get the current status of a job.
    """
    # This is a simplified implementation. In a real application, you would
    # have a more robust way of tracking job status.
    task_id = f"task_{job_id}"
    status_file = os.path.join("output", job_id, task_id, "training_status.json")
    if os.path.exists(status_file):
        try:
            with open(status_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            return {"task_id": task_id, "status": "error", "message": f"Error reading status: {str(e)}"}
    return {"task_id": task_id, "status": "unknown", "message": "Task status not found"}

@app.get("/jobs/{job_id}/results", response_model=Dict[str, Any])
def get_job_results(job_id: str):
    """
    Get the results of a completed job.
    """
    # This is a placeholder implementation.
    return {"job_id": job_id, "results": "Not implemented yet."}

@app.post("/jobs/{job_id}/stop", response_model=Dict[str, str])
def stop_job(job_id: str, training_task_manager: TrainingTaskManager = Depends(get_training_task_manager)):
    """
    Stop a running job.
    """
    task_id = f"task_{job_id}"
    training_task_manager.stop_task(task_id)
    return {"message": "Stop signal sent", "job_id": job_id}

@app.post("/jobs/{job_id}/test", response_model=Dict[str, str])
def test_job(job_id: str, background_tasks: BackgroundTasks):
    """
    Starts the testing process for a specific job.
    """
    # This is a simplified implementation. In a real application, you would
    # retrieve the job details and pass them to a testing service.
    task_info = {"job_id": job_id, "task_id": f"task_{job_id}"}
    # background_tasks.add_task(testing_service.process_task, task_info, None)
    return {"message": "Testing started", "job_id": job_id}

@app.get("/jobs/{job_id}/testing_status", response_model=TaskStatusResponse)
def get_job_testing_status(job_id: str):
    """
    Get the current status of a testing job.
    """
    # This is a simplified implementation. In a real application, you would
    # have a more robust way of tracking job status.
    task_id = f"task_{job_id}"
    status_file = os.path.join("output", job_id, task_id, "testing_status.json")
    if os.path.exists(status_file):
        try:
            with open(status_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            return {"task_id": task_id, "status": "error", "message": f"Error reading status: {str(e)}"}
    return {"task_id": task_id, "status": "unknown", "message": "Task status not found"}

@app.get("/server/status")
def server_status():
    """
    Returns the current status of the server.
    """
    return {
        "status": "online",
        "timestamp": time.time(),
    }

# This block allows running the server directly for development
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)