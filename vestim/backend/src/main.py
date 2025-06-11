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

@app.get("/")
def read_root():
    """
    Root endpoint for the API. Provides a simple welcome message.
    """
    return {"message": "Welcome to the VEstim Backend API", "status": "online", "timestamp": time.time()}

@app.post("/jobs", response_model=Dict[str, str])
def create_new_job(payload: JobPayload, job_service: JobService = Depends(get_job_service)):
    """
    Creates a new job and returns the job ID and folder path.
    """
    job_id, job_folder = job_service.create_new_job(payload.selections)
    return {"job_id": job_id, "job_folder": job_folder}

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

@app.post("/jobs/clear", response_model=Dict[str, str])
def clear_all_jobs(job_service: JobService = Depends(get_job_service)):
    """
    Deletes all jobs.
    """
    success = job_service.clear_all_jobs()
    if success:
        return {"message": "All jobs have been cleared."}
    else:
        raise HTTPException(status_code=500, detail="Failed to clear all jobs")

@app.post("/tasks/stop_all", response_model=Dict[str, str])
def stop_all_tasks():
    """
    Stops all running tasks.
    """
    global running_tasks
    for task_id, service in running_tasks.items():
        service.stop_task()
    running_tasks.clear()
    return {"status": "Stop signal sent to all tasks"}

@app.post("/tasks/start", response_model=Dict[str, str])
def start_training_task(
    payload: TaskInfoPayload,
    background_tasks: BackgroundTasks,
    job_service: JobService = Depends(get_job_service)
):
    """
    Starts a new training task in the background.
    """
    task_id = payload.task_info.get("task_id")
    if not task_id:
        raise HTTPException(status_code=400, detail="Task ID is missing from payload.")

    # Set the job ID in the job service
    job_id = payload.task_info.get("job_id")
    if job_id:
        job_service.set_job_id(job_id)
    
    training_service = TrainingService(
        task_info=payload.task_info,
        job_service=job_service,
        global_params=payload.global_params
    )
    running_tasks[task_id] = training_service
    background_tasks.add_task(training_service.run_task)
    
    return {"status": "Training task started in background", "task_id": task_id}

@app.get("/tasks/{task_id}/status", response_model=TaskStatusResponse)
def get_training_task_status(task_id: str):
    """
    Retrieves the status of a specific training task.
    """
    # Try to get the task from running tasks first
    if task_id in running_tasks:
        service = running_tasks[task_id]
        job_id = service.job_service.get_job_id()
        status_file = os.path.join("output", job_id, task_id, "training_status.json")
    else:
        # If task not in running tasks, try to find its status file
        status_file = None
        for job_dir in os.listdir("output"):
            if job_dir.startswith("job_"):
                task_status_file = os.path.join("output", job_dir, task_id, "training_status.json")
                if os.path.exists(task_status_file):
                    status_file = task_status_file
                    break
    
    # Return status if found
    if status_file and os.path.exists(status_file):
        try:
            with open(status_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            return {"task_id": task_id, "status": "error", "message": f"Error reading status: {str(e)}"}
    
    # Task not found
    return {"task_id": task_id, "status": "unknown", "message": "Task status not found"}

@app.post("/tasks/{task_id}/stop", response_model=Dict[str, str])
def stop_training_task(task_id: str):
    """
    Stops a running training task.
    """
    if task_id in running_tasks:
        running_tasks[task_id].stop_task()
        return {"status": "Stop signal sent", "task_id": task_id}
    else:
        raise HTTPException(status_code=404, detail="Task not found or not running.")

@app.get("/server/status")
def server_status():
    """
    Returns the current status of the server.
    """
    return {
        "status": "online",
        "timestamp": time.time(),
        "running_tasks": len(running_tasks),
        "task_ids": list(running_tasks.keys())
    }

# This block allows running the server directly for development
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)