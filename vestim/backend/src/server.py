# ---------------------------------------------------------------------------------
# Author: Biswanath Dehury
# Date: 2025-06-10
# Version: 1.0.0
# Description: Main entrypoint for the VEstim backend service.
# ---------------------------------------------------------------------------------

from fastapi import FastAPI, Depends
import uvicorn
import sys
import os

# Add the project root to the Python path to allow for absolute imports
# This is a common practice for standalone server applications.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi import FastAPI, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any
import uvicorn
import sys
import os

# Add the project root to the Python path to allow for absolute imports
# This is a common practice for standalone server applications.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from vestim.backend.src.services.job_service import JobService
from vestim.backend.src.services.setup_service import SetupService
from vestim.backend.src.services.training_service import TrainingService

# --- Pydantic Models for Request Bodies ---
class HyperparameterPayload(BaseModel):
    """Defines the expected structure for the hyperparameter configuration."""
    params: Dict[str, Any] = Field(..., description="A dictionary containing all hyperparameter settings.")

class TaskInfoPayload(BaseModel):
    """Defines the expected structure for starting a training task."""
    task_info: Dict[str, Any]
    global_params: Dict[str, Any]

# Create the FastAPI application instance
app = FastAPI(
    title="VEstim Backend Service",
    description="Handles all computational tasks for the VEstim application.",
    version="1.0.0"
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
    return {"message": "Welcome to the VEstim Backend API"}

@app.post("/jobs")
def create_new_job(
    payload: HyperparameterPayload,
    background_tasks: BackgroundTasks,
    job_service: JobService = Depends(get_job_service)
):
    """
    Creates a new job, generates all training task configurations from the
    hyperparameters, and starts them in the background.
    """
    job_id, job_folder = job_service.create_new_job()
    
    # Generate all task configurations from the provided hyperparameters
    task_configs = job_service.generate_task_configs(payload.params)
    
    # For each configuration, create and start a training task
    for i, task_params in enumerate(task_configs):
        task_id = f"task_{i+1}"
        task_dir = os.path.join(job_folder, task_id)
        os.makedirs(task_dir, exist_ok=True)

        task_info = {
            "task_id": task_id,
            "task_dir": task_dir,
            "db_log_file": os.path.join(job_folder, "job_log.db"),
            "hyperparams": task_params,
            # Add other necessary fields that might be missing
        }

        training_service = TrainingService(
            task_info=task_info,
            job_service=job_service,
            global_params=payload.params  # Pass the original payload for global settings
        )
        running_tasks[task_id] = training_service
        background_tasks.add_task(training_service.run_task)
        
    return {
        "job_id": job_id,
        "job_folder": job_folder,
        "status": f"{len(task_configs)} training tasks started in background."
    }

@app.get("/jobs/{job_id}/status")
def get_job_status(job_id: str):
    """
    Retrieves the status of a specific job.
    
    NOTE: This is a mocked endpoint for now. It will be updated to read
    a status file from the job's directory.
    """
    # In the future, this will read a status.json file.
    # For now, we simulate completion to allow UI development.
    return {"job_id": job_id, "status": "complete", "message": "Setup tasks completed.", "task_count": 10}

@app.post("/tasks/start")
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

    training_service = TrainingService(
        task_info=payload.task_info,
        job_service=job_service,
        global_params=payload.global_params
    )
    running_tasks[task_id] = training_service
    background_tasks.add_task(training_service.run_task)
    
    return {"status": "Training task started in background", "task_id": task_id}

@app.get("/tasks/{task_id}/status")
def get_training_task_status(task_id: str):
    """
    Retrieves the status of a specific training task.
    """
    status_file = os.path.join("output", running_tasks[task_id].job_service.get_job_id(), task_id, "training_status.json")
    if not os.path.exists(status_file):
        return {"task_id": task_id, "status": "pending", "message": "Task has not started yet."}
    
    with open(status_file, 'r') as f:
        return json.load(f)

@app.post("/tasks/{task_id}/stop")
def stop_training_task(task_id: str):
    """
    Stops a running training task.
    """
    if task_id in running_tasks:
        running_tasks[task_id].stop_task()
        return {"status": "Stop signal sent", "task_id": task_id}
    else:
        raise HTTPException(status_code=404, detail="Task not found or not running.")

# This block allows running the server directly for development
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)