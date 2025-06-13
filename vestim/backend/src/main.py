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
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import uvicorn
import json
import time
import os

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
    response = await call_next(request)
    return response

# --- Dependency Injection ---
def get_job_manager():
    return JobManager()

def get_job_service():
    return JobService()

def get_training_task_manager():
    return TrainingTaskManager()

def get_data_processing_service():
    return DataProcessingService()

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the VEstim Backend API", "status": "online", "timestamp": time.time()}

@app.post("/jobs", response_model=JobResponse)
def create_new_job(payload: JobPayload, job_service: JobService = Depends(get_job_service)):
    job_id, _ = job_service.create_new_job(payload.selections)
    if not job_id:
        raise HTTPException(status_code=500, detail="Failed to create job.")
    job = job_service.get_job_by_id(job_id)
    return job

@app.get("/jobs", response_model=List[JobResponse])
def get_jobs(job_service: JobService = Depends(get_job_service)):
    return job_service.get_all_jobs()

@app.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str, job_service: JobService = Depends(get_job_service)):
    job = job_service.get_job_by_id(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job with ID {job_id} not found")
    return job

@app.post("/jobs/{job_id}/hyperparameters", response_model=Dict[str, str])
def save_hyperparameters(job_id: str, payload: HyperparametersPayload, job_service: JobService = Depends(get_job_service)):
    try:
        job_service.save_hyperparameters(job_id, payload.hyperparameters)
        return {"message": "Hyperparameters saved successfully", "job_id": job_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save hyperparameters: {str(e)}")

@app.delete("/jobs/{job_id}", response_model=Dict[str, str])
def delete_job(job_id: str, job_service: JobService = Depends(get_job_service)):
    if not job_service.delete_job(job_id):
        raise HTTPException(status_code=404, detail=f"Job with ID {job_id} not found")
    return {"message": f"Job {job_id} deleted"}

@app.post("/jobs/{job_id}/setup-training", response_model=Dict[str, Any])
def setup_training(job_id: str, job_service: JobService = Depends(get_job_service)):
    try:
        task_count, job_folder = job_service.setup_training_tasks(job_id)
        return {"message": "Training setup completed successfully", "task_count": task_count, "job_folder": job_folder}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to setup training: {str(e)}")

@app.post("/jobs/process-and-create", response_model=JobResponse)
def process_and_create_job(payload: JobPayload, job_service: JobService = Depends(get_job_service), dps: DataProcessingService = Depends(get_data_processing_service)):
    """
    Creates a new job, processes the associated data, and returns the job details.
    """
    try:
        selections = payload.selections
        job_id, _ = job_service.create_new_job(selections)
        if not job_id:
            raise HTTPException(status_code=500, detail="Failed to create job.")

        dps.process_data(
            job_id=job_id,
            train_files=selections.get("train_files", []),
            test_files=selections.get("test_files", []),
            data_source=selections.get("data_source")
        )
        
        job = job_service.get_job_by_id(job_id)
        return job
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process and create job: {str(e)}")

@app.post("/jobs/{job_id}/start", response_model=Dict[str, str])
def start_job(job_id: str, payload: TaskInfoPayload, job_manager: JobManager = Depends(get_job_manager), ttm: TrainingTaskManager = Depends(get_training_task_manager)):
    """
    Starts a specific task for a job (e.g., training) in a separate process.
    """
    try:
        # Here, you would determine which target function to run based on job type
        # For now, we'll assume it's always a training task.
        target_func = ttm.process_task_in_background
        
        # The task_info for the process needs to be self-contained
        task_info = payload.task_info
        task_info['job_id'] = job_id

        job_manager.start_job(job_id, target_func, task_info)
        return {"message": "Job started successfully", "job_id": job_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start job: {str(e)}")

@app.get("/jobs/{job_id}/status", response_model=JobResponse)
def get_job_status(job_id: str, job_manager: JobManager = Depends(get_job_manager)):
    """
    Get the current status and details of a job from the in-memory registry.
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job with ID {job_id} not found")
    return job

@app.post("/jobs/{job_id}/stop", response_model=Dict[str, str])
def stop_job(job_id: str, job_manager: JobManager = Depends(get_job_manager)):
    """
    Stop a running job process.
    """
    try:
        if job_manager.stop_job(job_id):
            return {"message": "Stop signal sent", "job_id": job_id}
        else:
            raise HTTPException(status_code=400, detail="Job is not running.")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/server/status")
def server_status():
    """
    Returns the current status of the server.
    """
    return {
        "status": "online",
        "timestamp": time.time(),
    }

@app.post("/server/shutdown")
def shutdown_server():
    """
    Shuts down the server.
    """
    os.kill(os.getpid(), 9)
    return {"message": "Server is shutting down"}

# This block allows running the server directly for development
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)