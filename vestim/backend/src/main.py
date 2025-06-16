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
        return job
    except HTTPException:
        raise
    except Exception as e:
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

@app.post("/jobs/{job_id}/start")
def start_job(job_id: str, payload: TaskInfoPayload, 
             job_service: JobService = Depends(get_job_service),
             training_task_manager: TrainingTaskManager = Depends(get_training_task_manager)):
    try:
        # Add the job_id to the task_info
        task_info = payload.task_info
        task_info['job_id'] = job_id
        
        # Start the training process
        success = job_service.start_job(
            job_id, 
            training_task_manager.process_task_in_background, 
            task_info
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to start job.")
        
        return {"status": "success", "message": f"Job {job_id} started successfully."}
    except Exception as e:
        print(f"Error starting job {job_id}: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to start job: {str(e)}")

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

@app.post("/jobs/process-and-create", response_model=JobResponse)
def process_and_create_job(
    payload: JobPayload,
    job_service: JobService = Depends(get_job_service),
    data_processing_service: DataProcessingService = Depends(get_data_processing_service)
):
    try:
        selections = payload.selections
        train_files = selections.get('train_files', [])
        test_files = selections.get('test_files', [])
        data_source = selections.get('data_source', 'Unknown')
        
        print(f"Processing data for new job. Source: {data_source}")
        print(f"Train files: {train_files}")
        print(f"Test files: {test_files}")
        
        # Create a new job first to get a job_id and folder
        job_id, job_folder = job_service.create_new_job(selections)
        if not job_id or not job_folder:
            raise HTTPException(status_code=500, detail="Failed to create new job.")
        
        # Process the data files
        success = data_processing_service.process_data_files(
            job_id=job_id,
            job_folder=job_folder,
            train_files=train_files,
            test_files=test_files,
            data_source=data_source
        )
        
        if not success:
            # If data processing fails, delete the job and notify the client
            job_service.delete_job(job_id)
            raise HTTPException(status_code=500, detail="Failed to process data files.")
        
        # Update job status to indicate data has been processed
        job_service.job_manager.update_job_details(job_id, {
            "data_processed": True, 
            "data_source": data_source
        })
        
        # Get the updated job info to return
        job_info = job_service.get_job_by_id(job_id)
        return job_info
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
    # We would need to implement a proper shutdown mechanism
    # For now, just return a message that we received the request
    return {"message": "Shutdown signal received. Server will stop."}

# --- Server Startup ---
def start_server(host="0.0.0.0", port=8001):
    """Start the FastAPI server with the specified host and port."""
    print(f"Starting VEstim backend server on {host}:{port}...")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    # If this file is run directly, start the server
    print("Starting VEstim backend server...")
    start_server()