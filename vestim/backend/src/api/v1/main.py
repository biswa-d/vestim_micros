# ---------------------------------------------------------------------------------
# Author: Your Name
# Date: 2025-06-11
# Version: 1.0.0
# Description: The main API gateway for the VEstim backend service.
# ---------------------------------------------------------------------------------

import sys
import os
from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any

# Add the project root to the Python path to allow for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

from vestim.backend.src.managers.job_manager import JobManager

# Create the FastAPI application instance
app = APIRouter()

# --- Dependency Injection ---

def get_job_manager() -> JobManager:
    """
    Dependency injector that provides the singleton instance of the JobManager.
    """
    return JobManager()

# --- API Endpoints ---

@app.post("/jobs", response_model=Dict[str, str], status_code=201)
def create_job(jm: JobManager = Depends(get_job_manager)):
    """
    Creates a new job and returns its ID.
    """
    job_id = jm.create_new_job()
    if not job_id:
        raise HTTPException(status_code=500, detail="Failed to create a new job.")
    return {"job_id": job_id}

@app.get("/jobs", response_model=List[Dict[str, Any]])
def get_all_jobs_summary(jm: JobManager = Depends(get_job_manager)):
    """
    Gets a high-level summary of all current jobs.
    """
    return jm.get_job_summary()

@app.get("/server/status", response_model=Dict[str, Any])
def get_server_status(jm: JobManager = Depends(get_job_manager)):
    """
    Returns the current status of the server, including the number of active jobs.
    """
    active_jobs = len(jm.get_job_summary())
    return {"status": "running", "active_jobs": active_jobs}
@app.delete("/jobs/{job_id}", status_code=204)
def kill_job_endpoint(job_id: str, jm: JobManager = Depends(get_job_manager)):
    """
    Kills a job and its active task.
    """
    if not jm.kill_job(job_id):
        raise HTTPException(status_code=404, detail=f"Job with ID {job_id} not found or could not be killed.")
    return