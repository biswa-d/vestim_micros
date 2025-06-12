import time
import threading
from typing import List, Dict, Any
import logging
from PyQt5.QtCore import QObject, pyqtSignal, QTimer
from vestim.gui.src.api_gateway import APIGateway


class DashboardManager(QObject):
    """
    Frontend manager for the dashboard that handles communication with the backend
    and manages job status updates for the GUI.
    """
    
    # Signals for GUI updates
    jobs_updated = pyqtSignal(list)  # Emitted when job list is updated
    job_status_changed = pyqtSignal(str, dict)  # Emitted when specific job status changes
    server_error = pyqtSignal(str)  # Emitted on server communication errors
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.api_gateway = APIGateway()
        self.logger = logging.getLogger(__name__)
        self.current_jobs = {}  # job_id -> job_data
        
        # Auto-refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_jobs)
        self.refresh_timer.start(5000)  # Refresh every 5 seconds
        
        # Track if we're monitoring
        self.monitoring_active = True

    def refresh_jobs(self):
        """Refresh the job list from the backend."""
        try:
            jobs = self.api_gateway.get("jobs")
            if jobs is not None:
                self.current_jobs = {job['job_id']: job for job in jobs}
                self.jobs_updated.emit(jobs)
            else:
                self.server_error.emit("Failed to fetch jobs from server")
                
        except Exception as e:
            self.logger.error(f"Error communicating with server: {str(e)}")
            self.server_error.emit(f"Error communicating with server: {str(e)}")

    def get_job_details(self, job_id: str) -> Dict[str, Any]:
        """Get detailed information for a specific job."""
        try:
            job_details = self.api_gateway.get(f"jobs/{job_id}")
            if job_details:
                # Update local cache
                self.current_jobs[job_id] = job_details
                return job_details
            else:
                return {}
        except Exception as e:
            self.logger.error(f"Error fetching job details: {str(e)}")
            self.server_error.emit(f"Error fetching job details: {str(e)}")
            return {}

    def create_new_job(self, selections: Dict[str, Any]) -> bool:
        """Create a new job via the backend."""
        try:
            result = self.api_gateway.post("jobs", json={"selections": selections})
            if result:
                # Refresh job list after creating new job
                self.refresh_jobs()
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error creating job: {str(e)}")
            self.server_error.emit(f"Error creating job: {str(e)}")
            return False

    def stop_job(self, job_id: str) -> bool:
        """Stop a running job."""
        try:
            result = self.api_gateway.post(f"jobs/{job_id}/stop")
            if result:
                # Refresh job list after stopping job
                self.refresh_jobs()
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error stopping job: {str(e)}")
            self.server_error.emit(f"Error stopping job: {str(e)}")
            return False

    def delete_job(self, job_id: str) -> bool:
        """Delete a job."""
        try:
            result = self.api_gateway.delete(f"jobs/{job_id}")
            if result:
                # Remove from local cache
                if job_id in self.current_jobs:
                    del self.current_jobs[job_id]
                # Refresh job list after deleting job
                self.refresh_jobs()
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error deleting job: {str(e)}")
            self.server_error.emit(f"Error deleting job: {str(e)}")
            return False
            return None
        except Exception as e:
            self.logger.error(f"Error killing job {job_id}: {e}")
            return None

    def stop_server(self) -> bool:
        """Stop the backend server."""
        try:
            result = self.api_gateway.post("server/shutdown")
            return result is not None
        except Exception as e:
            self.logger.error(f"Error stopping server: {str(e)}")
            self.server_error.emit(f"Error stopping server: {str(e)}")
            return False

    def get_training_status(self, job_id: str) -> Dict[str, Any]:
        """Get training status for a specific job."""
        try:
            status = self.api_gateway.get(f"jobs/{job_id}/status")
            return status or {}
        except Exception as e:
            self.logger.error(f"Error fetching training status: {str(e)}")
            self.server_error.emit(f"Error fetching training status: {str(e)}")
            return {}

    def start_training(self, job_id: str, task_data: Dict[str, Any] = None) -> bool:
        """Start training for a specific job."""
        try:
            payload = {"task_id": f"task_{job_id}"}
            if task_data:
                payload.update(task_data)
            
            result = self.api_gateway.post(f"jobs/{job_id}/train", json=payload)
            return result is not None
        except Exception as e:
            self.logger.error(f"Error starting training: {str(e)}")
            self.server_error.emit(f"Error starting training: {str(e)}")
            return False

    def start_testing(self, job_id: str) -> bool:
        """Start testing for a specific job."""
        try:
            result = self.api_gateway.post(f"jobs/{job_id}/test")
            return result is not None
        except Exception as e:
            self.logger.error(f"Error starting testing: {str(e)}")
            self.server_error.emit(f"Error starting testing: {str(e)}")
            return False

    def get_cached_job(self, job_id: str) -> Dict[str, Any]:
        """Get job from local cache."""
        return self.current_jobs.get(job_id, {})

    def get_all_cached_jobs(self) -> List[Dict[str, Any]]:
        """Get all jobs from local cache."""
        return list(self.current_jobs.values())

    def is_server_running(self) -> bool:
        """Check if the backend server is running."""
        try:
            status = self.api_gateway.get("server/status")
            return status is not None and status.get("status") == "online"
        except:
            return False

    def shutdown(self):
        """Shutdown the dashboard manager."""
        self.monitoring_active = False
        if hasattr(self, 'refresh_timer'):
            self.refresh_timer.stop()

    # Legacy methods for compatibility
    def kill_job(self, job_id: str) -> bool:
        """Legacy method - use stop_job instead."""
        return self.stop_job(job_id)

    def resume_job(self, job_id: str) -> bool:
        """Resume a stopped or failed job."""
        try:
            result = self.api_gateway.post(f"jobs/{job_id}/resume")
            if result:
                self.refresh_jobs()  # Refresh the job list
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error resuming job: {str(e)}")
            self.server_error.emit(f"Error resuming job: {str(e)}")
            return False
    
    def get_resumable_jobs(self) -> List[Dict[str, Any]]:
        """Get a list of jobs that can be resumed."""
        try:
            result = self.api_gateway.get("jobs/resumable")
            return result if result else []
        except Exception as e:
            self.logger.error(f"Error getting resumable jobs: {str(e)}")
            self.server_error.emit(f"Error getting resumable jobs: {str(e)}")
            return []
    
    def can_resume_job(self, job_id: str) -> bool:
        """Check if a job can be resumed."""
        try:
            result = self.api_gateway.get(f"jobs/{job_id}/can_resume")
            return result.get('can_resume', False) if result else False
        except Exception as e:
            self.logger.error(f"Error checking if job can be resumed: {str(e)}")
            return False