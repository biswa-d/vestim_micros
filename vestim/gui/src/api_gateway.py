import requests
import json
import logging
import time
from PyQt5.QtCore import QObject, pyqtSignal

class APIGateway(QObject):
    """
    A centralized gateway for all API communications from the GUI.
    This class handles HTTP requests to the backend server, providing a clean
    and consistent interface for the rest of the application.
    """
    # Define signals that can be connected to GUI components
    connectionError = pyqtSignal(str)
    requestCompleted = pyqtSignal(dict)
    requestListCompleted = pyqtSignal(list)  # New signal for list responses
    
    def __init__(self, base_url="http://127.0.0.1:8001"):
        super().__init__()
        self.base_url = base_url
        self.logger = logging.getLogger(__name__)
        self.timeout = 10  # Default timeout in seconds
        self.max_retries = 3  # Number of retries for transient errors

    def _make_request(self, method, endpoint, **kwargs):
        """Internal method to make HTTP requests with retries and better error handling."""
        url = f"{self.base_url}/{endpoint}"
        self.logger.info(f"Making {method.upper()} request to {url} with kwargs: {kwargs}")
        
        # Add timeout if not specified
        if 'timeout' not in kwargs:
            kwargs['timeout'] = self.timeout
        
        # Implement retry logic for transient errors
        retries = 0
        last_exception = None
        
        while retries < self.max_retries:
            try:
                response = requests.request(method, url, **kwargs)
                  # Handle HTTP errors
                if response.status_code >= 400:
                    error_detail = f"HTTP Error {response.status_code}"
                    try:
                        error_json = response.json()
                        if 'detail' in error_json:
                            error_detail = f"{error_detail}: {error_json['detail']}"
                    except:
                        try:
                            error_detail = f"{error_detail}: {response.text}"
                        except:
                            pass
                    
                    self.logger.error(f"HTTP Error for {method.upper()} {url}: {error_detail}")
                    
                    # Emit the connection error signal
                    self.connectionError.emit(error_detail)
                    
                    # Return detailed error information instead of raising an exception
                    # This allows the caller to handle the error gracefully
                    return {"status": "error", "message": error_detail, "detail": error_detail, "http_status": response.status_code}
                
                # Success case - parse JSON
                try:
                    result = response.json()
                    # Emit the appropriate signal based on response type
                    if isinstance(result, list):
                        self.requestListCompleted.emit(result)
                    else:
                        self.requestCompleted.emit(result)
                    return result
                except json.JSONDecodeError:
                    self.logger.error(f"Failed to decode JSON from response for {method.upper()} {url}")
                    return {"status": "error", "message": "Invalid JSON response from server"}
                
            except requests.exceptions.Timeout:
                self.logger.warning(f"Timeout for {method.upper()} {url} (attempt {retries+1}/{self.max_retries})")
                retries += 1
                last_exception = f"Request timed out after {self.timeout} seconds"
                time.sleep(1)  # Wait before retrying
                
            except requests.exceptions.ConnectionError:
                self.logger.warning(f"Connection error for {method.upper()} {url} (attempt {retries+1}/{self.max_retries})")
                retries += 1
                last_exception = "Could not connect to server"
                time.sleep(1)  # Wait before retrying
                
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Request failed for {method.upper()} {url}: {e}")
                last_exception = str(e)
                break  # Don't retry other types of errors
          # If we get here, all retries failed or a non-retryable error occurred
        error_message = last_exception or "Unknown error occurred"
        self.connectionError.emit(error_message)
        return {"status": "error", "message": error_message}
        
    def get(self, endpoint, params=None):
        """Sends a GET request."""
        return self._make_request("get", endpoint, params=params)
        
    def post(self, endpoint, data=None, json=None, timeout=None):
        """Sends a POST request."""
        kwargs = {'data': data, 'json': json}
        if timeout:
            kwargs['timeout'] = timeout
        return self._make_request("post", endpoint, **kwargs)

    def delete(self, endpoint):
        """Sends a DELETE request."""
        return self._make_request("delete", endpoint)

    def is_server_available(self):
        """Check if the server is available by making a simple health check request."""
        try:
            # Try the dedicated health endpoint first
            try:
                response = requests.get(f"{self.base_url}/health", timeout=2)
                if response.status_code == 200:
                    self.logger.info("Server health check successful")
                    return True
            except:
                # If health endpoint fails, try the root endpoint
                self.logger.warning("Health endpoint check failed, trying root endpoint")
                try:
                    response = requests.get(f"{self.base_url}/", timeout=2)
                    if response.status_code == 200:
                        self.logger.info("Server root endpoint check successful")
                        return True
                except:
                    pass
              # If we get here, both checks failed
            self.logger.error("Server is not available - all endpoint checks failed")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during server availability check: {e}")
            return False

    # --- Job-specific methods ---
    def get_all_jobs(self):
        """Get all jobs with proper error handling."""
        try:
            result = self.get("jobs")
            if isinstance(result, dict) and result.get("status") == "error":
                self.logger.error(f"Error getting jobs: {result.get('message', 'Unknown error')}")
                return []  # Return empty list on error
            return result if isinstance(result, list) else []
        except Exception as e:
            self.logger.error(f"Failed to get all jobs: {e}")
            self.connectionError.emit(f"Failed to retrieve jobs: {str(e)}")
            return []  # Return empty list instead of None to avoid NoneType errors

    def get_job(self, job_id):
        """Get a specific job with error handling."""
        try:
            return self.get(f"jobs/{job_id}")
        except Exception as e:
            self.logger.error(f"Failed to get job {job_id}: {e}")
            return None

    def create_job(self, selections):
        """Create a new job."""
        return self.post("jobs", json={"selections": selections})

    def process_and_create_job(self, selections):
        """Process data files and create a new job."""
        return self.post("jobs/process-and-create", json={"selections": selections})

    def start_job(self, job_id, task_info):
        """Start a job with the given task info."""
        return self.post(f"jobs/{job_id}/start", json={"task_info": task_info})

    def save_hyperparameters(self, job_id, params):
        """Saves the hyperparameters for a specific job."""
        return self.post(f"jobs/{job_id}/hyperparameters", json={"hyperparameters": params})

    def stop_job(self, job_id):
        """Stop a running job."""
        return self.post(f"jobs/{job_id}/stop")

    def delete_job(self, job_id):
        """Delete a job."""
        return self.delete(f"jobs/{job_id}")

    def ensure_job_exists(self, job_id):
        """Ensure a job exists in the system. Creates it if missing (if folder exists)."""
        try:
            return self.post(f"jobs/{job_id}/ensure")
        except Exception as e:
            self.logger.error(f"Failed to ensure job {job_id} exists: {e}")
            return {"status": "error", "message": str(e)}

    # --- Server methods ---
    def shutdown_server(self):
        """Send a shutdown signal to the server."""
        try:
            return self.post("server/shutdown")
        except requests.exceptions.RequestException:
            # This is expected as the server will shut down before responding
            return {"message": "Shutdown signal sent."}
    
    # --- Training task-specific methods ---
    def start_training_task(self, job_id, task_id):
        """Start a specific training task for a job."""
        try:
            return self.post(f"jobs/{job_id}/tasks/{task_id}/start")
        except Exception as e:
            self.logger.error(f"Failed to start training task {task_id} for job {job_id}: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_training_task_status(self, job_id, task_id):
        """Get the status and progress of a specific training task."""
        try:
            return self.get(f"jobs/{job_id}/tasks/{task_id}/status")
        except Exception as e:
            self.logger.error(f"Failed to get status for training task {task_id} of job {job_id}: {e}")
            return {"status": "error", "message": str(e)}
    
    def setup_training(self, job_id, timeout=60):
        """Set up the training tasks for a job."""
        try:
            return self.post(f"jobs/{job_id}/setup-training", timeout=timeout)
        except Exception as e:
            self.logger.error(f"Failed to set up training tasks for job {job_id}: {e}")
            return {"status": "error", "message": str(e)}