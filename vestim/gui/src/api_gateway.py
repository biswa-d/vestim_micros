import requests
import json
import logging

class APIGateway:
    """
    A centralized gateway for all API communications from the GUI.
    This class handles HTTP requests to the backend server, providing a clean
    and consistent interface for the rest of the application.
    """
    def __init__(self, base_url="http://127.0.0.1:8001"):
        self.base_url = base_url
        self.logger = logging.getLogger(__name__)

    def _make_request(self, method, endpoint, **kwargs):
        """Internal method to make HTTP requests."""
        url = f"{self.base_url}/{endpoint}"
        self.logger.info(f"Making {method.upper()} request to {url} with kwargs: {kwargs}")
        try:
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            self.logger.error(f"HTTP Error for {method.upper()} {url}: {e.response.text}")
            raise  # Re-raise the exception to be handled by the caller
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed for {method.upper()} {url}: {e}")
            raise
        except json.JSONDecodeError:
            self.logger.error(f"Failed to decode JSON from response for {method.upper()} {url}")
            return None # Or raise a custom exception

    def get(self, endpoint, params=None):
        """Sends a GET request."""
        return self._make_request("get", endpoint, params=params)

    def post(self, endpoint, data=None, json=None):
        """Sends a POST request."""
        return self._make_request("post", endpoint, data=data, json=json)

    def delete(self, endpoint):
        """Sends a DELETE request."""
        return self._make_request("delete", endpoint)

    # --- Job-specific methods ---
    def get_all_jobs(self):
        return self.get("jobs")

    def get_job(self, job_id):
        return self.get(f"jobs/{job_id}")

    def create_job(self, selections):
        return self.post("jobs", json={"selections": selections})

    def start_job(self, job_id, task_info):
        return self.post(f"jobs/{job_id}/start", json={"task_info": task_info})

    def save_hyperparameters(self, job_id, params):
        """Saves the hyperparameters for a specific job."""
        return self.post(f"jobs/{job_id}/hyperparameters", json={"hyperparameters": params})

    def stop_job(self, job_id):
        return self.post(f"jobs/{job_id}/stop")

    def delete_job(self, job_id):
        return self.delete(f"jobs/{job_id}")

    # --- Server methods ---
    def shutdown_server(self):
        try:
            return self.post("server/shutdown")
        except requests.exceptions.RequestException:
            # This is expected as the server will shut down before responding
            return {"message": "Shutdown signal sent."}