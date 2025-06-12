import requests
import time
import json

# --- Configuration ---
BASE_URL = "http://127.0.0.1:8001"

def print_status(message, is_error=False):
    """Prints a formatted status message."""
    prefix = "[ERROR]" if is_error else "[INFO]"
    print(f"{prefix} {message}")

def create_job():
    """Creates a new job."""
    print_status("Attempting to create a new job...")
    try:
        payload = {
            "selections": {
                "model_type": "LSTM",
                "dataset": "sample_data.csv"
            }
        }
        response = requests.post(f"{BASE_URL}/jobs", json=payload)
        response.raise_for_status()
        job = response.json()
        print_status(f"Successfully created job: {job['job_id']}")
        return job['job_id']
    except requests.exceptions.RequestException as e:
        print_status(f"Failed to create job: {e}", is_error=True)
        return None

def start_job(job_id):
    """Starts a job."""
    print_status(f"Attempting to start job: {job_id}...")
    try:
        payload = {
            "task_info": {
                "hyperparams": {
                    "MAX_EPOCHS": 15,
                    "learning_rate": 0.001
                },
                "DEVICE_SELECTION": "cpu"
            }
        }
        response = requests.post(f"{BASE_URL}/jobs/{job_id}/start", json=payload)
        response.raise_for_status()
        print_status(f"Successfully sent start signal for job: {job_id}")
        return True
    except requests.exceptions.RequestException as e:
        print_status(f"Failed to start job {job_id}: {e}", is_error=True)
        return False

def monitor_job(job_id):
    """Monitors the status of a job until it completes or fails."""
    print_status(f"Monitoring job {job_id}...")
    while True:
        try:
            response = requests.get(f"{BASE_URL}/jobs/{job_id}/status")
            response.raise_for_status()
            status_data = response.json()
            
            status = status_data.get('status')
            details = status_data.get('details', {})
            message = details.get('message', 'No message.')
            
            print_status(f"Job Status: {status} | Message: {message}")
            
            if details.get('history'):
                print(json.dumps(details['history'][-1], indent=2))


            if status in ['complete', 'error', 'stopped']:
                print_status(f"Job {job_id} finished with status: {status}")
                break
                
            time.sleep(2)
            
        except requests.exceptions.RequestException as e:
            print_status(f"Failed to get status for job {job_id}: {e}", is_error=True)
            break

def stop_job(job_id):
    """Stops a running job."""
    print_status(f"Attempting to stop job: {job_id}...")
    try:
        response = requests.post(f"{BASE_URL}/jobs/{job_id}/stop")
        response.raise_for_status()
        print_status(f"Successfully sent stop signal for job: {job_id}")
    except requests.exceptions.RequestException as e:
        print_status(f"Failed to stop job {job_id}: {e}", is_error=True)

def delete_job(job_id):
    """Deletes a job."""
    print_status(f"Attempting to delete job: {job_id}...")
    try:
        response = requests.delete(f"{BASE_URL}/jobs/{job_id}")
        response.raise_for_status()
        print_status(f"Successfully deleted job: {job_id}")
    except requests.exceptions.RequestException as e:
        print_status(f"Failed to delete job {job_id}: {e}", is_error=True)

def main():
    """Main function to run the test."""
    
    # --- Test Case 1: Create, Start, and Monitor a Job to Completion ---
    print_status("--- Running Test Case 1: Full Job Lifecycle ---")
    job_id_1 = create_job()
    if job_id_1:
        if start_job(job_id_1):
            monitor_job(job_id_1)
        delete_job(job_id_1)

    print("\n" + "="*50 + "\n")

    # --- Test Case 2: Create, Start, and Stop a Job ---
    print_status("--- Running Test Case 2: Stop a Running Job ---")
    job_id_2 = create_job()
    if job_id_2:
        if start_job(job_id_2):
            time.sleep(5)  # Let it run for a few seconds
            stop_job(job_id_2)
            time.sleep(2) # Give time for status to update
            monitor_job(job_id_2) # Check final status
        delete_job(job_id_2)


if __name__ == "__main__":
    main()