#!/usr/bin/env python
"""
Test script to check if the APIGateway can connect to the server and retrieve jobs.
"""

import sys
import os
import requests
import time

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_server_connection():
    """Test if the server is available."""
    try:
        # Try the health endpoint
        print("Testing server connection to health endpoint...")
        response = requests.get("http://127.0.0.1:8001/health", timeout=2)
        if response.status_code == 200:
            print("✅ Success: Server health endpoint is accessible")
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"❌ Error: Server health endpoint returned status code {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Error connecting to health endpoint: {e}")
    
    # Try the root endpoint as fallback
    try:
        print("\nTesting server connection to root endpoint...")
        response = requests.get("http://127.0.0.1:8001/", timeout=2)
        if response.status_code == 200:
            print("✅ Success: Server root endpoint is accessible")
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"❌ Error: Server root endpoint returned status code {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Error connecting to root endpoint: {e}")
    
    return False

def test_get_all_jobs():
    """Test if we can retrieve all jobs from the server."""
    try:
        print("\nTesting get all jobs endpoint...")
        response = requests.get("http://127.0.0.1:8001/jobs", timeout=5)
        if response.status_code == 200:
            jobs = response.json()
            print(f"✅ Success: Retrieved {len(jobs)} jobs")
            for i, job in enumerate(jobs):
                print(f"Job {i+1}: ID={job.get('job_id')}, Status={job.get('status')}")
            return True
        else:
            print(f"❌ Error: Failed to get jobs, status code {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Error getting jobs: {e}")
    
    return False

def test_create_job():
    """Test if we can create a new job."""
    try:
        print("\nTesting create job endpoint...")
        payload = {
            "selections": {
                "test_key": "test_value",
                "timestamp": time.time()
            }
        }
        response = requests.post("http://127.0.0.1:8001/jobs", json=payload, timeout=5)
        if response.status_code == 200:
            job = response.json()
            print(f"✅ Success: Created job with ID {job.get('job_id')}")
            print(f"Job details: {job}")
            return job.get('job_id')
        else:
            print(f"❌ Error: Failed to create job, status code {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Error creating job: {e}")
    
    return None

if __name__ == "__main__":
    print("Starting API Gateway tests...")
    
    # Test server connection
    if not test_server_connection():
        print("\n❌ Server connection tests failed. Exiting.")
        sys.exit(1)
    
    # Test getting all jobs
    test_get_all_jobs()
    
    # Test creating a job
    job_id = test_create_job()
    if job_id:
        print(f"\n✅ All tests completed successfully.")
    else:
        print(f"\n❌ Some tests failed. Check the output above for details.")
