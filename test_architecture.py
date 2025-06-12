#!/usr/bin/env python3

import subprocess
import sys
import time
import requests
import threading
import os

def start_server():
    """Start the VEstim server."""
    cmd = [sys.executable, "-m", "vestim.scripts.run_server", "--host=127.0.0.1", "--port=8001"]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def is_server_running(host="127.0.0.1", port=8001):
    """Check if the server is running and responsive."""
    try:
        response = requests.get(f"http://{host}:{port}/server/status", timeout=1)
        return response.status_code == 200
    except requests.RequestException:
        return False

def test_architecture():
    """Test the complete VEstim architecture."""
    print("=== Testing VEstim Architecture ===")
    
    # Check if server is already running
    if is_server_running():
        print("✓ Server is already running")
        server_process = None
    else:
        print("Starting server...")
        server_process = start_server()
        
        # Wait for server to start
        max_wait = 30
        start_time = time.time()
        while not is_server_running():
            if time.time() - start_time > max_wait:
                print("✗ Server failed to start within 30 seconds")
                if server_process:
                    server_process.terminate()
                return False
            time.sleep(1)
            print(".", end="", flush=True)
        print()
        print("✓ Server started successfully")
    
    # Test endpoints
    base_url = "http://127.0.0.1:8001"
    
    try:
        # Test server status
        response = requests.get(f"{base_url}/server/status")
        if response.status_code == 200:
            print("✓ Server status endpoint working")
        else:
            print(f"✗ Server status endpoint failed: {response.status_code}")
            
        # Test jobs endpoint
        response = requests.get(f"{base_url}/jobs")
        if response.status_code == 200:
            jobs = response.json()
            print(f"✓ Jobs endpoint working - Found {len(jobs.get('jobs', []))} jobs")
        else:
            print(f"✗ Jobs endpoint failed: {response.status_code}")
            
    except Exception as e:
        print(f"✗ Error testing endpoints: {e}")
        
    # Cleanup
    if server_process:
        print("Stopping server...")
        server_process.terminate()
        server_process.wait()
        print("✓ Server stopped")
    
    print("=== Test Complete ===")

if __name__ == "__main__":
    os.chdir("c:\\Users\\dehuryb\\OneDrive - McMaster University\\Models\\ML_LiB_Models\\vestim_micros")
    test_architecture()
