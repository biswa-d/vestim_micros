#!/usr/bin/env python3

import requests
import time

def is_server_running(host="127.0.0.1", port=8001):
    """Check if the server is running and responsive."""
    try:
        response = requests.get(f"http://{host}:{port}/server/status", timeout=1)
        return response.status_code == 200
    except requests.RequestException:
        return False

def test_server_endpoints():
    """Test server endpoints."""
    base_url = "http://127.0.0.1:8001"
    
    print("Testing server endpoints...")
    
    # Test server status
    try:
        response = requests.get(f"{base_url}/server/status")
        print(f"Server status: {response.json()}")
    except Exception as e:
        print(f"Error testing server status: {e}")
        return False
    
    # Test jobs endpoint
    try:
        response = requests.get(f"{base_url}/jobs")
        print(f"Jobs list: {response.json()}")
    except Exception as e:
        print(f"Error testing jobs endpoint: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Testing VEstim server functionality...")
    
    if is_server_running():
        print("✓ Server is running")
        if test_server_endpoints():
            print("✓ All endpoints working")
        else:
            print("✗ Some endpoints failed")
    else:
        print("✗ Server is not running")
        print("Please start the server first with: python -m vestim.scripts.run_server")
