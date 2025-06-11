import os
import psutil
import appdirs
import time
import requests
import subprocess
import sys

def is_server_running(url="http://127.0.0.1:8001/server/status"):
    """
    Checks if the server is running by making a request to its status endpoint.
    """
    try:
        response = requests.get(url, timeout=1)
        return response.status_code == 200
    except requests.ConnectionError:
        return False

def start_server():
    """
    Starts the server as a background process.
    """
    try:
        # Use subprocess.Popen to start the server in a new process
        # that won't block the current script.
        subprocess.Popen([sys.executable, "-m", "vestim.scripts.run_server"])
        time.sleep(2) # Give the server a moment to start
    except Exception as e:
        print(f"Failed to start server: {e}")
