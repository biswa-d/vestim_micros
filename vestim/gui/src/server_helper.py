#!/usr/bin/env python
# Utility to help manage socket reuse for the server
import socket
import subprocess
import sys
import time
import os
import logging

HOST = "127.0.0.1"
PORT = 8001 # Corrected port

def get_pid_file():
    """Returns the path to the PID file, consistent with run_server.py."""
    return os.path.join(os.path.expanduser("~"), ".vestim", "server.pid")

def is_server_running():
    """
    Checks if the server is running by attempting to create a socket connection.
    This is more reliable than a PID check, as the process could be running but not listening.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        try:
            s.connect((HOST, PORT))
            return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            return False

def start_server():
    """Starts the Uvicorn server in a detached process."""
    try:
        # Use CREATE_NEW_PROCESS_GROUP to run the server completely detached
        server_process = subprocess.Popen(
            [sys.executable, "-m", "vestim.scripts.run_server"],
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )
        time.sleep(3) # Give the server a moment to start and write its PID file
        return server_process
    except Exception as e:
        logging.getLogger("vestim.server_helper").error(f"Failed to start server: {e}")
        return None