import argparse
import os
import sys
import subprocess
import time
import requests
from PyQt5.QtWidgets import QApplication, QMessageBox
from vestim.gui.src.job_dashboard_gui_qt import JobDashboard

def is_server_running(host="127.0.0.1", port=8001):
    """Check if the server is running and responsive."""
    try:
        response = requests.get(f"http://{host}:{port}/server/status", timeout=2)
        return response.status_code == 200 and response.json().get("status") == "online"
    except requests.RequestException:
        return False

def wait_for_server(host="127.0.0.1", port=8001, max_wait=30):
    """Wait for the server to become available."""
    print("Waiting for server to start...")
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        if is_server_running(host, port):
            print("Server is ready!")
            return True
        time.sleep(1)
        print(".", end="", flush=True)
    
    print(f"\nServer did not start within {max_wait} seconds.")
    return False

def start_server_background(host="127.0.0.1", port=8001):
    """Start the server in the background."""
    command = [sys.executable, "-m", "vestim.scripts.run_server", f"--host={host}", f"--port={port}"]
    
    try:
        if sys.platform == "win32":
            # On Windows, use DETACHED_PROCESS to run server independently
            subprocess.Popen(command, creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
        else:
            # On Unix-like systems
            subprocess.Popen(command, start_new_session=True)
            
        return True
    except Exception as e:
        print(f"Failed to start server: {e}")
        return False
        
        # Wait for the server to be ready
        if not check_server_status(f"http://{args.host}:{args.port}"):
            print("Failed to start the backend server. Exiting.")
            return 1def main():
    """
    Launch the VEstim application with dashboard-first architecture.
    This script ensures the server is running before launching the GUI.
    """
    host = "127.0.0.1"
    port = 8001

    print("Starting VEstim...")
    
    # Check if server is already running
    if not is_server_running(host, port):
        print("Server not running. Starting backend server...")
        if not start_server_background(host, port):
            print("Failed to start server. Exiting.")
            return 1
            
        # Wait for server to be ready
        if not wait_for_server(host, port):
            print("Server failed to start properly. Exiting.")
            return 1
    else:
        print("Server is already running.")

    print("Launching dashboard...")
    
    # Launch the GUI (Dashboard)
    app = QApplication(sys.argv)
    
    try:
        dashboard = JobDashboard()
        dashboard.show()
        return app.exec_()
    except Exception as e:
        QMessageBox.critical(None, "Error", f"Failed to start dashboard: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())