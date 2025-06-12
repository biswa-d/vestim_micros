import subprocess
import sys
import time
import requests
from PyQt5.QtWidgets import QApplication, QMessageBox

def is_server_running(host="127.0.0.1", port=8001):
    """Check if the server is running and responsive."""
    try:
        response = requests.get(f"http://{host}:{port}/server/status", timeout=1)
        return response.status_code == 200
    except requests.RequestException:
        return False

def main():
    """
    Launch the VEstim server and GUI.
    This script ensures the server is running before launching the GUI.
    """
    host = "127.0.0.1"
    port = 8001

    if not is_server_running(host, port):
        # Start the server in the background
        print("Starting VEstim server...")
        server_command = [sys.executable, "-m", "vestim.scripts.run_server", f"--host={host}", f"--port={port}"]
        subprocess.Popen(server_command)

        # Wait for the server to be ready
        print("Waiting for server to initialize...")
        max_wait_time = 30  # seconds
        start_time = time.time()
        
        while not is_server_running(host, port):
            if time.time() - start_time > max_wait_time:
                print("Server did not start within the expected time. Aborting.")
                sys.exit(1)
            time.sleep(1)
            print("...")
    else:
        print("Server is already running.")

    print("Server is running. Launching GUI...")
    
    # Launch the GUI
    from vestim.gui.src.job_dashboard_gui_qt import JobDashboard
    
    app = QApplication(sys.argv)
    dashboard = JobDashboard()
    dashboard.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
