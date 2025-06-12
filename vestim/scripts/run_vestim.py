import sys
import os
import time
import subprocess
from PyQt5.QtWidgets import QApplication
import psutil

# Add the project root to the Python path to allow for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from vestim.gui.src.job_dashboard_gui_qt import JobDashboard

def start_server():
    """Starts the FastAPI server in a separate process."""
    # We use subprocess.Popen to run the server in a way that is detached
    # from the main GUI process.
    command = [sys.executable, "-m", "vestim.backend.src.main"]
    
    # Using DETACHED_PROCESS flag on Windows to ensure the server process
    # doesn't get terminated if the parent (this script) is closed unexpectedly.
    # For Linux/macOS, a simple Popen is usually sufficient.
    creationflags = 0
    if sys.platform == "win32":
        #creationflags = subprocess.DETACHED_PROCESS
        pass # Ensure there's an indented block if the above is commented

    # Ensure this block is correctly indented
    server_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=creationflags)
    print(f"Started server with PID: {server_process.pid}")
    # stdout, stderr = server_process.communicate() # This line blocks GUI
    # print(f"Server stdout:\n{stdout.decode()}")
    # print(f"Server stderr:\n{stderr.decode()}")
    
    return server_process

def main():
    """Main function to start the server and launch the GUI."""
    
    server_process = None
    try:
        # Start the backend server
        server_process = start_server()
        
        # Give the server a moment to initialize
        print("Waiting for server to start...")
        time.sleep(3)

        # Launch the GUI
        app = QApplication(sys.argv)
        dashboard = JobDashboard()
        dashboard.show()
        
        # Start the Qt event loop
        app.exec_()

    finally:
        # This block will run when the GUI is closed
        if server_process:
            print(f"Shutting down server process (PID: {server_process.pid})...")
            try:
                # Use psutil to gracefully terminate the process and its children
                parent = psutil.Process(server_process.pid)
                for child in parent.children(recursive=True):
                    child.terminate()
                parent.terminate()
                
                # Wait for the process to terminate
                server_process.wait(timeout=5)
                print("Server process terminated.")
            except psutil.NoSuchProcess:
                print("Server process was already terminated.")
            except psutil.TimeoutExpired:
                print("Server process did not terminate in time, killing it.")
                server_process.kill()


if __name__ == "__main__":
    main()