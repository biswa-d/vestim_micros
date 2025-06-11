import argparse
import os
import sys
import subprocess
import time
import requests
from PyQt5.QtWidgets import QApplication
from vestim.gui.src.job_dashboard_gui_qt import JobDashboard

def check_server_status(url="http://127.0.0.1:8001", retries=5, delay=1):
    """Checks if the backend server is ready to accept connections."""
    for i in range(retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print("Backend server is ready.")
                return True
        except requests.ConnectionError:
            pass
        print(f"Server not ready yet. Retrying in {delay} second(s)...")
        time.sleep(delay)
    print("Could not connect to the backend server.")
    return False

def start_server(args):
    """Starts the backend server."""
    # Use the same arguments passed to this script
    command = [sys.executable, "-m", "vestim.scripts.run_server"]
    
    # Add any server-specific arguments
    if args.host:
        command.extend(["--host", args.host])
    if args.port:
        command.extend(["--port", str(args.port)])
    if args.reload:
        command.append("--reload")
    
    # Determine how to run the server based on the mode
    if args.mode == "server":
        # For server-only mode, just run the command directly (blocking)
        print("Starting server in foreground mode...")
        return subprocess.call(command)
    else:
        # For GUI mode, start server as a background process
        print("Starting server in background mode...")
        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.DETACHED_PROCESS
        
        subprocess.Popen(command, creationflags=creationflags, close_fds=True)
        
        # Wait for the server to be ready
        if not check_server_status(f"http://{args.host}:{args.port}"):
            print("Failed to start the backend server. Exiting.")
            return 1
    
    return 0

def start_gui(args):
    """Starts the GUI."""
    # Start the GUI
    app = QApplication(sys.argv)
    gui = JobDashboard()
    gui.show()
    return app.exec_()

def main():
    """
    Main entry point for the VEstim application.
    """
    parser = argparse.ArgumentParser(description="VEstim - Machine Learning Model Training Tool")
    parser.add_argument("--mode", choices=["all", "server", "gui"], default="all",
                        help="Run mode: 'all' to run both server and GUI, 'server' for server only, 'gui' for GUI only")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind the server to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    # Mode handling
    if args.mode in ["all", "server"]:
        server_result = start_server(args)
        if server_result != 0 and args.mode == "server":
            return server_result
    
    if args.mode in ["all", "gui"]:
        # Check if the server is already running before starting the GUI
        if args.mode == "gui" and not check_server_status(f"http://{args.host}:{args.port}"):
            print(f"Error: GUI mode requires a running server at {args.host}:{args.port}")
            print("Please start the server first with: vestim --mode server")
            return 1
        
        return start_gui(args)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())