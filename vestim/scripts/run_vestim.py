import argparse
import sys
import subprocess
import time
from vestim.scripts.server_helper import is_server_running, stop_server, get_pid_file_path

def start_server():
    """Starts the backend server as a detached process."""
    if is_server_running():
        print("Server is already running.")
        return
    
    print("Starting the VEstim backend server...")
    
    try:
        # Start the server as a background process
        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.DETACHED_PROCESS
            
        process = subprocess.Popen(
            [sys.executable, "-m", "vestim.scripts.run_server"],
            creationflags=creationflags,
            close_fds=True
        )
        
        # Write the PID to a file
        pid_file = get_pid_file_path()
        with open(pid_file, "w") as f:
            f.write(str(process.pid))
            
        print(f"Server started with PID {process.pid}.")
        
    except Exception as e:
        print(f"Failed to start server: {e}")

def launch_gui():
    """Launches the VEstim GUI."""
    print("Launching the VEstim GUI...")
    
    # Wait for the server to be ready
    for _ in range(10):  # Wait up to 10 seconds
        if is_server_running():
            break
        time.sleep(1)
    else:
        print("Error: Server did not start in time. Cannot launch GUI.")
        return
        
    try:
        subprocess.run([sys.executable, "-m", "vestim.scripts.run_gui"])
    except Exception as e:
        print(f"Failed to launch GUI: {e}")

def main():
    """
    Main entry point for the VEstim tool.
    Provides commands to start, stop, and manage the application.
    """
    parser = argparse.ArgumentParser(description="VEstim Tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start the VEstim server and launch the GUI")
    
    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop the VEstim server")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check if the VEstim server is running")
    
    args = parser.parse_args()
    
    if args.command == "start":
        start_server()
        launch_gui()
    elif args.command == "stop":
        stop_server()
    elif args.command == "status":
        if is_server_running():
            print("VEstim server is running.")
        else:
            print("VEstim server is not running.")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()