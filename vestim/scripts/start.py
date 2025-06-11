import subprocess
import sys
from vestim.scripts.server_helper import is_server_running, start_server

def main():
    """
    Starts the VEstim application.
    Checks if the server is running. If not, it starts the server.
    Then, it launches the GUI.
    """
    if not is_server_running():
        print("Server is not running. Starting server...")
        start_server()

    # Start the GUI
    gui_process = subprocess.Popen([sys.executable, "-m", "vestim.scripts.run_gui"])

    # Wait for the GUI to exit
    gui_process.wait()

if __name__ == "__main__":
    main()