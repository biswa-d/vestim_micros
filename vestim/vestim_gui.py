import os
import subprocess
import json
import sys
import requests
import psutil
import signal
from PyQt5.QtWidgets import QApplication
from vestim.gui.src.data_import_gui_qt_flask import DataImportGUI
from vestim.gui.src.hyper_param_gui_qt_flask import VEstimHyperParamGUI
from vestim.gui.src.training_task_gui_qt_flask import VEstimTrainingTaskGUI
from vestim.gui.src.testing_gui_qt_flask import VEstimTestingGUI


# Path to tool_state.json
TOOL_STATE_FILE = "vestim/tool_state.json"
FLASK_SERVER_URL = "http://localhost:5000"

def kill_existing_flask_process():
    """Kill the Flask process running on port 5000, without affecting other Flask apps."""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        # Only consider 'python' processes
        if proc.info['name'] == 'python':
            cmdline = ' '.join(proc.info['cmdline'])
            # Check if the process is running flask_app.py on port 5000
            if 'flask_app.py' in cmdline and '--port=5000' in cmdline:
                print(f"Found running Flask process on port 5000 with PID {proc.info['pid']}. Terminating it.")
                try:
                    os.kill(proc.info['pid'], signal.SIGTERM)
                    print(f"Successfully terminated Flask process {proc.info['pid']}.")
                except Exception as e:
                    print(f"Error terminating Flask process {proc.info['pid']}: {e}")

def start_flask_server():
    """Restart the Flask server, killing any running instance first."""
    try:
        # Check if the Flask server is running
        response = requests.get(f"{FLASK_SERVER_URL}/")
        if response.status_code == 200:
            print("Flask server is running. Restarting it.")
            kill_existing_flask_process()  # Kill the running server

    except requests.ConnectionError:
        print("Flask server is not running. Starting a new instance.")
    
    # Start the Flask server fresh
    subprocess.Popen([sys.executable, 'vestim/flask_app.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("Flask server started fresh.")

def load_tool_state():
    """Load the tool state from the tool_state.json file."""
    if not os.path.exists(TOOL_STATE_FILE):
        return None

    with open(TOOL_STATE_FILE, 'r') as f:
        tool_state = json.load(f)
    return tool_state

def save_tool_state(state):
    """Save the updated state to the tool_state.json file."""
    with open(TOOL_STATE_FILE, 'w') as f:
        json.dump(state, f, indent=4)

def launch_gui(app, current_screen):
    """Launch the appropriate GUI based on the current tool state."""
    if current_screen == "DataImportGUI":
        gui = DataImportGUI()
    elif current_screen == "VEstimHyperParamGUI":
        gui = VEstimHyperParamGUI()
    elif current_screen == "VEstimTrainingTaskGUI":
        gui = VEstimTrainingTaskGUI()
    elif current_screen == "VEstimTestingGUI":
        gui = VEstimTestingGUI()
    else:
        print("Unknown screen. Defaulting to Data Import.")
        gui = DataImportGUI()  # Default to DataImportGUI if state is unknown

    gui.show()
    sys.exit(app.exec_())

def main():
    # Check tool state
    tool_state = load_tool_state()

    # If no state exists, default to DataImportGUI and restart Flask server
    if tool_state is None or tool_state.get('current_state') == 'data_import':
        print("No tool state found or starting from the beginning.")
        start_flask_server()  # Restart the server only when starting fresh
        tool_state = {
            "current_state": "data_import",
            "current_screen": "DataImportGUI"
        }
        save_tool_state(tool_state)

    # Determine the current screen from the state
    current_screen = tool_state.get("current_screen", "DataImportGUI")

    # Launch the appropriate GUI
    app = QApplication(sys.argv)
    launch_gui(app, current_screen)

if __name__ == "__main__":
    main()
