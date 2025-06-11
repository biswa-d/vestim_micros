import sys
import time
import requests
from PyQt5.QtWidgets import QApplication, QMessageBox
from vestim.gui.src.job_dashboard_gui_qt import JobDashboard

def is_server_running(url, retries=3, delay=1):
    """Checks if the backend server is ready to accept connections."""
    for i in range(retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print("Backend server is available.")
                return True
        except requests.ConnectionError:
            pass
        print(f"Server not available. Retrying in {delay} second(s)...")
        time.sleep(delay)
    return False

def show_server_error_message():
    """Shows an error message if the server is not running."""
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setWindowTitle("Server Connection Error")
    msg.setText("Could not connect to the VEstim backend server.")
    msg.setInformativeText(
        "The server is not running or cannot be reached. Please start the server first using:\n\n"
        "vestim-server\n\n"
        "Or with:\n\n"
        "python -m vestim.scripts.run_server"
    )
    msg.setStandardButtons(QMessageBox.Ok)
    return msg.exec_()

def main():
    """
    Main entry point for the VEstim GUI.
    Connects to the backend server and starts the dashboard.
    """
    # Check if the server is already running
    if not is_server_running("http://127.0.0.1:8001"):
        print("Backend server is not running.")
        
        # Create a basic QApplication to show the error message
        app = QApplication(sys.argv)
        show_server_error_message()
        return 1

    # Start the GUI
    app = QApplication(sys.argv)
    gui = JobDashboard()
    gui.show()
    return app.exec_()

if __name__ == "__main__":
    sys.exit(main())