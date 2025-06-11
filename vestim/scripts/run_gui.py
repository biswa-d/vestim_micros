import sys
from PyQt5.QtWidgets import QApplication, QMessageBox
from vestim.gui.src.job_dashboard_gui_qt import JobDashboard
from vestim.scripts.server_helper import is_server_running

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
    app = QApplication(sys.argv)
    
    if not is_server_running():
        show_server_error_message()
        return 1

    gui = JobDashboard()
    gui.show()
    return app.exec_()

if __name__ == "__main__":
    sys.exit(main())