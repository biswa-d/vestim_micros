import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QListWidget, QListWidgetItem, QLabel, QMessageBox, QDialog, QLineEdit,
    QGridLayout, QTextEdit, QComboBox, QDialogButtonBox, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import QTimer, Qt
import json
import os
from datetime import datetime
from vestim.gui.src.data_import_gui_qt import DataImportGUI
from vestim.gui.src.training_task_gui_qt import VEstimTrainingTaskGUI
from vestim.gui.src.testing_gui_qt import VEstimTestingGUI
from vestim.gui.src.api_gateway import APIGateway
from vestim.gui.src.hyper_param_gui_qt import VEstimHyperParamGUI

class JobDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.api = APIGateway()
        self.setWindowTitle("VEstim Job Dashboard")
        self.setGeometry(100, 100, 900, 600)

        # A dictionary to keep track of open GUI windows for each job
        self.open_windows = {}

        # Map job statuses to the GUI component that should handle them
        self.gui_map = {
            "created": VEstimHyperParamGUI,
            "data_augmented": VEstimHyperParamGUI,
            "hyperparameters_set": VEstimTrainingTaskGUI, # Assuming this status is set by the hyper_param_gui
            "training": VEstimTrainingTaskGUI,
            "testing": VEstimTestingGUI,
        }

        self.init_ui()
        self.start_polling()

    def init_ui(self):
        """Initialize the UI components."""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.title_label = QLabel("VEstim Active Jobs")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.title_label)

        self.job_table = QTableWidget()
        self.job_table.setColumnCount(5)
        self.job_table.setHorizontalHeaderLabels(["Job ID", "Time Since Start", "Status", "Open", "Kill"])
        self.job_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.job_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.layout.addWidget(self.job_table)

        button_layout = QHBoxLayout()
        self.new_job_button = QPushButton("Start New Job")
        self.new_job_button.clicked.connect(self.create_new_job)
        button_layout.addWidget(self.new_job_button)

        self.stop_server_button = QPushButton("Stop Server & Exit")
        self.stop_server_button.clicked.connect(self.stop_server)
        button_layout.addWidget(self.stop_server_button)
        
        self.layout.addLayout(button_layout)
        self.statusBar().showMessage("Connecting to server...")

    def start_polling(self):
        """Starts the timer to periodically refresh the job list."""
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self.refresh_jobs)
        self.poll_timer.start(3000)  # Poll every 3 seconds
        self.refresh_jobs() # Initial refresh

    def refresh_jobs(self):
        """Fetches jobs from the backend and updates the table."""
        try:
            jobs = self.api.get_all_jobs()
            if jobs is None:
                self.statusBar().showMessage("Error: Could not connect to the server.")
                return

            self.job_table.setRowCount(len(jobs))
            for i, job in enumerate(jobs):
                self.populate_job_row(i, job)
            
            self.statusBar().showMessage(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

        except Exception as e:
            self.statusBar().showMessage(f"Error refreshing jobs: {e}")

    def populate_job_row(self, row_index, job_data):
        """Populates a single row in the job table."""
        job_id = job_data.get('job_id', 'N/A')
        status = job_data.get('status', 'unknown')
        
        # Calculate time since start
        created_at_str = job_data.get('created_at')
        time_since_start = "N/A"
        if created_at_str:
            try:
                created_at = datetime.fromisoformat(created_at_str)
                delta = datetime.now() - created_at
                time_since_start = str(delta).split('.')[0]
            except ValueError:
                pass

        self.job_table.setItem(row_index, 0, QTableWidgetItem(job_id))
        self.job_table.setItem(row_index, 1, QTableWidgetItem(time_since_start))
        self.job_table.setItem(row_index, 2, QTableWidgetItem(status))

        # Open Job Button
        open_button = QPushButton("Open")
        open_button.clicked.connect(lambda _, j=job_id, s=status: self.open_job_gui(j, s))
        self.job_table.setCellWidget(row_index, 3, open_button)

        # Kill Job Button
        kill_button = QPushButton("Kill")
        kill_button.clicked.connect(lambda _, j=job_id: self.kill_job(j))
        self.job_table.setCellWidget(row_index, 4, kill_button)

    def open_job_gui(self, job_id, status):
        """Opens the appropriate GUI for a job based on its status."""
        if job_id in self.open_windows and self.open_windows[job_id].isVisible():
            self.open_windows[job_id].activateWindow()
            return

        gui_class = self.gui_map.get(status)
        if gui_class:
            # Pass the API gateway and job_id to the specific GUI
            gui_instance = gui_class(api_gateway=self.api, job_id=job_id)
            self.open_windows[job_id] = gui_instance
            gui_instance.show()
        else:
            QMessageBox.information(self, "Info", f"No specific GUI for status '{status}'.")

    def create_new_job(self):
        """Launches the Data Import GUI to start the new job workflow."""
        # This instance should be a member of the class to prevent it from
        # being garbage collected immediately.
        self.data_import_gui = DataImportGUI(api_gateway=self.api)
        
        # Connect the jobCreated signal to the refresh_jobs slot
        self.data_import_gui.jobCreated.connect(self.refresh_jobs)
        
        self.data_import_gui.show()

    def kill_job(self, job_id):
        """Sends a request to delete/kill a job."""
        reply = QMessageBox.question(self, 'Confirm Kill', f"Are you sure you want to kill job {job_id}?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            try:
                self.api.delete_job(job_id)
                self.refresh_jobs()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to kill job: {e}")

    def stop_server(self):
        """Shuts down the backend server and closes the application."""
        reply = QMessageBox.question(self, 'Confirm Shutdown', "Are you sure you want to stop the server?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            try:
                self.api.shutdown_server()
                self.close()
            except Exception as e:
                # The shutdown request might not get a response, which is okay.
                self.close()
    
    def closeEvent(self, event):
        """Ensure the polling timer is stopped when the window closes."""
        self.poll_timer.stop()
        # Close all child windows
        for window in self.open_windows.values():
            window.close()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dashboard = JobDashboard()
    dashboard.show()
    sys.exit(app.exec_())