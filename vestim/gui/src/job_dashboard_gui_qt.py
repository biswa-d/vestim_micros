import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QListWidget, QListWidgetItem, QLabel, QMessageBox, QDialog, QLineEdit,
    QGridLayout, QTextEdit, QComboBox, QDialogButtonBox, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import QTimer, Qt, pyqtSlot, pyqtSignal, QThread
import json
import os
from datetime import datetime
from vestim.gui.src.data_import_gui_qt import DataImportGUI
from vestim.gui.src.training_task_gui_qt import VEstimTrainingTaskGUI
from vestim.gui.src.testing_gui_qt import VEstimTestingGUI
from vestim.gui.src.api_gateway import APIGateway
from vestim.gui.src.hyper_param_gui_qt import VEstimHyperParamGUI

class JobDashboard(QMainWindow):
    # Define signals
    jobCreated = pyqtSignal(str)
    jobDeleted = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.api = APIGateway()
        self.setWindowTitle("VEstim Job Dashboard")
        self.setGeometry(100, 100, 900, 600)

        # A dictionary to keep track of open GUI windows for each job
        self.open_windows = {}
        
        # Track server connection status
        self.server_connected = False
        self.connection_attempts = 0
        self.max_connection_attempts = 10  # Maximum number of connection attempts before showing error
        
        # Map job statuses to the GUI component that should handle them
        self.gui_map = {
            "created": VEstimHyperParamGUI,
            "data_augmented": VEstimHyperParamGUI,
            "hyperparameters_set": VEstimTrainingTaskGUI,  # Assuming this status is set by the hyper_param_gui
            "training": VEstimTrainingTaskGUI,
            "testing": VEstimTestingGUI,
        }

        # Connect API gateway signals
        self.api.connectionError.connect(self.handle_connection_error)
        self.api.requestCompleted.connect(self.handle_request_completed)
        self.api.requestListCompleted.connect(self.handle_request_list_completed)  # Add connection to list signal

        self.init_ui()
        self.check_server_connection()

    def init_ui(self):
        """Initialize the UI components."""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Status indicator 
        self.connection_status = QLabel("Connecting to server...")
        self.connection_status.setStyleSheet("color: orange; font-weight: bold;")
        self.connection_status.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.connection_status)

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
        self.new_job_button.setEnabled(False)  # Initially disabled until connected
        button_layout.addWidget(self.new_job_button)

        self.retry_connection_button = QPushButton("Retry Connection")
        self.retry_connection_button.clicked.connect(self.check_server_connection)
        self.retry_connection_button.setVisible(False)  # Initially hidden
        button_layout.addWidget(self.retry_connection_button)

        self.stop_server_button = QPushButton("Stop Server & Exit")
        self.stop_server_button.clicked.connect(self.stop_server)
        self.stop_server_button.setEnabled(False)  # Initially disabled until connected
        button_layout.addWidget(self.stop_server_button)
        
        self.layout.addLayout(button_layout)
        self.statusBar().showMessage("Connecting to server...")

    def check_server_connection(self):
        """Check if the backend server is available."""
        self.connection_status.setText("Connecting to server...")
        self.connection_status.setStyleSheet("color: orange; font-weight: bold;")
        self.retry_connection_button.setVisible(False)
        
        # Disable buttons during connection check
        self.new_job_button.setEnabled(False)
        self.stop_server_button.setEnabled(False)
        
        # Use a separate thread to check server availability to avoid UI freezing
        # during network operations
        class ServerCheckThread(QThread):
            resultReady = pyqtSignal(bool)
            
            def __init__(self, api):
                super().__init__()
                self.api = api
                
            def run(self):
                result = self.api.is_server_available()
                self.resultReady.emit(result)
        
        # Create and start the thread
        self.check_thread = ServerCheckThread(self.api)
        self.check_thread.resultReady.connect(self._handle_connection_check_result)
        self.check_thread.start()
        
        # Show status in UI while checking
        self.statusBar().showMessage(f"Checking server connection... Attempt {self.connection_attempts+1}/{self.max_connection_attempts}")
        
    def _handle_connection_check_result(self, is_available):
        """Handle the result of the server connection check."""
        if is_available:
            self.handle_successful_connection()
        else:
            self.connection_attempts += 1
            if self.connection_attempts >= self.max_connection_attempts:
                self.handle_failed_connection()
            else:
                # Try again after a delay
                QTimer.singleShot(2000, self.check_server_connection)
                self.statusBar().showMessage(f"Connecting to server... Attempt {self.connection_attempts}/{self.max_connection_attempts}")

    def handle_successful_connection(self):
        """Handle successful connection to the server."""
        self.server_connected = True
        self.connection_attempts = 0
        self.connection_status.setText("Connected to server")
        self.connection_status.setStyleSheet("color: green; font-weight: bold;")
        self.statusBar().showMessage(f"Connected to server at {self.api.base_url}")
        
        # Enable buttons
        self.new_job_button.setEnabled(True)
        self.stop_server_button.setEnabled(True)
        self.retry_connection_button.setVisible(False)
        
        # Start polling for job updates
        self.start_polling()

    def handle_failed_connection(self):
        """Handle failed connection to the server."""
        self.server_connected = False
        self.connection_status.setText("Failed to connect to server")
        self.connection_status.setStyleSheet("color: red; font-weight: bold;")
        self.statusBar().showMessage("Server connection failed. Please check if the server is running.")
        
        # Show retry button
        self.retry_connection_button.setVisible(True)
        
        # Disable action buttons
        self.new_job_button.setEnabled(False)
        self.stop_server_button.setEnabled(False)

    @pyqtSlot(str)
    def handle_connection_error(self, error_message):
        """Handle connection errors from the API gateway."""
        self.statusBar().showMessage(f"Connection error: {error_message}")
        
        # If we were previously connected, try to reconnect
        if self.server_connected:
            self.server_connected = False
            self.connection_status.setText("Connection lost")
            self.connection_status.setStyleSheet("color: red; font-weight: bold;")
            self.retry_connection_button.setVisible(True)

    @pyqtSlot(dict)
    def handle_request_completed(self, response):
        """Handle successful API responses."""
        # If this is the first successful response after a connection loss, update status
        if not self.server_connected:
            self.handle_successful_connection()

    @pyqtSlot(list)
    def handle_request_list_completed(self, response):
        """Handle successful API responses that return lists."""
        # If this is the first successful response after a connection loss, update status
        if not self.server_connected:
            self.handle_successful_connection()
            
        # If this is a jobs list, update the job table
        self.refresh_jobs_with_data(response)

    def start_polling(self):
        """Starts the timer to periodically refresh the job list."""
        if hasattr(self, 'poll_timer') and self.poll_timer.isActive():
            return  # Already polling
            
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self.refresh_jobs)
        self.poll_timer.start(3000)  # Poll every 3 seconds
        self.refresh_jobs() # Initial refresh

    def refresh_jobs(self):
        """Fetches jobs from the backend and updates the table."""
        if not self.server_connected:
            # Don't try to refresh if not connected
            return
            
        try:
            # The jobs data will be handled by the requestListCompleted signal
            self.api.get_all_jobs()
        except Exception as e:
            self.statusBar().showMessage(f"Error refreshing jobs: {e}")
            # Check if server is still connected
            if self.server_connected:
                QTimer.singleShot(5000, self.check_server_connection)

    def refresh_jobs_with_data(self, jobs):
        """Updates the jobs table with provided job data."""
        if not isinstance(jobs, list):
            return
            
        try:
            self.job_table.setRowCount(len(jobs))
            for i, job in enumerate(jobs):
                self.populate_job_row(i, job)
            
            self.statusBar().showMessage(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
        except Exception as e:
            self.statusBar().showMessage(f"Error refreshing jobs: {e}")
            # Check if server is still connected
            if self.server_connected:
                QTimer.singleShot(5000, self.check_server_connection)

    def populate_job_row(self, row_index, job_data):
        """Populates a single row in the job table."""
        try:
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
        except Exception as e:
            print(f"Error populating job row: {e}")
            # Create a row with error information
            self.job_table.setItem(row_index, 0, QTableWidgetItem("Error"))
            self.job_table.setItem(row_index, 1, QTableWidgetItem(""))
            self.job_table.setItem(row_index, 2, QTableWidgetItem(str(e)))
            self.job_table.setItem(row_index, 3, QTableWidgetItem(""))
            self.job_table.setItem(row_index, 4, QTableWidgetItem(""))

    def open_job_gui(self, job_id, status):
        """Opens the appropriate GUI for a job based on its status."""
        if not self.server_connected:
            QMessageBox.warning(self, "Warning", "Cannot open job: Not connected to server.")
            return
            
        if job_id in self.open_windows and self.open_windows[job_id].isVisible():
            self.open_windows[job_id].activateWindow()
            return

        gui_class = self.gui_map.get(status)
        if gui_class:
            # Pass the API gateway and job_id to the specific GUI
            try:
                gui_instance = gui_class(api_gateway=self.api, job_id=job_id)
                self.open_windows[job_id] = gui_instance
                gui_instance.show()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to open job GUI: {e}")
        else:
            QMessageBox.information(self, "Info", f"No specific GUI for status '{status}'.")

    def create_new_job(self):
        """Launches the Data Import GUI to start the new job workflow."""
        if not self.server_connected:
            QMessageBox.warning(self, "Warning", "Cannot create job: Not connected to server.")
            return
            
        try:
            # This instance should be a member of the class to prevent it from
            # being garbage collected immediately.
            self.data_import_gui = DataImportGUI(api_gateway=self.api)
            
            # Connect the jobCreated signal to the refresh_jobs slot
            self.data_import_gui.jobCreated.connect(self.on_job_created)
            
            self.data_import_gui.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open Data Import GUI: {e}")
    
    @pyqtSlot(str)
    def on_job_created(self, job_id):
        """Handle when a new job is created."""
        self.refresh_jobs()
        self.jobCreated.emit(job_id)

    def kill_job(self, job_id):
        """Sends a request to delete/kill a job."""
        if not self.server_connected:
            QMessageBox.warning(self, "Warning", "Cannot kill job: Not connected to server.")
            return
            
        reply = QMessageBox.question(self, 'Confirm Kill', f"Are you sure you want to kill job {job_id}?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            try:
                result = self.api.delete_job(job_id)
                if result.get('status') == 'success':
                    QMessageBox.information(self, "Success", f"Job {job_id} killed successfully.")
                    # Close the GUI if it's open
                    if job_id in self.open_windows and self.open_windows[job_id].isVisible():
                        self.open_windows[job_id].close()
                        del self.open_windows[job_id]
                    self.jobDeleted.emit(job_id)
                else:
                    QMessageBox.warning(self, "Warning", f"Job kill response: {result.get('message', 'Unknown error')}")
                self.refresh_jobs()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to kill job: {e}")

    def stop_server(self):
        """Shuts down the backend server and closes the application."""
        if not self.server_connected:
            # If not connected, just close the application
            self.close()
            return
            
        reply = QMessageBox.question(self, 'Confirm Shutdown', "Are you sure you want to stop the server?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            try:
                self.api.shutdown_server()
                QMessageBox.information(self, "Server Shutdown", "Server shutdown signal sent. The application will now close.")
                self.close()
            except Exception as e:
                # The shutdown request might not get a response, which is okay.
                QMessageBox.information(self, "Server Shutdown", "Server shutdown attempted. The application will now close.")
                self.close()
    
    def closeEvent(self, event):
        """Ensure the polling timer is stopped when the window closes."""
        if hasattr(self, 'poll_timer') and self.poll_timer.isActive():
            self.poll_timer.stop()
        # Close all child windows
        for window in list(self.open_windows.values()):
            window.close()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dashboard = JobDashboard()
    dashboard.show()
    sys.exit(app.exec_())