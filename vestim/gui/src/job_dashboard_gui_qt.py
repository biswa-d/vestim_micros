import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QListWidget, QListWidgetItem, QLabel, QMessageBox, QDialog, QLineEdit, 
    QGridLayout, QTextEdit, QComboBox, QDialogButtonBox
)
from PyQt5.QtCore import QTimer, Qt
import requests
import json
import os
from datetime import datetime

class JobDetailsDialog(QDialog):
    """Dialog for displaying and interacting with job details."""
    def __init__(self, job_id, parent=None):
        super().__init__(parent)
        self.job_id = job_id
        self.setWindowTitle(f"Job Details - {job_id}")
        self.setMinimumSize(600, 400)
        
        self.layout = QVBoxLayout(self)
        
        # Job info section
        self.info_layout = QGridLayout()
        self.layout.addLayout(self.info_layout)
        
        self.info_layout.addWidget(QLabel("Job ID:"), 0, 0)
        self.info_layout.addWidget(QLabel(job_id), 0, 1)
        
        self.status_label = QLabel("Status:")
        self.info_layout.addWidget(self.status_label, 1, 0)
        self.status_value = QLabel("Loading...")
        self.info_layout.addWidget(self.status_value, 1, 1)
        
        self.created_label = QLabel("Created:")
        self.info_layout.addWidget(self.created_label, 2, 0)
        self.created_value = QLabel("Loading...")
        self.info_layout.addWidget(self.created_value, 2, 1)
        
        # Task list section
        self.layout.addWidget(QLabel("Tasks:"))
        self.task_list = QListWidget()
        self.layout.addWidget(self.task_list)
        
        # Action buttons
        self.button_layout = QHBoxLayout()
        self.layout.addLayout(self.button_layout)
        
        self.stop_all_button = QPushButton("Stop All Tasks")
        self.stop_all_button.clicked.connect(self.stop_all_tasks)
        self.button_layout.addWidget(self.stop_all_button)
        
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_job_details)
        self.button_layout.addWidget(self.refresh_button)
        
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        self.button_layout.addWidget(self.close_button)
        
        # Initial load
        self.refresh_job_details()
        
        # Auto-refresh timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_job_details)
        self.timer.start(5000)  # Refresh every 5 seconds
    
    def refresh_job_details(self):
        """Load the job details from the server."""
        try:
            response = requests.get(f"http://127.0.0.1:8001/jobs/{self.job_id}")
            response.raise_for_status()
            job = response.json()
            
            # Update the UI with job details
            self.status_value.setText(job.get("status", "Unknown"))
            
            created_at = job.get("created_at", "")
            if created_at:
                try:
                    dt = datetime.fromisoformat(created_at)
                    self.created_value.setText(dt.strftime("%Y-%m-%d %H:%M:%S"))
                except:
                    self.created_value.setText(created_at)
            
            # TODO: Load tasks associated with this job when API endpoint is available
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching job details: {e}")
    
    def stop_all_tasks(self):
        """Stop all tasks associated with this job."""
        try:
            # Call the API to stop all tasks
            response = requests.post("http://127.0.0.1:8001/tasks/stop_all")
            response.raise_for_status()
            
            QMessageBox.information(self, "Tasks Stopped", "All tasks have been stopped.")
            self.refresh_job_details()
        except requests.exceptions.RequestException as e:
            QMessageBox.warning(self, "Error", f"Failed to stop tasks: {e}")
    
    def closeEvent(self, event):
        """Clean up resources when dialog is closed."""
        self.timer.stop()
        super().closeEvent(event)

class NewJobDialog(QDialog):
    """Dialog for creating a new job."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create New Job")
        self.setMinimumWidth(400)
        
        self.layout = QVBoxLayout(self)
        
        # Form layout
        self.form_layout = QGridLayout()
        self.layout.addLayout(self.form_layout)
        
        self.form_layout.addWidget(QLabel("Data Folder:"), 0, 0)
        self.folder_input = QLineEdit()
        self.form_layout.addWidget(self.folder_input, 0, 1)
        
        self.form_layout.addWidget(QLabel("Data Type:"), 1, 0)
        self.data_type = QComboBox()
        self.data_type.addItems(["Battery", "Supercapacitor", "Other"])
        self.form_layout.addWidget(self.data_type, 1, 1)
        
        self.form_layout.addWidget(QLabel("Notes:"), 2, 0)
        self.notes_input = QTextEdit()
        self.notes_input.setMaximumHeight(100)
        self.form_layout.addWidget(self.notes_input, 2, 1)
        
        # Button box
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)
    
    def get_selections(self):
        """Return the form data as a selections dictionary."""
        return {
            "folder_path": self.folder_input.text(),
            "data_type": self.data_type.currentText(),
            "notes": self.notes_input.toPlainText(),
            "timestamp": datetime.now().isoformat()
        }

class JobDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VEstim Job Dashboard")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.title_label = QLabel("Active Jobs")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.title_label)

        self.job_list_widget = QListWidget()
        self.job_list_widget.itemDoubleClicked.connect(self.show_job_details)
        self.layout.addWidget(self.job_list_widget)

        self.button_layout = QHBoxLayout()
        self.layout.addLayout(self.button_layout)

        self.new_job_button = QPushButton("New Job")
        self.new_job_button.clicked.connect(self.create_new_job)
        self.button_layout.addWidget(self.new_job_button)

        self.stop_all_button = QPushButton("Stop All Tasks")
        self.stop_all_button.clicked.connect(self.stop_all_tasks)
        self.button_layout.addWidget(self.stop_all_button)

        self.clear_all_button = QPushButton("Clear All Jobs")
        self.clear_all_button.clicked.connect(self.clear_all_jobs)
        self.button_layout.addWidget(self.clear_all_button)
        
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_job_list)
        self.button_layout.addWidget(self.refresh_button)

        # Status bar
        self.statusBar().showMessage("Ready")

        # Auto-refresh timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_job_list)
        self.timer.start(5000)  # Refresh every 5 seconds

        # Initial refresh
        self.refresh_job_list()

    def refresh_job_list(self):
        """Fetch and display the list of jobs from the server."""
        try:
            response = requests.get("http://127.0.0.1:8001/jobs")
            response.raise_for_status()
            jobs = response.json()
            
            self.job_list_widget.clear()
            
            if not jobs:
                self.statusBar().showMessage("No jobs found")
                return
                
            for job in jobs:
                job_id = job.get('job_id', 'Unknown')
                status = job.get('status', 'Unknown')
                created_at = job.get('created_at', '')
                
                if created_at:
                    try:
                        dt = datetime.fromisoformat(created_at)
                        date_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        date_str = created_at
                else:
                    date_str = 'Unknown date'
                
                item = QListWidgetItem(f"{job_id} - Status: {status} - Created: {date_str}")
                item.setData(Qt.UserRole, job_id)
                self.job_list_widget.addItem(item)
                
            self.statusBar().showMessage(f"Found {len(jobs)} jobs")
            
        except requests.exceptions.ConnectionError:
            self.statusBar().showMessage("Error: Cannot connect to server")
        except requests.exceptions.RequestException as e:
            self.statusBar().showMessage(f"Error fetching jobs: {e}")

    def create_new_job(self):
        """Open dialog to create a new job."""
        dialog = NewJobDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            selections = dialog.get_selections()
            try:
                response = requests.post(
                    "http://127.0.0.1:8001/jobs", 
                    json={"selections": selections}
                )
                response.raise_for_status()
                result = response.json()
                
                QMessageBox.information(
                    self, 
                    "Job Created", 
                    f"Job created successfully.\nJob ID: {result['job_id']}"
                )
                
                self.refresh_job_list()
            except requests.exceptions.RequestException as e:
                QMessageBox.critical(self, "Error", f"Error creating new job: {e}")

    def stop_all_tasks(self):
        """Stop all running tasks."""
        try:
            response = requests.post("http://127.0.0.1:8001/tasks/stop_all")
            response.raise_for_status()
            QMessageBox.information(self, "Tasks Stopped", "All tasks have been stopped.")
            self.refresh_job_list()
        except requests.exceptions.RequestException as e:
            QMessageBox.critical(self, "Error", f"Error stopping tasks: {e}")

    def clear_all_jobs(self):
        """Clear all jobs from the server."""
        reply = QMessageBox.question(
            self, 
            "Confirm Delete", 
            "Are you sure you want to delete ALL jobs and their data?\nThis action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                response = requests.post("http://127.0.0.1:8001/jobs/clear")
                response.raise_for_status()
                QMessageBox.information(self, "Jobs Cleared", "All jobs have been cleared.")
                self.refresh_job_list()
            except requests.exceptions.RequestException as e:
                QMessageBox.critical(self, "Error", f"Error clearing jobs: {e}")
    
    def show_job_details(self, item):
        """Show details for the selected job."""
        job_id = item.data(Qt.UserRole)
        if job_id:
            dialog = JobDetailsDialog(job_id, self)
            dialog.exec_()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    dashboard = JobDashboard()
    dashboard.show()
    sys.exit(app.exec_())