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

class JobDetailsDialog(QDialog):
    """Dialog for displaying and interacting with job details."""
    def __init__(self, job_id, parent=None):
        super().__init__(parent)
        self.job_id = job_id
        self.api_gateway = APIGateway()
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
        self.task_list.itemDoubleClicked.connect(self.open_task_gui)
        self.layout.addWidget(self.task_list)
        
        # Action buttons
        self.button_layout = QHBoxLayout()
        self.layout.addLayout(self.button_layout)
        
        self.open_task_button = QPushButton("Open Task")
        self.open_task_button.clicked.connect(self.open_task_gui)
        self.button_layout.addWidget(self.open_task_button)

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
            job = self.api_gateway.get(f"jobs/{self.job_id}")
            if not job:
                print(f"Error fetching job details for {self.job_id}")
                return

            # Update the UI with job details
            self.status_value.setText(job.get("status", "Unknown"))
            
            created_at = job.get("created_at", "")
            if created_at:
                try:
                    dt = datetime.fromisoformat(created_at)
                    self.created_value.setText(dt.strftime("%Y-%m-%d %H:%M:%S"))
                except:
                    self.created_value.setText(created_at)
            
            self.refresh_task_list()
            
        except Exception as e:
            print(f"Error refreshing job details: {e}")

    def stop_all_tasks(self):
        """Stop all tasks associated with this job."""
        try:
            self.api_gateway.post(f"jobs/{self.job_id}/stop")
            QMessageBox.information(self, "Job Stopped", f"Stop signal sent to job {self.job_id}.")
            self.refresh_job_details()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to stop job: {e}")

    def refresh_task_list(self):
        """Refreshes the list of tasks for the current job."""
        try:
            status = self.api_gateway.get(f"jobs/{self.job_id}/status")
            if status is None:
                return
            self.task_list.clear()
            item = QListWidgetItem(f"{status['task_id']} - Status: {status['status']}")
            item.setData(Qt.UserRole, status)
            self.task_list.addItem(item)
        except Exception as e:
            print(f"Error refreshing task list: {e}")

    def open_task_gui(self):
        """Opens the training GUI for the selected task."""
        try:
            self.api_gateway.post(f"jobs/{self.job_id}/train")
            QMessageBox.information(self, "Training Started", f"Training started for job {self.job_id}.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to start training: {e}")
    
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
        self.api_gateway = APIGateway()
        self.setWindowTitle("VEstim Job Dashboard")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.title_label = QLabel("Active Jobs")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.title_label)

        self.job_table_widget = QTableWidget()
        self.job_table_widget.setColumnCount(5)
        self.job_table_widget.setHorizontalHeaderLabels(["Job ID", "Time Since Start", "Present State", "Open Job", "Kill Job"])
        self.job_table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.job_table_widget.setEditTriggers(QTableWidget.NoEditTriggers)
        self.layout.addWidget(self.job_table_widget)

        self.button_layout = QHBoxLayout()
        self.layout.addLayout(self.button_layout)

        self.new_job_button = QPushButton("New Job")
        self.new_job_button.clicked.connect(self.create_new_job)
        self.button_layout.addWidget(self.new_job_button)

        self.stop_server_button = QPushButton("Stop Server")
        self.stop_server_button.clicked.connect(self.stop_server)
        self.button_layout.addWidget(self.stop_server_button)

        
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
            jobs = self.api_gateway.get("jobs")
            if jobs is None:
                self.job_table_widget.setRowCount(0)
                self.statusBar().showMessage("Error: Cannot connect to the server. Please ensure it is running.")
                return

            self.job_table_widget.setRowCount(0)
            
            if not jobs:
                self.statusBar().showMessage("No jobs found")
                return
                
            self.job_table_widget.setRowCount(len(jobs))
            for i, job in enumerate(jobs):
                job_id = job.get('job_id', 'Unknown')
                status = job.get('status', 'Unknown')
                created_at = job.get('created_at', '')
                
                if created_at:
                    try:
                        dt = datetime.fromisoformat(created_at)
                        time_since_start = datetime.now() - dt
                        time_str = str(time_since_start).split('.')[0]
                    except:
                        time_str = "Unknown"
                else:
                    time_str = 'Unknown'

                self.job_table_widget.setItem(i, 0, QTableWidgetItem(job_id))
                self.job_table_widget.setItem(i, 1, QTableWidgetItem(time_str))
                self.job_table_widget.setItem(i, 2, QTableWidgetItem(status))
                
                open_button = QPushButton("Open")
                open_button.clicked.connect(lambda _, j=job_id, s=status: self.show_job_details(j, s))
                self.job_table_widget.setCellWidget(i, 3, open_button)

                kill_button = QPushButton("Kill")
                kill_button.clicked.connect(lambda _, j=job_id: self.kill_job(j))
                self.job_table_widget.setCellWidget(i, 4, kill_button)
                
            self.statusBar().showMessage(f"Found {len(jobs)} jobs")
            
        except Exception as e:
            self.job_table_widget.setRowCount(0)
            self.statusBar().showMessage(f"Error fetching jobs: {e}")
            
    def create_new_job(self):
        """Open the data import GUI to start a new job."""
        self.data_import_gui = DataImportGUI()
        self.data_import_gui.show()

    def stop_server(self):
        """Stop the backend server and close the application."""
        reply = QMessageBox.question(
            self,
            "Confirm Stop",
            "Are you sure you want to stop the server?\nThis will close the application.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                self.api_gateway.post("server/shutdown")
                QMessageBox.information(self, "Server Stopped", "The server has been stopped.")
                self.close()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error stopping server: {e}")

    
    def show_job_details(self, job_id, status):
        """Show details for the selected job."""
        if not job_id:
            return
            
        if status == "created":
            self.hyper_param_gui = VEstimHyperParamGUI(job_id)
            self.hyper_param_gui.show()
        elif status == "training":
            self.training_gui = VEstimTrainingTaskGUI(job_id)
            self.training_gui.show()
        elif status == "testing":
            self.testing_gui = VEstimTestingGUI(job_id)
            self.testing_gui.show()
        else:
            dialog = JobDetailsDialog(job_id, self)
            dialog.exec_()

    def kill_job(self, job_id):
        """Stop a running job."""
        reply = QMessageBox.question(
            self,
            "Confirm Kill",
            f"Are you sure you want to kill job {job_id}?\nThis action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                response = self.api_gateway.delete(f"jobs/{job_id}")
                if response:
                    QMessageBox.information(self, "Job Killed", f"Job {job_id} has been killed.")
                    self.refresh_job_list()
                else:
                    QMessageBox.critical(self, "Error", f"Failed to kill job {job_id}.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error killing job: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    dashboard = JobDashboard()
    dashboard.show()
    sys.exit(app.exec_())