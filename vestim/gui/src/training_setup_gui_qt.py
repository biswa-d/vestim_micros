from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
import sys
import time
from vestim.gui.src.api_gateway import APIGateway
from vestim.gui.src.training_task_gui_qt import VEstimTrainingTaskGUI
import logging

class SetupWorker(QThread):
    progress_signal = pyqtSignal(str, str, int)  # Signal to update the status in the main GUI
    finished_signal = pyqtSignal(dict)  # Signal when the setup is finished, passing the response data

    def __init__(self, job_id, api_gateway):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.job_id = job_id
        self.api_gateway = api_gateway
        self.is_cancelled = False

    def run(self):
        self.logger.info(f"Starting training setup for job {self.job_id} via API call.")
        self.progress_signal.emit("Setting up training tasks...", "", 0)
        
        try:
            # First, log that we're about to make the API call
            self.logger.info(f"Making API call to setup training for job {self.job_id}")
            self.progress_signal.emit("Sending setup request to server...", "", 0)
              # Use the new setup_training method with a longer timeout
            response = self.api_gateway.setup_training(self.job_id, timeout=60)
            
            if response and response.get("status") == "success":
                task_count = response.get("task_count", 0)
                message = response.get("message", f"Setup completed with {task_count} tasks.")
                self.progress_signal.emit(message, "", task_count)
                self.logger.info(f"Training setup successful: {message}")
            else:
                error_message = response.get("detail", response.get("message", "Unknown error")) if response else "No response"
                self.progress_signal.emit(f"Error: {error_message}", "", 0)
                self.logger.error(f"Training setup failed: {error_message}")
                
            # Always emit the finished signal with whatever response we got
            self.finished_signal.emit(response or {"status": "error", "message": "No response from server"})
            
        except Exception as e:
            self.logger.error(f"Error during training setup API call: {e}", exc_info=True)
            self.progress_signal.emit(f"Error occurred: {str(e)}", "", 0)
            self.finished_signal.emit({"status": "error", "message": str(e)})

class VEstimTrainSetupGUI(QWidget):
    def __init__(self, job_id: str, api_gateway: APIGateway):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.job_id = job_id
        self.api_gateway = api_gateway
        self.params = {}
        self.timer_running = True
        self.setup_completed = False
        self.setup_in_progress = False
        self.training_task_gui = None
        
        # Get job status for state restoration
        try:
            self.job_status = self.api_gateway.get_job_detailed_status(self.job_id)
            if self.job_status:
                self.logger.info(f"Retrieved job status: {self.job_status.get('status')} - {self.job_status.get('progress_message')}")
            else:
                self.logger.warning("Could not retrieve detailed job status")
                self.job_status = {"status": "hyperparameters_set", "phase_progress": {}}
        except Exception as e:
            self.logger.error(f"Error retrieving job status: {e}")
            self.job_status = {"status": "hyperparameters_set", "phase_progress": {}}
        
        # Get job details including hyperparameters from JobContainer's hyperparam manager
        try:
            job_info = self.api_gateway.get_job(self.job_id)
            self.params = job_info.get("details", {}).get("hyperparameters", {})
            self.logger.info(f"Retrieved hyperparameters for job {self.job_id}: {self.params}")
        except Exception as e:
            self.logger.error(f"Error retrieving job information: {e}")
            self.params = {}
        
        self.param_labels = {
            "LAYERS": "Layers",
            "HIDDEN_UNITS": "Hidden Units",
            "BATCH_SIZE": "Batch Size",
            "MAX_EPOCHS": "Max Epochs",  # Updated to match the correct parameter name
            "EPOCHS": "Max Epochs",  # Keep for backward compatibility
            "INITIAL_LR": "Initial Learning Rate",
            "LR_PARAM": "LR Parameter",
            "LR_PERIOD": "LR Period",
            "EARLY_STOPPING_PATIENCE": "Validation Patience",
            "VALIDATION_FREQ": "Validation Frequency",
            "LOOKBACK": "Lookback Sequence Length",
            "REPETITIONS": "Repetitions",
            "FEATURE_COLUMNS": "Feature Columns",
            "TARGET_COLUMN": "Target Column",
            "MODEL_TYPE": "Model Type",
            "TRAINING_METHOD": "Training Method",
            "DEVICE": "Device"
        }

        self.logger.info(f"Initializing VEstimTrainSetupGUI for job_id: {self.job_id}")
        self.build_gui()
        self.restore_gui_state()
        self.fetch_hyperparameters_and_start()

    def restore_gui_state(self):
        """Restore GUI state based on current job status"""
        if not hasattr(self, 'job_status') or not self.job_status:
            return
        
        current_status = self.job_status.get('status', '')
        phase_progress = self.job_status.get('phase_progress', {})
        
        self.logger.info(f"Restoring TrainingSetup GUI state for status: {current_status}")
        
        # Check if training setup is already completed
        setup_progress = phase_progress.get('training_setup', {})
        setup_status = setup_progress.get('status', 'pending')
        
        if current_status == 'training_setup_completed' or setup_status == 'completed':
            # Training setup already completed - show completion state
            self.logger.info("Training setup already completed, showing completion state")
            self.setup_completed = True
            self.setup_in_progress = False
            
            # Show completion message
            task_count = setup_progress.get('task_count', 0)
            self.status_label.setText(f"Setup completed successfully! {task_count} training tasks created.")
            self.status_label.setStyleSheet("color: green; font-size: 12pt; font-weight: bold;")
            
            # Show continue button (will be added in fetch_hyperparameters_and_start)
            
        elif setup_status == 'in_progress':
            # Training setup in progress
            self.setup_in_progress = True
            self.status_label.setText("Training setup in progress...")
            self.status_label.setStyleSheet("color: #FF8C00; font-size: 12pt; font-weight: bold;")
        else:
            # Default state - ready to create tasks
            self.status_label.setText("Ready to create training tasks")
            self.status_label.setStyleSheet("color: #3a3a3a; font-size: 12pt;")
            
        # If status is 'hyperparameters_set' or 'pending', show normal initial state (default)
    
    def add_continue_button(self):
        """Add continue button to proceed to training task GUI"""
        self.continue_button = QPushButton("Continue to Training Tasks")
        self.continue_button.setStyleSheet("background-color: #0b6337; color: white; font-size: 12pt; font-weight: bold;")
        self.continue_button.clicked.connect(self.transition_to_training_gui)
        self.main_layout.addWidget(self.continue_button)

    def fetch_hyperparameters_and_start(self):
        try:
            job_details = self.api_gateway.get_job(self.job_id)
            self.params = job_details.get("details", {}).get("hyperparameters", {})
            if not self.params:
                self.status_label.setText("Could not load hyperparameters.")
                return
            
            self.display_hyperparameters(self.hyperparam_layout)
            
            # Show appropriate button based on current state
            if self.setup_completed:
                # Setup already completed - show "Continue to Training" button
                if not hasattr(self, 'continue_button'):
                    self.add_continue_button()
            else:
                # Setup not completed - show "Create Training Tasks" button
                if not hasattr(self, 'create_tasks_button'):
                    self.add_create_tasks_button()
                
        except Exception as e:
            self.status_label.setText(f"Error fetching parameters: {e}")

    def add_create_tasks_button(self):
        """Add button to create training tasks"""
        self.create_tasks_button = QPushButton("Create Training Tasks")
        self.create_tasks_button.setStyleSheet("background-color: #0b6337; color: white; font-size: 12pt; font-weight: bold;")
        self.create_tasks_button.clicked.connect(self.start_setup)
        self.main_layout.addWidget(self.create_tasks_button)

    def build_gui(self):
        self.setWindowTitle("VEstim - Setting Up Training")
        self.setMinimumSize(900, 600)
        self.setMaximumSize(900, 600)  # This makes it appear "fixed"
        
        # Main layout
        self.main_layout = QVBoxLayout()

        # Title label
        title_label = QLabel("Building LSTM Models and Training Tasks\nwith Hyperparameter Set")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: #3a3a3a;")
        self.main_layout.addWidget(title_label)

        # Time tracking label
        time_layout = QHBoxLayout()
        self.static_text_label = QLabel("Time Since Setup Started:")
        self.static_text_label.setStyleSheet("color: blue; font-size: 10pt;")
        self.time_value_label = QLabel("00h:00m:00s")
        self.time_value_label.setStyleSheet("color: purple; font-size: 12pt; font-weight: bold;")
        time_layout.addStretch(1)
        time_layout.addWidget(self.static_text_label)
        time_layout.addWidget(self.time_value_label)
        time_layout.addStretch(1)
        self.main_layout.addLayout(time_layout)

        # Hyperparameter display area
        self.hyperparam_frame = QFrame()
        self.hyperparam_layout = QGridLayout()
        self.hyperparam_frame.setLayout(self.hyperparam_layout)
        self.main_layout.addWidget(self.hyperparam_frame)
        
        # Status label
        self.status_label = QLabel("Setting up training...")
        self.status_label.setStyleSheet("color: green; font-size: 12pt; font-weight: bold;")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.status_label)
        # Set the main layout
        self.setLayout(self.main_layout)

    def display_hyperparameters(self, layout):
        items = list(self.params.items())

        # Iterate through the rows and columns to display hyperparameters
        for i, (param, value) in enumerate(items):
            row = i // 2
            col = (i % 2) * 2

            label_text = self.param_labels.get(param, param)
            value_str = str(value)

            # Truncate long value strings for display
            display_value = ', '.join(value_str.split(',')[:2]) + '...' if ',' in value_str and len(value_str.split(',')) > 2 else value_str

            param_label = QLabel(f"{label_text}:")
            param_label.setStyleSheet("font-size: 10pt; background-color: #f0f0f0; padding: 5px;")
            layout.addWidget(param_label, row, col)

            value_label = QLabel(display_value)
            value_label.setStyleSheet("font-size: 10pt; color: #005878; font-weight: bold; background-color: #f0f0f6; padding: 5px;")
            layout.addWidget(value_label, row, col + 1)

    def start_setup(self):
        """Starts the training setup process via API call."""
        if self.setup_in_progress:
            self.logger.info("Setup is already in progress, ignoring request.")
            return
            
        self.setup_in_progress = True
        print("Starting training setup...")
        self.logger.info("Starting training setup...")
        self.start_time = time.time()
        
        # Update UI to show setup is starting
        self.status_label.setText("Initializing training setup...")
        self.status_label.setStyleSheet("color: #FF8C00; font-size: 12pt; font-weight: bold;")  # Orange color for processing

        # Ensure the window is fully built and ready
        self.show()

        # Move the training setup to a separate thread
        self.worker = SetupWorker(self.job_id, self.api_gateway)
        self.worker.progress_signal.connect(self.update_status)
        self.worker.finished_signal.connect(self.on_setup_finished)

        # Start the worker thread
        self.worker.start()
        print("Training setup started...")

        # Update elapsed time in the main thread
        self.update_elapsed_time()

    def update_status(self, message, path="", task_count=None):
        task_message = f"{task_count} training tasks created,\n" if task_count else ""
        # Format the status with the job folder and tasks count
        formatted_message = f"{task_message}{message}\n{path}" if path else f"{task_message}{message}"

        # Update the status label with the new message
        self.status_label.setText(formatted_message)
        self.status_label.setStyleSheet("color: green; font-size: 12pt; font-weight: bold;")

    def on_setup_finished(self, response_data):
        """Handle completion of training setup process."""
        self.logger.info(f"Training setup finished with response: {response_data}")
        self.timer_running = False
        self.setup_in_progress = False
        
        if hasattr(self, 'worker') and self.worker:
            self.worker.quit()
            self.worker.wait()

        # Calculate time taken
        elapsed_time = time.time() - self.start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        total_time_taken = f"{int(hours):02}h:{int(minutes):02}m:{int(seconds):02}s"        # Check if setup was successful
        if response_data and response_data.get("status") == "success":
            self.setup_completed = True
            task_count = response_data.get("task_count", 0)
            job_id = response_data.get("job_id", self.job_id)

            # Update backend status to indicate training setup is completed
            try:
                self.api_gateway.update_job_status(
                    job_id=self.job_id,
                    status="training_setup_completed",
                    message=f"Training setup completed with {task_count} tasks created",
                    progress_percent=80
                )
                self.logger.info(f"Updated job {self.job_id} status to training_setup_completed")
            except Exception as e:
                self.logger.error(f"Failed to update job status: {e}")

            # Hide the "Create Training Tasks" button if it exists
            if hasattr(self, 'create_tasks_button'):
                self.create_tasks_button.setVisible(False)

            # Fetch job details to get folder path
            try:
                job_info = self.api_gateway.get(f"jobs/{job_id}")
                job_folder = job_info.get("job_folder", "Unknown")
            except:
                job_folder = "Unknown"

            formatted_message = f"""
            Setup Complete!<br><br>
            <font color='#FF5733' size='+0'><b>{task_count}</b></font> training tasks created and saved for job:<br>
            <font color='#1a73e8' size='-1'><i>{job_id}</i></font><br><br>
            Time taken for task setup: <b>{total_time_taken}</b>
            """
            self.status_label.setText(formatted_message)
            self.status_label.setStyleSheet("color: green; font-size: 12pt; font-weight: bold; text-align: center;")

            # Add "Proceed to Training" button
            proceed_button = QPushButton("Proceed to Training")
            proceed_button.setStyleSheet("""
                background-color: #0b6337;
                font-weight: bold;
                padding: 10px 20px;
                color: white;
            """)
            proceed_button.clicked.connect(self.transition_to_training_gui)

            button_layout = QHBoxLayout()
            button_layout.addStretch(1)
            button_layout.addWidget(proceed_button, alignment=Qt.AlignCenter)
            button_layout.addStretch(1)
            self.main_layout.addLayout(button_layout)
        else:
            # Handle error case
            if isinstance(response_data, dict):
                error_message = response_data.get("detail", response_data.get("message", "An unknown error occurred."))
            else:
                error_message = "An unknown error occurred during setup."
                
            self.status_label.setText(f"Setup Failed: {error_message}")
            self.status_label.setStyleSheet("color: red; font-size: 12pt; font-weight: bold;")
            
            # Add retry button
            retry_button = QPushButton("Retry Setup")
            retry_button.setStyleSheet("""
                background-color: #d94a38;
                font-weight: bold;
                padding: 10px 20px;
                color: white;
            """)
            retry_button.clicked.connect(self.start_setup)
            
            button_layout = QHBoxLayout()
            button_layout.addStretch(1)
            button_layout.addWidget(retry_button, alignment=Qt.AlignCenter)
            button_layout.addStretch(1)
            self.main_layout.addLayout(button_layout)

    def transition_to_training_gui(self):
        try:
            # Update job status to indicate training setup is completed
            self.api_gateway.update_job_status(
                job_id=self.job_id,
                status="training_setup_completed",
                message="Training tasks created, ready for training execution",
                progress_percent=80
            )
            self.logger.info(f"Updated job {self.job_id} status to training_setup_completed")
            
            # The task list is now managed by the backend.
            # We can either fetch it again or assume the next GUI will.
            # For now, we'll just open the next GUI.
            self.training_gui = VEstimTrainingTaskGUI(api_gateway=self.api_gateway, job_id=self.job_id)
            current_geometry = self.geometry()
            self.training_gui.setGeometry(current_geometry)
            self.training_gui.show()
            self.logger.info("Transitioning to training task GUI.")
            self.close()
        except Exception as e:
            self.logger.error(f"Error while transitioning to the task screen: {e}")

    def update_elapsed_time(self):
        if self.timer_running:
            elapsed_time = time.time() - self.start_time
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.time_value_label.setText(f"{int(hours):02}h:{int(minutes):02}m:{int(seconds):02}s")
            
            # Schedule the next timer update only if the timer is still running
            if self.timer_running:
                QTimer.singleShot(1000, self.update_elapsed_time)


if __name__ == "__main__":
    # This part is for standalone testing and should be updated or removed
    # if it's no longer the primary way to run this GUI.
    pass
