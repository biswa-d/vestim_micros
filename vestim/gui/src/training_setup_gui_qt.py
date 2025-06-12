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

    def run(self):
        self.logger.info(f"Starting training setup for job {self.job_id} via API call.")
        try:
            # Sending an empty JSON object with the POST request
            response = self.api_gateway.post(f"jobs/{self.job_id}/setup-training", json={})
            self.finished_signal.emit(response)
        except Exception as e:
            self.logger.error(f"Error during training setup API call: {e}")
            self.progress_signal.emit(f"Error occurred: {str(e)}", "", 0)

class VEstimTrainSetupGUI(QWidget):
    def __init__(self, job_id: str, api_gateway: APIGateway):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.job_id = job_id
        self.api_gateway = api_gateway
        self.params = {}
        self.timer_running = True
        self.param_labels = {
            "LAYERS": "Layers",
            "HIDDEN_UNITS": "Hidden Units",
            "BATCH_SIZE": "Batch Size",
            "MAX_EPOCHS": "Max Epochs",
            "INITIAL_LR": "Initial Learning Rate",
            "LR_DROP_FACTOR": "LR Drop Factor",
            "LR_DROP_PERIOD": "LR Drop Period",
            "VALID_PATIENCE": "Validation Patience",
            "ValidFrequency": "Validation Frequency",
            "LOOKBACK": "Lookback Sequence Length",
            "REPETITIONS": "Repetitions"
        }

        self.logger.info(f"Initializing VEstimTrainSetupGUI for job_id: {self.job_id}")
        self.build_gui()
        self.fetch_hyperparameters_and_start()

    def fetch_hyperparameters_and_start(self):
        try:
            job_details = self.api_gateway.get_job(self.job_id)
            self.params = job_details.get("details", {}).get("hyperparameters", {})
            if not self.params:
                self.status_label.setText("Could not load hyperparameters.")
                return
            self.display_hyperparameters(self.hyperparam_layout)
            self.start_setup()
        except Exception as e:
            self.status_label.setText(f"Error fetching parameters: {e}")

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
        print("Starting training setup...")
        self.logger.info("Starting training setup...")
        self.start_time = time.time()

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
        self.logger.info(f"Training setup finished with response: {response_data}")
        self.timer_running = False
        self.worker.quit()
        self.worker.wait()

        if response_data and "task_count" in response_data:
            task_count = response_data.get("task_count", 0)
            job_folder = response_data.get("job_folder", "N/A")
            
            elapsed_time = time.time() - self.start_time
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            total_time_taken = f"{int(hours):02}h:{int(minutes):02}m:{int(seconds):02}s"

            formatted_message = f"""
            Setup Complete!<br><br>
            <font color='#FF5733' size='+0'><b>{task_count}</b></font> training tasks created and saved to:<br>
            <font color='#1a73e8' size='-1'><i>{job_folder}</i></font><br><br>
            Time taken for task setup: <b>{total_time_taken}</b>
            """
            self.status_label.setText(formatted_message)

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
            error_message = response_data.get("detail", "An unknown error occurred.") if response_data else "An unknown error occurred."
            self.status_label.setText(f"Setup Failed: {error_message}")
            self.status_label.setStyleSheet("color: red; font-size: 12pt; font-weight: bold;")

    def transition_to_training_gui(self):
        try:
            # The task list is now managed by the backend.
            # We can either fetch it again or assume the next GUI will.
            # For now, we'll just open the next GUI.
            self.training_gui = VEstimTrainingTaskGUI(job_id=self.job_id, api_gateway=self.api_gateway)
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
