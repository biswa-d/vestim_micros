import requests
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
import sys, json, os
import time
from vestim.gui.src.training_task_gui_qt_flask import VEstimTrainingTaskGUI
import logging

class VEstimTrainSetupGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)  # Set up logger
        self.params = None  # Initialize params to None
        self.timer_running = True  # Ensure this flag is initialized in __init__
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

        # Setup GUI
        self.logger.info("Initializing VEstimTrainSetupGUI")
        self.build_gui()
        # Start the setup process
        self.start_setup()

    def fetch_hyper_params(self):
            """Fetches and stores the hyperparameters from the Hyper Param Manager API."""
            if self.params is None:
                try:
                    response_params = requests.get("http://localhost:5000/hyper_param_manager/get_params")
                    if response_params.status_code == 200:
                        self.params = response_params.json()
                    else:
                        raise Exception("Failed to fetch hyperparameters")
                except Exception as e:
                    self.logger.error(f"Error fetching hyperparameters: {str(e)}")
                    raise e

    def build_gui(self):
        self.setWindowTitle("VEstim - Setting Up Training")
        self.setMinimumSize(900, 600)
        self.setMaximumSize(900, 600)

        self.main_layout = QVBoxLayout()
        title_label = QLabel("Building LSTM Models and Training Tasks\nwith Hyperparameter Set")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: #3a3a3a;")
        self.main_layout.addWidget(title_label)

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

        self.hyperparam_frame = QFrame()
        hyperparam_layout = QGridLayout()
        self.display_hyperparameters(hyperparam_layout)
        self.hyperparam_frame.setLayout(hyperparam_layout)
        self.main_layout.addWidget(self.hyperparam_frame)

        self.status_label = QLabel("Setting up training...")
        self.status_label.setStyleSheet("color: green; font-size: 12pt; font-weight: bold;")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.status_label)

        self.setLayout(self.main_layout)

    def display_hyperparameters(self, layout):
        self.fetch_hyper_params()
        items = list(self.params.items())
        for i, (param, value) in enumerate(items):
            row = i // 2
            col = (i % 2) * 2

            label_text = self.param_labels.get(param, param)
            value_str = str(value)
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

        self.show()

        # Directly make the Flask API call to set up training
        try:
            response = requests.post("http://localhost:5000/training_setup/setup_training")
            if response.status_code == 200:
                data = response.json()
                task_count = data.get("task_count", 0)
                job_folder = data.get("job_folder", "")

                self.update_status(
                    "Task summary saved in the job folder",
                    job_folder,
                    task_count
                )

                self.show_proceed_button()

            else:
                error_message = response.json().get('error', 'Unknown error occurred during setup')
                self.update_status(f"Error in training setup: {error_message}", "", 0)

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error occurred while communicating with the server: {str(e)}")
            self.update_status(f"Error occurred: {str(e)}", "", 0)

        self.update_elapsed_time()


    def update_status(self, message, path="", task_count=None):
        task_message = f"{task_count} training tasks created,\n" if task_count else ""
        formatted_message = f"{task_message}{message}\n{path}" if path else f"{task_message}{message}"
        self.status_label.setText(formatted_message)
        self.status_label.setStyleSheet("color: green; font-size: 12pt; font-weight: bold;")

    def show_proceed_button(self):
        self.logger.info("Training setup complete, showing proceed button.")
        print("Training setup complete! Enabling proceed button...")
        self.timer_running = False


        elapsed_time = time.time() - self.start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        total_time_taken = f"{int(hours):02}h:{int(minutes):02}m:{int(seconds):02}s"

        # Fetch tasks and job folder from Flask API instead of job_manager
        response = requests.get("http://localhost:5000/training_setup/get_tasks")
        if response.status_code == 200:
            task_list = response.json()
            task_count = len(task_list)  # Ensure len works on the list
        else:
            print(f"Error fetching task list: {response.json().get('error', 'Unknown error')}")
            return

        job_response = requests.get("http://localhost:5000/job_manager/get_job_folder")
        if job_response.status_code == 200:
            job_folder = job_response.json().get('job_folder')
        else:
            print(f"Error fetching job folder: {job_response.json().get('error', 'Unknown error')}")
            return

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
        proceed_button.adjustSize()
        proceed_button.clicked.connect(self.transition_to_training_gui)

        button_layout = QHBoxLayout()
        button_layout.addStretch(1)
        button_layout.addWidget(proceed_button, alignment=Qt.AlignCenter)
        button_layout.addStretch(1)
        self.main_layout.addLayout(button_layout)


    def transition_to_training_gui(self):
        try:
            # Update tool state to reflect the transition to the training task GUI
            tool_state = {
                "current_state": "training",
                "current_screen": "VEstimTrainingTaskGUI"
            }
            with open("vestim/tool_state.json", "w") as f:
                json.dump(tool_state, f)

            # Transition to the training task GUI
            self.training_gui = VEstimTrainingTaskGUI()
            current_geometry = self.geometry()
            self.training_gui.setGeometry(current_geometry)
            self.training_gui.show()
            self.logger.info("Transitioning to training task GUI.")
            self.close()
        except Exception as e:
            print(f"Error while transitioning to the task screen: {e}")

    def update_elapsed_time(self):
        if self.timer_running:
            elapsed_time = time.time() - self.start_time
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.time_value_label.setText(f"{int(hours):02}h:{int(minutes):02}m:{int(seconds):02}s")
            if self.timer_running:
                QTimer.singleShot(1000, self.update_elapsed_time)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    params = {}  # Assuming you pass hyperparameters here
    gui = VEstimTrainSetupGUI(params)
    gui.show()
    sys.exit(app.exec_())
