from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
import sys
import time
from src.gateway.src.training_setup_manager_qt import VEstimTrainingSetupManager
from src.gui.src.training_task_gui_qt import VEstimTrainingTaskGUI
from src.gateway.src.job_manager import JobManager

class SetupWorker(QThread):
    progress_signal = pyqtSignal(str, str, int)  # Signal to update the status in the main GUI
    finished_signal = pyqtSignal()  # Signal when the setup is finished

    def __init__(self, job_manager):
        super().__init__()
        self.job_manager = job_manager
        # Directly use VEstimTrainingSetupManager(), __new__ ensures singleton
        self.training_setup_manager = VEstimTrainingSetupManager(progress_signal=self.progress_signal, job_manager=self.job_manager)

    def run(self):
        print("Starting training setup in a separate thread...")
        try:
            # Perform the training setup
            self.training_setup_manager.setup_training()
            print("Training setup started successfully!")

            # Get the number of training tasks created
            task_count = len(self.training_setup_manager.get_task_list())

            # Emit a signal to update the status
            self.progress_signal.emit(
                "Task summary saved in the job folder",
                self.job_manager.get_job_folder(),
                task_count
            )

            # Emit a signal when finished
            self.finished_signal.emit()

        except Exception as e:
            # Emit the error message via the progress signal
            self.progress_signal.emit(f"Error occurred: {str(e)}", "", 0)

class VEstimTrainSetupGUI(QWidget):
    def __init__(self, params):
        super().__init__()

        self.params = params
        self.job_manager = JobManager()  # Initialize the JobManager singleton instance directly
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
        self.build_gui()
        # Start the setup process
        self.start_setup()

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
        hyperparam_layout = QGridLayout()
        self.display_hyperparameters(hyperparam_layout)
        self.hyperparam_frame.setLayout(hyperparam_layout)
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
        self.start_time = time.time()

        # Ensure the window is fully built and ready
        self.show()

        # Move the training setup to a separate thread
        self.worker = SetupWorker(self.job_manager)
        self.worker.progress_signal.connect(self.update_status)
        self.worker.finished_signal.connect(self.show_proceed_button)

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

    def show_proceed_button(self):
        print("Training setup complete! Enabling proceed button...")
        # Stop the timer
        self.timer_running = False

        # Cleanup the worker thread
        self.worker.quit()
        self.worker.wait()

        # Calculate the total elapsed time
        elapsed_time = time.time() - self.start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        total_time_taken = f"{int(hours):02}h:{int(minutes):02}m:{int(seconds):02}s"

        # Get task count and job folder path
        task_list = self.worker.training_setup_manager.get_task_list() 
        task_count = len(task_list)
        job_folder = self.job_manager.get_job_folder()

        # Final update: tasks created, job folder, and time taken
        formatted_message = f"""
        Setup Complete!<br><br>
        <font color='#FF5733' size='+0'><b>{task_count}</b></font> training tasks created and saved to:<br>
        <font color='#1a73e8' size='-1'><i>{job_folder}</i></font><br><br>
        Time taken for task setup: <b>{total_time_taken}</b>
        """
        self.status_label.setText(formatted_message)

        # Show the proceed button when training setup is done
        proceed_button = QPushButton("Proceed to Training")
        proceed_button.setStyleSheet("""
            background-color: #0b6337; 
            font-weight: bold; 
            padding: 10px 20px; 
            color: white;
        """)
        proceed_button.adjustSize()  # Make sure the button size wraps text appropriately
        proceed_button.clicked.connect(self.transition_to_training_gui)

        # Center the button and control its layout
        button_layout = QHBoxLayout()
        button_layout.addStretch(1)
        button_layout.addWidget(proceed_button, alignment=Qt.AlignCenter)
        button_layout.addStretch(1)
        self.main_layout.addLayout(button_layout)

    def transition_to_training_gui(self):
        try:
            # Initialize the task screen
            task_list = self.worker.training_setup_manager.get_task_list()  # Get the task list
            self.training_gui = VEstimTrainingTaskGUI(task_list, self.params)
            # Get the current window's geometry and set it for the new window
            current_geometry = self.geometry()
            self.training_gui.setGeometry(current_geometry) 
            self.training_gui.show()   
            # After the new window is successfully displayed, close the current one
            self.close()
        except Exception as e:
            print(f"Error while transitioning to the task screen: {e}")

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
    app = QApplication(sys.argv)
    params = {}  # Assuming you pass hyperparameters here
    gui = VEstimTrainSetupGUI(params)
    gui.show()
    sys.exit(app.exec_())
