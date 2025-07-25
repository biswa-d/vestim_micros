from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
import sys
import time
from vestim.gateway.src.training_setup_manager_qt import VEstimTrainingSetupManager
from vestim.gateway.src.optuna_setup_manager_qt import OptunaSetupManager
from vestim.gui.src.training_task_gui_qt import VEstimTrainingTaskGUI
from vestim.gateway.src.job_manager_qt import JobManager
from vestim.gateway.src.hyper_param_manager_qt import VEstimHyperParamManager
import logging

class SetupWorker(QThread):
    progress_signal = pyqtSignal(str, str, int)
    finished_signal = pyqtSignal()

    def __init__(self, job_manager, optuna_configs=None, base_params=None):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        if not job_manager:
            raise ValueError("JobManager instance is required.")
        self.job_manager = job_manager
        self.optuna_configs = optuna_configs
        self.base_params = base_params
        self.training_setup_manager = VEstimTrainingSetupManager(progress_signal=self.progress_signal, job_manager=self.job_manager)

    def run(self):
        self.logger.info("Starting training setup in a separate thread.")
        try:
            if self.optuna_configs:
                self.logger.info("Running setup with Optuna configurations.")
                optuna_setup_manager = OptunaSetupManager(job_manager=self.job_manager)
                optuna_setup_manager.setup_training_from_optuna(self.optuna_configs, self.base_params)
                self.training_setup_manager.training_tasks = optuna_setup_manager.get_task_list()
            else:
                self.logger.info("Running setup with grid search.")
                self.training_setup_manager.setup_training()
            
            self.logger.info("Training setup process completed.")
            self.finished_signal.emit()
        except Exception as e:
            self.logger.error(f"Error occurred during setup: {str(e)}")
            self.progress_signal.emit(f"Error occurred: {str(e)}", "", 0)

class VEstimTrainSetupGUI(QWidget):
    def __init__(self, params=None, optuna_configs=None, job_manager=None):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.optuna_configs = optuna_configs
        self.params = params

        if optuna_configs:
            self.param_list = [config['params'] for config in optuna_configs]
            self.is_multiple_configs = True
        else:
            self.param_list = [params] if params else []
            self.is_multiple_configs = False

        self.job_manager = job_manager if job_manager else JobManager()
        self.hyper_param_manager = VEstimHyperParamManager(job_manager=self.job_manager)
        self.timer_running = True
        self.auto_proceed_timer = QTimer(self)
        self.auto_proceed_timer.setSingleShot(True)
        self.auto_proceed_timer.timeout.connect(self.transition_to_training_gui)
        self.param_labels = {
            "LAYERS": "Layers", "HIDDEN_UNITS": "Hidden Units", "BATCH_SIZE": "Batch Size",
            "MAX_EPOCHS": "Max Epochs", "INITIAL_LR": "Initial Learning Rate",
            "LR_DROP_FACTOR": "LR Drop Factor", "LR_DROP_PERIOD": "LR Drop Period",
            "VALID_PATIENCE": "Validation Patience", "VALID_FREQUENCY": "Validation Freq",
            "LOOKBACK": "Lookback Sequence Length", "REPETITIONS": "Repetitions"
        }

        self.logger.info("Initializing VEstimTrainSetupGUI")
        self.build_gui()
        self.start_setup()

    def build_gui(self):
        if self.is_multiple_configs:
            self.setWindowTitle("VEstim - Setting Up Training (Optuna Optimized)")
            title_text = f"Building Models and Training Tasks\nwith {len(self.param_list)} Optuna-Optimized Configurations"
        else:
            self.setWindowTitle("VEstim - Setting Up Training (Grid Search)")
            title_text = "Building Models and Training Tasks\nwith Exhaustive Grid Search"
            
        self.setGeometry(100, 100, 1200, 800)
        
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        # --- Top Section ---
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 20, 0, 20)
        title_label = QLabel(title_text)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #0b6337; margin-bottom: 15px;")
        top_layout.addWidget(title_label)
        time_layout = QHBoxLayout()
        time_layout.setContentsMargins(0, 10, 0, 10)
        self.static_text_label = QLabel("Time Since Setup Started:")
        self.static_text_label.setStyleSheet("color: #555; font-size: 10pt;")
        self.time_value_label = QLabel("00h:00m:00s")
        self.time_value_label.setStyleSheet("color: #005878; font-size: 12pt; font-weight: bold;")
        time_layout.addStretch(1)
        time_layout.addWidget(self.static_text_label)
        time_layout.addWidget(self.time_value_label)
        time_layout.addStretch(1)
        top_layout.addLayout(time_layout)
        self.main_layout.addWidget(top_widget)

        # --- Hyperparameter Section ---
        self.hyperparam_frame = QFrame()
        self.hyperparam_frame.setObjectName("hyperparamFrame")
        self.hyperparam_frame.setStyleSheet("""
            #hyperparamFrame {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                background-color: #ffffff;
            }
        """)
        hyperparam_layout = QGridLayout()
        hyperparam_layout.setContentsMargins(30, 30, 30, 30)
        hyperparam_layout.setHorizontalSpacing(25)
        hyperparam_layout.setVerticalSpacing(20)
        self.display_hyperparameters(hyperparam_layout)
        self.hyperparam_frame.setLayout(hyperparam_layout)
        h_layout = QHBoxLayout()
        h_layout.addStretch(1)
        h_layout.addWidget(self.hyperparam_frame)
        h_layout.addStretch(1)
        self.main_layout.addLayout(h_layout)

        # Add stretch to push the bottom content down
        self.main_layout.addStretch(1)

        # --- Bottom Section ---
        self.bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(self.bottom_widget)
        bottom_layout.setContentsMargins(0, 20, 0, 20)
        self.status_label = QLabel("Setting up training...")
        self.status_label.setStyleSheet("color: green; font-size: 12pt; font-weight: bold;")
        self.status_label.setAlignment(Qt.AlignCenter)
        bottom_layout.addWidget(self.status_label)
        self.main_layout.addWidget(self.bottom_widget)

    def display_hyperparameters(self, layout):
        items = list(self.params.items())
        num_cols = 4
        num_rows = (len(items) + num_cols - 1) // num_cols

        for i, (param, value) in enumerate(items):
            row = i % num_rows
            col = (i // num_rows) * 2
            label_text = self.param_labels.get(param, param.replace("_", " ").title())
            value_str = str(value)
            param_label = QLabel(f"{label_text}:")
            param_label.setStyleSheet("font-size: 9pt; color: #333; font-weight: bold;")
            param_label.setAlignment(Qt.AlignRight | Qt.AlignTop)
            value_label = QLabel(value_str)
            value_label.setStyleSheet("font-size: 9pt; color: #005878;")
            value_label.setWordWrap(True)
            value_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            layout.addWidget(param_label, row, col)
            layout.addWidget(value_label, row, col + 1)

        for c in range(num_cols):
            layout.setColumnStretch(c * 2, 0)
            layout.setColumnStretch(c * 2 + 1, 1)

    def start_setup(self):
        if not self.is_multiple_configs:
            is_valid, error_message = self.hyper_param_manager.validate_for_grid_search(self.params)
            if not is_valid:
                QMessageBox.critical(self, "Validation Error", error_message)
                self.close()
                return

        self.logger.info("Starting training setup...")
        self.start_time = time.time()
        self.show()

        self.worker = SetupWorker(
            job_manager=self.job_manager,
            optuna_configs=self.optuna_configs,
            base_params=self.params
        )
        self.worker.progress_signal.connect(self.update_status)
        self.worker.finished_signal.connect(self.show_proceed_button)
        self.worker.start()
        self.update_elapsed_time()

    def update_status(self, message, path="", task_count=None):
        task_message = f"{task_count} training tasks created,\n" if task_count else ""
        formatted_message = f"{task_message}{message}\n{path}" if path else f"{task_message}{message}"
        self.status_label.setText(formatted_message)
        self.status_label.setStyleSheet("color: green; font-size: 12pt; font-weight: bold;")

    def show_proceed_button(self):
        self.logger.info("Training setup complete, showing proceed button.")
        self.timer_running = False
        self.worker.quit()
        self.worker.wait()

        elapsed_time = time.time() - self.start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        total_time_taken = f"{int(hours):02}h:{int(minutes):02}m:{int(seconds):02}s"

        task_list = self.worker.training_setup_manager.get_task_list() 
        task_count = len(task_list)
        job_folder = self.job_manager.get_job_folder()

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
        self.auto_proceed_timer.start(60000)

        bottom_layout = self.bottom_widget.layout()
        button_layout = QHBoxLayout()
        button_layout.addStretch(1)
        button_layout.addWidget(proceed_button, alignment=Qt.AlignCenter)
        button_layout.addStretch(1)
        bottom_layout.addLayout(button_layout)

    def transition_to_training_gui(self):
        try:
            self.auto_proceed_timer.stop()
            task_list = self.worker.training_setup_manager.get_task_list()
            if not task_list:
                print("No tasks to train.")
                return

            first_task_params = task_list[0]['hyperparams']
            
            self.training_gui = VEstimTrainingTaskGUI(job_manager=self.job_manager, task_list=task_list, params=first_task_params)
            
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
    params = {}
    gui = VEstimTrainSetupGUI(params)
    gui.show()
    sys.exit(app.exec_())
