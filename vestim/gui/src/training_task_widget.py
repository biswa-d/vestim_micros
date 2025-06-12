from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QHBoxLayout, QVBoxLayout, QPushButton, QFrame, QTextEdit, QGridLayout, QGroupBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
import torch
import json, time
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import logging
from vestim.gui.src.api_gateway import APIGateway

class VEstimTrainingTaskWidget(QWidget):
    training_complete = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.job_id = None
        self.api_gateway = APIGateway()
        self.task_list = []
        self.current_task_index = 0
        self.timer_running = True
        self.initUI()

    def set_job_id(self, job_id):
        self.job_id = job_id
        try:
            tasks_response = self.api_gateway.get(f"jobs/{self.job_id}/tasks")
            self.task_list = tasks_response.get("tasks", []) if tasks_response else []
        except Exception as e:
            self.task_list = []
        
        if self.task_list:
            self.current_task_index = 0
            self.build_gui(self.task_list[self.current_task_index])
            self.start_task_processing()
        else:
            # Handle case with no tasks
            pass

    def initUI(self):
        self.main_layout = QVBoxLayout(self)
        self.setLayout(self.main_layout)

    def build_gui(self, task):
        # Clear previous widgets
        while self.main_layout.count():
            item = self.main_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        title_label = QLabel("Training Model")
        title_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(title_label)

        self.hyperparam_frame = QFrame(self)
        self.hyperparam_frame.setLayout(QGridLayout())
        self.main_layout.addWidget(self.hyperparam_frame)
        self.display_hyperparameters(task['hyperparams'])

        self.status_label = QLabel("Starting training...")
        self.main_layout.addWidget(self.status_label)

        self.setup_time_and_plot(task)

        log_group = QGroupBox("Training Log")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        self.main_layout.addWidget(log_group)

        self.stop_button = QPushButton("Stop Training")
        self.stop_button.clicked.connect(self.stop_training)
        self.main_layout.addWidget(self.stop_button)

        self.proceed_button = QPushButton("Proceed to Testing")
        self.proceed_button.hide()
        self.proceed_button.clicked.connect(self.finish_training)
        self.main_layout.addWidget(self.proceed_button)

    def display_hyperparameters(self, task_params):
        layout = self.hyperparam_frame.layout()
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        
        # Simplified display for brevity
        row = 0
        for key, value in task_params.items():
            layout.addWidget(QLabel(f"{key}:"), row, 0)
            layout.addWidget(QLabel(str(value)), row, 1)
            row += 1

    def setup_time_and_plot(self, task):
        time_layout = QHBoxLayout()
        self.static_text_label = QLabel("Current Task Time:")
        self.time_value_label = QLabel("00h:00m:00s")
        time_layout.addWidget(self.static_text_label)
        time_layout.addWidget(self.time_value_label)
        self.main_layout.addLayout(time_layout)

        fig = Figure(figsize=(6, 2.5), dpi=100)
        self.ax = fig.add_subplot(111)
        self.train_line, = self.ax.plot([], [], label='Train Loss')
        self.valid_line, = self.ax.plot([], [], label='Validation Loss')
        self.ax.legend()
        self.canvas = FigureCanvas(fig)
        self.main_layout.addWidget(self.canvas)

    def start_task_processing(self):
        self.status_label.setText(f"Task {self.current_task_index + 1}/{len(self.task_list)} is running.")
        self.start_time = time.time()
        self.api_gateway.post(f"jobs/{self.job_id}/train", json={"task_id": self.task_list[self.current_task_index]['task_id']})
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_status)
        self.timer.start(5000)
        self.update_elapsed_time()

    def update_elapsed_time(self):
        if self.timer_running:
            elapsed_time = time.time() - self.start_time
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.time_value_label.setText(f" {int(hours):02}h:{int(minutes):02}m:{int(seconds):02}s")
            QTimer.singleShot(1000, self.update_elapsed_time)

    def refresh_status(self):
        try:
            status = self.api_gateway.get(f"jobs/{self.job_id}/status")
            if not status:
                return

            current_status = status.get("status", "unknown")
            training_status = status.get("detailed_status", {}).get("training", {})
            history = training_status.get("history", [])
            
            if history:
                self.update_gui_from_history(history)

            if current_status == "training_complete":
                self.task_completed()

        except Exception as e:
            self.logger.error(f"Error refreshing status: {e}")

    def update_gui_from_history(self, history):
        # Simplified update logic
        train_loss = [item.get('train_loss') for item in history]
        valid_loss = [item.get('valid_loss') for item in history]
        epochs = list(range(1, len(train_loss) + 1))
        
        self.train_line.set_data(epochs, train_loss)
        self.valid_line.set_data(epochs, valid_loss)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()

    def stop_training(self):
        self.timer_running = False
        self.timer.stop()
        self.api_gateway.post(f"jobs/{self.job_id}/stop")
        self.status_label.setText("Training stopped.")

    def task_completed(self):
        self.current_task_index += 1
        if self.current_task_index < len(self.task_list):
            self.build_gui(self.task_list[self.current_task_index])
            self.start_task_processing()
        else:
            self.status_label.setText("All training tasks completed.")
            self.stop_button.hide()
            self.proceed_button.show()
            self.timer.stop()

    def finish_training(self):
        self.training_complete.emit(self.job_id)

if __name__ == '__main__':
    app = QApplication([])
    widget = VEstimTrainingTaskWidget()
    # A dummy job_id is required for the widget to initialize.
    widget.set_job_id("dummy_job_id")
    widget.show()
    app.exec_()