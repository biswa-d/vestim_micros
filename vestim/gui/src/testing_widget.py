from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, 
    QTreeWidget, QTreeWidgetItem, QProgressBar, QDialog, QMessageBox, 
    QGridLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont
import os, time
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import logging
import numpy as np
from vestim.gui.src.api_gateway import APIGateway

class TestingThread(QThread):
    result_signal = pyqtSignal(dict)
    testing_complete_signal = pyqtSignal()

    def __init__(self, job_id, api_gateway):
        super().__init__()
        self.job_id = job_id
        self.api_gateway = api_gateway
        self.stop_flag = False

    def run(self):
        try:
            self.api_gateway.post(f"jobs/{self.job_id}/test")
            while not self.stop_flag:
                status = self.api_gateway.get(f"jobs/{self.job_id}/testing_status")
                if not status or status.get("status") in ["complete", "error", "stopped"]:
                    break
                self.result_signal.emit(status)
                time.sleep(2)
            self.testing_complete_signal.emit()
        except Exception as e:
            pass
        finally:
            self.quit()

class VEstimTestingWidget(QWidget):
    testing_complete = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.job_id = None
        self.api_gateway = APIGateway()
        self.hyper_params = {}
        self.sl_no_counter = 1
        self.initUI()

    def set_job_id(self, job_id):
        self.job_id = job_id
        self.load_testing_data()

    def initUI(self):
        self.main_layout = QVBoxLayout(self)
        title_label = QLabel("Model Testing")
        title_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(title_label)

        self.hyperparam_frame = QWidget()
        self.main_layout.addWidget(self.hyperparam_frame)
        
        self.time_label = QLabel("Testing Time: 00h:00m:00s")
        self.time_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.time_label)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(9)
        self.tree.setHeaderLabels(["Sl.No", "Task ID", "Model", "File Name", "#W&Bs", "RMS Error", "Max Error", "MAPE (%)", "R²", "Plot"])
        self.main_layout.addWidget(self.tree)

        self.status_label = QLabel("Preparing test data...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.status_label)

        self.progress = QProgressBar(self)
        self.main_layout.addWidget(self.progress)

        self.done_button = QPushButton("Done", self)
        self.done_button.clicked.connect(self.finish_testing)
        self.main_layout.addWidget(self.done_button)
        self.done_button.hide()

    def load_testing_data(self):
        testing_info = self.api_gateway.get(f"jobs/{self.job_id}/testing_info")
        if testing_info:
            self.hyper_params = testing_info.get("hyper_params", {})
            self.display_hyperparameters(self.hyper_params)
            self.start_testing()
        else:
            self.update_status("Failed to load testing data.")

    def display_hyperparameters(self, params):
        if not params:
            return
        if self.hyperparam_frame.layout() is not None:
            while self.hyperparam_frame.layout().count():
                item = self.hyperparam_frame.layout().takeAt(0)
                widget = item.widget()
                if widget:
                    widget.setParent(None)
        grid_layout = QGridLayout()
        self.hyperparam_frame.setLayout(grid_layout)
        # Simplified display
        row = 0
        for key, value in params.items():
            grid_layout.addWidget(QLabel(f"{key}: "), row, 0)
            grid_layout.addWidget(QLabel(str(value)), row, 1)
            row += 1

    def start_testing(self):
        self.update_status("Testing in progress...")
        self.start_time = time.time()
        self.update_elapsed_time()
        self.testing_thread = TestingThread(self.job_id, self.api_gateway)
        self.testing_thread.result_signal.connect(self.add_result_row)
        self.testing_thread.testing_complete_signal.connect(self.all_tests_completed)
        self.testing_thread.start()

    def update_status(self, message):
        self.status_label.setText(message)

    def update_elapsed_time(self):
        if self.start_time:
            elapsed_time = time.time() - self.start_time
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.time_label.setText(f"Testing Time: {int(hours):02}h:{int(minutes):02}m:{int(seconds):02}s")
            QTimer.singleShot(1000, self.update_elapsed_time)

    def add_result_row(self, result):
        task_data = result.get('task_completed')
        if task_data:
            # Simplified result row addition
            row = QTreeWidgetItem([
                str(self.sl_no_counter),
                str(task_data.get("task_id", "N/A")),
                str(task_data.get("model", "Unknown")),
                str(task_data.get("file_name", "Unknown")),
                str(task_data.get("#params", "N/A")),
                f"{task_data.get('rms_error_mv', 0):.2f}",
                f"{task_data.get('max_abs_error_mv', 0):.2f}",
                f"{task_data.get('mape_percent', 0):.2f}",
                f"{task_data.get('r2', 0):.4f}"
            ])
            self.sl_no_counter += 1
            self.tree.addTopLevelItem(row)

    def all_tests_completed(self):
        self.update_status("All tests completed.")
        self.done_button.show()

    def finish_testing(self):
        self.testing_complete.emit(self.job_id)

if __name__ == '__main__':
    app = QApplication([])
    widget = VEstimTestingWidget()
    widget.set_job_id("dummy_job_id")
    widget.show()
    app.exec_()