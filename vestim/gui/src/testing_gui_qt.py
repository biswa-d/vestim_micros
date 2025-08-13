# ---------------------------------------------------------------------------------
# Author: Biswanath Dehury
# Date: 2023-03-02
# Version: 1.0.0
# Description: Description of the script
# Description: 
# This is the batchtesting without padding implementation for the unscaled data where the batch-size is used for testloader preparation but the model is tested
# one sequence at a time like a running window. The first part of the test file is padded with data to avoid the size mismatch and get the final prediction the same
# shape as the test file.
#
# Copyright (c) 2024 Biswanath Dehury, Dr. Phil Kollmeyer's Battery Lab at McMaster University
# ---------------------------------------------------------------------------------


from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, 
    QWidget, QTreeWidget, QTreeWidgetItem, QProgressBar, QDialog, QMessageBox, 
    QGridLayout, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QUrl
from PyQt5.QtGui import QFont, QDesktopServices, QPixmap
import os, sys, time
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from queue import Queue, Empty
import logging
import matplotlib.pyplot as plt
import numpy as np

# Import your services
from vestim.gateway.src.testing_manager_qt import VEstimTestingManager
from vestim.gateway.src.job_manager_qt import JobManager
from vestim.gateway.src.training_setup_manager_qt import VEstimTrainingSetupManager
from vestim.gateway.src.hyper_param_manager_qt import VEstimHyperParamManager
from vestim.utils.data_cleanup_manager import DataCleanupManager

class TestingThread(QThread):
    update_status_signal = pyqtSignal(str)
    result_signal = pyqtSignal(dict)
    testing_complete_signal = pyqtSignal()

    def __init__(self, testing_manager, queue):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.testing_manager = testing_manager
        self.queue = queue
        self.stop_flag = False

    def run(self):
        try:
            self.testing_manager.start_testing(self.queue)
            while not self.stop_flag:
                try:
                    result = self.queue.get(timeout=1)
                    if result:
                        if 'all_tasks_completed' in result:
                            self.testing_complete_signal.emit()
                            self.stop_flag = True
                        else:
                            self.result_signal.emit(result)
                except Empty:
                    continue
        except Exception as e:
            self.update_status_signal.emit(f"Error: {str(e)}")
        finally:
            print("Testing thread is stopping...")
            self.quit()


class VEstimTestingGUI(QMainWindow):
    def __init__(self, job_manager=None, params=None, task_list=None, training_results=None, testing_manager=None):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.job_manager = job_manager if job_manager else JobManager()
        self.params = params
        self.job_folder = self.job_manager.get_job_folder() if self.job_manager else None
        self.training_results = training_results if training_results is not None else {}
        self.testing_manager = testing_manager if testing_manager else VEstimTestingManager(job_manager=self.job_manager, params=self.params, task_list=task_list, training_results=self.training_results)
        self.hyper_param_manager = VEstimHyperParamManager(job_manager=self.job_manager)
        self.training_setup_manager = VEstimTrainingSetupManager(job_manager=self.job_manager)
        self.data_cleanup_manager = DataCleanupManager()

        self.param_labels = {
            "LAYERS": "Layers", "HIDDEN_UNITS": "Hidden Units", "BATCH_SIZE": "Batch Size",
            "MAX_EPOCHS": "Max Epochs", "INITIAL_LR": "Initial Learning Rate",
            "LR_DROP_FACTOR": "LR Drop Factor", "LR_DROP_PERIOD": "LR Drop Period",
            "VALID_PATIENCE": "Validation Patience", "VALID_FREQUENCY": "Validation Freq",
            "LOOKBACK": "Lookback Sequence Length", "REPETITIONS": "Repetitions",
            "NUM_WORKERS": "# CPU Threads", "PIN_MEMORY": "Fast CPU-GPU Transfer",
            "PREFETCH_FACTOR": "Batch Pre-loading", "USE_CUDA_GRAPHS": "CUDA Graphs"
        }

        self.queue = Queue()
        self.timer_running = True
        self.start_time = None
        self.testing_thread = None
        self.results_list = []
        self.hyper_params = {}
        self.sl_no_counter = 1

        self.initUI()
        self.start_testing()

    def initUI(self):
        self.setWindowTitle("VEstim Tool - Model Testing")
        self.setGeometry(100, 100, 1200, 800)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Add a global stylesheet for disabled buttons
        self.setStyleSheet("""
            QPushButton:disabled {
                background-color: #d3d3d3;
                color: #a9a9a9;
            }
        """)

        title_label = QLabel("Testing Models")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #0b6337; margin-bottom: 15px;")
        self.main_layout.addWidget(title_label)

        self.hyperparam_frame = QFrame()
        self.hyperparam_frame.setObjectName("hyperparamFrame")
        self.hyperparam_frame.setStyleSheet("""
            #hyperparamFrame {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                background-color: #ffffff;
            }
        """)
        self.main_layout.addWidget(self.hyperparam_frame)
        self.hyper_params = self.params
        self.display_hyperparameters(self.hyper_params)
        
        self.time_label = QLabel("Testing Time: 00h:00m:00s")
        self.time_label.setFont(QFont("Helvetica", 10))
        self.time_label.setStyleSheet("color: blue; margin-top: 10px;")
        self.time_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.time_label)

        result_summary_label = QLabel("Testing Result Summary")
        result_summary_label.setAlignment(Qt.AlignCenter)
        result_summary_label.setStyleSheet("font-size: 14px; font-weight: bold; margin-top: 10px;")
        self.main_layout.addWidget(result_summary_label)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(13)
        self.tree.setHeaderLabels(["Sl.No", "Model", "Task", "File Name", "#W&Bs", "Best Train Loss", "Best Valid Loss", "Epochs Trained", "Test RMSE", "Test MAXE", "MAPE (%)", "R²", "Plot"])
        self.tree.setColumnWidth(0, 50)
        self.tree.setColumnWidth(1, 120)
        self.tree.setColumnWidth(2, 220)
        self.tree.setColumnWidth(3, 150)
        self.tree.setColumnWidth(4, 70)
        self.tree.setColumnWidth(5, 100)
        self.tree.setColumnWidth(6, 100)
        self.tree.setColumnWidth(7, 100)
        self.tree.setColumnWidth(8, 100)
        self.tree.setColumnWidth(9, 100)
        self.tree.setColumnWidth(10, 70)
        self.tree.setColumnWidth(11, 60)
        self.tree.setColumnWidth(12, 100)
        self.main_layout.addWidget(self.tree)

        self.status_label = QLabel("Preparing test data...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 12px; font-weight: bold; color: #004d99;")
        self.main_layout.addWidget(self.status_label)

        self.progress = QProgressBar(self)
        self.progress.setMaximum(100)
        self.progress.setValue(0)
        self.progress.setStyleSheet("""
            QProgressBar { border: 1px solid #87CEEB; border-radius: 5px; text-align: center; background-color: #f0f8ff; }
            QProgressBar::chunk { background-color: #87CEEB; border-radius: 4px; }
        """)
        self.main_layout.addWidget(self.progress)

        self.open_results_button = QPushButton("Open Job Folder", self)
        self.open_results_button.setStyleSheet("""
            background-color: #0b6337; font-weight: bold; padding: 10px 20px; color: white;
        """)
        self.open_results_button.setFixedHeight(40)
        self.open_results_button.setMinimumWidth(150)
        self.open_results_button.setMaximumWidth(300)
        self.open_results_button.clicked.connect(self.open_job_folder)
        
        open_button_layout = QHBoxLayout()
        open_button_layout.addStretch(1)
        open_button_layout.addWidget(self.open_results_button, alignment=Qt.AlignCenter)
        open_button_layout.addStretch(1)
        open_button_layout.setContentsMargins(50, 20, 50, 20)
        self.main_layout.addLayout(open_button_layout)
        self.open_results_button.hide()

    def open_job_folder(self):
        job_folder = self.job_folder
        if job_folder and os.path.exists(job_folder):
            QDesktopServices.openUrl(QUrl.fromLocalFile(job_folder))
        else:
            QMessageBox.critical(self, "Error", f"Results folder not found: {job_folder}")

    def display_hyperparameters(self, params):
        if not params:
            return

        layout = self.hyperparam_frame.layout()
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
        
        grid_layout = QGridLayout()
        self.hyperparam_frame.setLayout(grid_layout)
        grid_layout.setContentsMargins(20, 20, 20, 20)
        grid_layout.setHorizontalSpacing(15)
        grid_layout.setVerticalSpacing(10)

        items = list(params.items())
        num_cols = 4
        num_rows = (len(items) + num_cols - 1) // num_cols

        for i, (param, value) in enumerate(items):
            row = i % num_rows
            col = (i // num_rows) * 2

            label_text = self.param_labels.get(param, param.replace("_", " ").title())
            value_str = str(value)

            param_label = QLabel(f"{label_text}:")
            param_label.setStyleSheet("font-size: 9pt; color: #333;")
            param_label.setAlignment(Qt.AlignRight | Qt.AlignTop)

            value_label = QLabel(value_str)
            value_label.setStyleSheet("font-size: 9pt; color: #000000; font-weight: bold;")
            value_label.setWordWrap(True)
            value_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)

            grid_layout.addWidget(param_label, row, col)
            grid_layout.addWidget(value_label, row, col + 1)

        for c in range(num_cols):
            grid_layout.setColumnStretch(c * 2, 0)
            grid_layout.setColumnStretch(c * 2 + 1, 1)

    def update_status(self, message):
        self.status_label.setText(message)

    def add_result_row(self, result):
        task_data_for_log = result.get('task_completed', {})
        log_summary = (
            f"Sl.No: {task_data_for_log.get('sl_no', 'N/A')}, "
            f"Model: {task_data_for_log.get('model_name', 'N/A')}, "
            f"File: {task_data_for_log.get('file_name', 'N/A')}"
        )
        self.logger.info(f"Adding result row: {log_summary}")

        if 'task_error' in result:
            print(f"Error in task: {result['task_error']}")
            return

        task_data = result.get('task_completed')

        if task_data:
            save_dir = task_data.get("saved_dir", "")
            model_name = task_data.get("model_name", "N/A")
            task_name = task_data.get("task_name", "N/A")
            task_id = task_data.get("task_info", {}).get("task_id", "")
            file_name = task_data.get("file_name", "Unknown File")
            num_learnable_params = str(task_data.get("#params", "N/A"))
            best_train_loss = task_data.get("best_train_loss", "N/A")
            best_valid_loss = task_data.get("best_valid_loss", "N/A")
            completed_epochs = task_data.get("completed_epochs", "N/A")
            
            target_column_name = task_data.get("target_column", "")
            predictions_file = task_data.get("predictions_file", "")

            unit_suffix = ""
            unit_display = ""
            if "voltage" in target_column_name.lower():
                unit_suffix = "_mv"
                unit_display = "(mV)"
            elif "soc" in target_column_name.lower():
                unit_suffix = "_percent"
                unit_display = "(% SOC)"
            elif "temperature" in target_column_name.lower() or "temp" in target_column_name.lower():
                unit_suffix = "_degC"
                unit_display = "(Deg C)"
            
            if 'unit_display' in task_data:
                unit_display = task_data['unit_display']
            
            if self.sl_no_counter == 1:
                current_headers = [self.tree.headerItem().text(i) for i in range(self.tree.columnCount())]
                current_headers[8] = f"Test RMSE {unit_display}"
                current_headers[9] = f"Test MAXE {unit_display}"
                self.tree.setHeaderLabels(current_headers)

            rms_key = f'rms_error{unit_suffix}'
            max_error_key = f'max_abs_error{unit_suffix}'
            
            rms_error_val = task_data.get(rms_key, 'N/A')
            max_error_val = task_data.get(max_error_key, task_data.get('max_error_mv', 'N/A'))
            mape = task_data.get('mape_percent', 'N/A')
            r2 = task_data.get('r2', 'N/A')

            try:
                best_train_loss = f"{float(best_train_loss):.4f}" if best_train_loss != 'N/A' else 'N/A'
                best_valid_loss = f"{float(best_valid_loss):.4f}" if best_valid_loss != 'N/A' else 'N/A'
                rms_error_str = f"{float(rms_error_val):.2f}" if rms_error_val != 'N/A' else 'N/A'
                max_error_str = f"{float(max_error_val):.2f}" if max_error_val != 'N/A' else 'N/A'
                mape_str = f"{float(mape):.2f}" if mape != 'N/A' else 'N/A'
                r2_str = f"{float(r2):.4f}" if r2 != 'N/A' else 'N/A'
            except (ValueError, TypeError) as e:
                print(f"Error converting metrics to float: {e}")
                rms_error_str, max_error_str, mape_str, r2_str = 'N/A', 'N/A', 'N/A', 'N/A'

            row = QTreeWidgetItem([
                str(self.sl_no_counter), str(model_name), str(task_name), str(file_name),
                str(num_learnable_params), str(best_train_loss), str(best_valid_loss),
                str(completed_epochs), str(rms_error_str), str(max_error_str),
                str(mape_str), str(r2_str)
            ])
            self.sl_no_counter += 1

            button_widget = QWidget()
            button_layout = QHBoxLayout(button_widget)
            button_layout.setContentsMargins(4, 0, 4, 0)
            plot_button = QPushButton("Plot Result")
            plot_button.setStyleSheet("background-color: #800080; color: white; padding: 5px;")
            plot_path = predictions_file if predictions_file and os.path.exists(predictions_file) else None
            if plot_path:
                plot_button.clicked.connect(lambda _, p=plot_path, s=save_dir, tcn=target_column_name: 
                                         self.plot_model_result(p, s, tcn))
            else:
                plot_button.setDisabled(True)
                plot_button.setToolTip("Predictions file not found")
            button_layout.addWidget(plot_button)

            self.tree.addTopLevelItem(row)
            self.tree.setItemWidget(row, 12, button_widget)

            training_history_path = os.path.join(save_dir, f'training_history_{task_id}.png')
            if os.path.exists(training_history_path):
                self.show_training_history_plot(training_history_path, task_id)

    def plot_model_result(self, predictions_file, save_dir, target_column_name):
        try:
            if not os.path.exists(predictions_file):
                QMessageBox.critical(self, "Error", f"Predictions file not found: {predictions_file}")
                return

            df = pd.read_csv(predictions_file)
            
            true_col, pred_col, error_col, timestamp_col = None, None, None, None
            for col in df.columns:
                col_lower = col.lower()
                if 'true' in col_lower and target_column_name.lower() in col_lower: true_col = col
                elif 'predicted' in col_lower and target_column_name.lower() in col_lower: pred_col = col
                elif 'error' in col_lower: error_col = col
                elif 'timestamp' in col_lower or 'time' in col_lower: timestamp_col = col
            
            if not true_col or not pred_col:
                QMessageBox.critical(self, "Error", f"Required columns not found in predictions file.\nAvailable columns: {list(df.columns)}")
                return
                
            unit_display_long, error_unit = target_column_name, ""
            if "voltage" in target_column_name.lower():
                unit_display_long, error_unit = "Voltage (V)", "mV"
            elif "soc" in target_column_name.lower():
                unit_display_long, error_unit = "SOC (% SOC)", "% SOC"
            elif "temperature" in target_column_name.lower():
                unit_display_long, error_unit = "Temperature (°C)", "°C"
            
            errors_for_plot = df[error_col] if error_col else np.abs(df[true_col] - df[pred_col])
            if not error_col:
                if "voltage" in target_column_name.lower(): errors_for_plot *= 1000
                elif "soc" in target_column_name.lower() and np.max(np.abs(df[true_col])) <= 1.0: errors_for_plot *= 100

            rms_error = np.sqrt(np.mean(errors_for_plot**2))
            max_error = np.max(errors_for_plot)
            mean_error = np.mean(errors_for_plot)
            std_error = np.std(errors_for_plot)

            x_axis, x_label = (df.index, "Sample Index")
            if timestamp_col:
                try:
                    timestamps = pd.to_datetime(df[timestamp_col], errors='coerce')
                    if not timestamps.isna().all():
                        time_seconds = (timestamps - timestamps.iloc[0]).dt.total_seconds()
                        x_axis, x_label = time_seconds, "Time (seconds)"
                except: pass

            plot_dialog = QDialog(self)
            plot_dialog.setWindowTitle(f"Test Results: {os.path.basename(predictions_file)}")
            plot_dialog.setGeometry(100, 100, 1400, 1000)
            layout = QVBoxLayout(plot_dialog)
            
            fig = Figure(figsize=(14, 10), dpi=100)
            canvas = FigureCanvas(fig)
            from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
            toolbar = NavigationToolbar(canvas, plot_dialog)
            plt.style.use('seaborn-v0_8-darkgrid')
            
            ax1 = fig.add_subplot(3, 1, 1)
            ax1.plot(x_axis, df[true_col], label='True Values', color='#2E86AB', linewidth=2, alpha=0.8)
            ax1.plot(x_axis, df[pred_col], label='Predictions', color='#A23B72', linewidth=2, linestyle='--', alpha=0.8)
            ax1.set_title(f'Model Predictions vs. True Values\n{os.path.basename(predictions_file)}', fontsize=14, fontweight='bold', pad=20)
            ax1.set_ylabel(unit_display_long, fontsize=12)
            ax1.legend(fontsize=11, loc='upper right')
            ax1.grid(True, alpha=0.3)
            
            ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
            ax2.plot(x_axis, errors_for_plot, label=f'Absolute Error ({error_unit})', color='#F18F01', linewidth=1.5, alpha=0.7)
            ax2.set_title('Prediction Error Over Time', fontsize=12, fontweight='bold')
            ax2.set_xlabel(x_label, fontsize=12)
            ax2.set_ylabel(f'Error ({error_unit})', fontsize=12)
            ax2.legend(fontsize=11)
            ax2.grid(True, alpha=0.3)
            
            stats_text = f'RMS Error: {rms_error:.3f} {error_unit}\nMax Error: {max_error:.3f} {error_unit}\nMean Error: {mean_error:.3f} {error_unit}\nStd Error: {std_error:.3f} {error_unit}'
            ax2.text(0.98, 0.02, stats_text, transform=ax2.transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8, edgecolor='navy'))
            
            ax3 = fig.add_subplot(3, 1, 3)
            ax3.hist(errors_for_plot, bins=50, alpha=0.7, color='#F18F01', edgecolor='black', linewidth=0.5)
            ax3.axvline(x=mean_error, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.3f}')
            ax3.set_title('Error Distribution', fontsize=12, fontweight='bold')
            ax3.set_xlabel(f'Error ({error_unit})', fontsize=12)
            ax3.set_ylabel('Frequency', fontsize=12)
            ax3.legend(fontsize=10)
            ax3.grid(True, alpha=0.3)
            
            fig.tight_layout(pad=3.0)
            layout.addWidget(toolbar)
            layout.addWidget(canvas)
            
            button_layout = QHBoxLayout()
            save_button = QPushButton("Save Plot")
            save_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px; font-weight: bold;")
            save_button.clicked.connect(lambda: self.save_plot(fig, predictions_file, save_dir))
            button_layout.addWidget(save_button)
            close_button = QPushButton("Close")
            close_button.setStyleSheet("background-color: #f44336; color: white; padding: 8px; font-weight: bold;")
            close_button.clicked.connect(plot_dialog.close)
            button_layout.addWidget(close_button)
            layout.addLayout(button_layout)
            
            plot_dialog.exec_()

        except Exception as e:
            QMessageBox.critical(self, "Plotting Error", f"An error occurred while plotting: {e}")

    def save_plot(self, fig, test_file_path, save_dir):
        try:
            file_name = os.path.splitext(os.path.basename(test_file_path))[0]
            save_path = os.path.join(save_dir, f"{file_name}_test_plot.png")
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "Plot Saved", f"Plot saved to {save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save plot: {e}")

    def show_training_history_plot(self, plot_path, task_id):
        try:
            dialog = QDialog(self)
            dialog.setWindowTitle(f"Training History for Task {task_id}")
            layout = QVBoxLayout(dialog)
            pixmap = QPixmap(plot_path)
            label = QLabel()
            label.setPixmap(pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            layout.addWidget(label)
            dialog.exec_()
        except Exception as e:
            print(f"Error showing training history plot: {e}")

    def start_testing(self):
        self.start_time = time.time()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_elapsed_time)
        self.timer.start(1000)

        self.testing_thread = TestingThread(self.testing_manager, self.queue)
        self.testing_thread.result_signal.connect(self.add_result_row)
        self.testing_thread.testing_complete_signal.connect(self.all_tests_completed)
        self.testing_thread.update_status_signal.connect(self.update_status)
        self.testing_thread.start()
        QTimer.singleShot(100, self.process_queue)

    def update_elapsed_time(self):
        if self.timer_running:
            elapsed = int(time.time() - self.start_time)
            hours, rem = divmod(elapsed, 3600)
            minutes, seconds = divmod(rem, 60)
            self.time_label.setText(f"Testing Time: {hours:02d}h:{minutes:02d}m:{seconds:02d}s")

    def process_queue(self):
        try:
            while not self.queue.empty():
                result = self.queue.get_nowait()
                if 'task_completed' in result: self.add_result_row(result)
                elif 'all_tasks_completed' in result:
                    self.all_tests_completed()
                    return
                elif 'task_error' in result: self.update_status(f"Error: {result['task_error']}")
        except Empty: pass
        except Exception as e: self.update_status(f"Error processing queue: {e}")
        if self.timer_running: QTimer.singleShot(100, self.process_queue)

    def all_tests_completed(self):
        self.timer_running = False
        self.status_label.setText("All testing tasks completed. Exporting results to CSV...")
        self.progress.setValue(100)
        self.export_to_csv()
        self.cleanup_training_data()
        self.open_results_button.show()

    def cleanup_training_data(self):
        try:
            job_folder = self.job_manager.get_job_folder()
            if not job_folder or not os.path.exists(job_folder):
                self.logger.warning("Job folder not found. Skipping data cleanup.")
                return
            
            self.logger.info(f"Starting automatic data cleanup for job: {job_folder}")
            cleanup_success = self.data_cleanup_manager.save_file_references_and_cleanup(job_folder)
            
            if cleanup_success:
                self.logger.info("Automatic data cleanup completed successfully")
            else:
                self.logger.warning("Automatic data cleanup failed or was incomplete")
                
        except Exception as e:
            self.logger.error(f"Error during automatic data cleanup: {e}", exc_info=True)

    def export_to_csv(self):
        save_path = os.path.join(self.job_manager.get_job_folder(), "test_results_summary.csv")
        summary_data = self.testing_manager.get_results_summary()
        if not summary_data:
            self.logger.warning("No summary data to export.")
            self.status_label.setText("Export failed: No summary data to export.")
            return
        try:
            df = pd.DataFrame(summary_data)
            df.to_csv(save_path, index=False)
            self.status_label.setText(f"Test results exported to {os.path.basename(save_path)}")
            self.logger.info(f"Results exported to {save_path}")
        except Exception as e:
            self.logger.error(f"Could not export to CSV: {e}")
            QMessageBox.critical(self, "Export Failed", f"Could not export to CSV: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = VEstimTestingGUI(params={}, task_list=[], training_results={})
    gui.show()
    sys.exit(app.exec_())
