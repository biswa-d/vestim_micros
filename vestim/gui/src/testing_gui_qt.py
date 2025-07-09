# ---------------------------------------------------------------------------------
# Author: Biswanath Dehury
# Date: `{{date:2023-03-02}}`
# Version: 1.0.0
# Description: Description of the script
# Descrition: 
# This is the batchtesting without padding implementation for the unscaled data where the batch-size is used for testloader preparation but the model is tested
# one sequence at a time like a running window. The first part of the test file is padded with data to avoid the size mismatch and get the final prediction the same
# shape as the test file.
#
# Copyright (c) 2024 Biswanath Dehury, Dr. Phil Kollmeyer's Battery Lab at McMaster University
# ---------------------------------------------------------------------------------


from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, 
    QWidget, QTreeWidget, QTreeWidgetItem, QProgressBar, QDialog, QMessageBox, 
    QGridLayout
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
    def __init__(self, params, task_list, training_results=None):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.job_manager = JobManager()
        self.testing_manager = VEstimTestingManager(params=params, task_list=task_list, training_results=training_results)
        self.hyper_param_manager = VEstimHyperParamManager()
        self.training_setup_manager = VEstimTrainingSetupManager()
        self.data_cleanup_manager = DataCleanupManager()  # Add cleanup manager
        self.training_results = training_results if training_results is not None else {}

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

        self.queue = Queue()  # Queue to handle test results
        self.timer_running = True
        self.start_time = None
        self.testing_thread = None
        self.results_list = []  # List to store results
        self.hyper_params = {}  # Placeholder for hyperparameters
        self.sl_no_counter = 1  # Counter for sequential Sl.No


        self.initUI()
        self.start_testing()

    def initUI(self):
        self.setWindowTitle("VEstim Tool - Model Testing")
        self.setGeometry(100, 100, 900, 700)

        # Create a central widget and set the layout
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Title Label
        title_label = QLabel("Testing Models")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: green;")
        self.main_layout.addWidget(title_label)


        # Hyperparameters Display
        self.hyperparam_frame = QWidget()
        self.main_layout.addWidget(self.hyperparam_frame)
        self.hyper_params = self.hyper_param_manager.get_hyper_params()
        self.display_hyperparameters(self.hyper_params)
        print(f"Displayed hyperparameters: {self.hyper_params}")
        
        # Timer Label
        self.time_label = QLabel("Testing Time: 00h:00m:00s")
        # Set the font
        self.time_label.setFont(QFont("Helvetica", 10))  # Set the font family and size
        # Set the text color using CSS
        self.time_label.setStyleSheet("color: blue;")
        self.time_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.time_label)

        # Result Summary Label (above tree view)
        result_summary_label = QLabel("Testing Result Summary")
        result_summary_label.setAlignment(Qt.AlignCenter)
        result_summary_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.main_layout.addWidget(result_summary_label)

        # TreeWidget to display results
        self.tree = QTreeWidget()
        self.tree.setColumnCount(13)
        # Initial generic headers, will be updated by first result
        self.tree.setHeaderLabels(["Sl.No", "Task ID", "Model", "File Name", "#W&Bs", "Best Train Loss", "Best Valid Loss", "Epochs Trained", "Test RMSE", "Test MAXE", "MAPE (%)", "R²", "Plot"])

        # Set optimized column widths
        self.tree.setColumnWidth(0, 50)   # Sl.No column
        self.tree.setColumnWidth(1, 100)  # Task ID column
        self.tree.setColumnWidth(2, 200)  # Model name column (Wider)
        self.tree.setColumnWidth(3, 200)  # File name column (Wider)
        self.tree.setColumnWidth(4, 70)   # Number of learnable parameters
        self.tree.setColumnWidth(5, 100)  # Best Train Loss
        self.tree.setColumnWidth(6, 100)  # Best Valid Loss
        self.tree.setColumnWidth(7, 100)   # Epochs Trained
        self.tree.setColumnWidth(8, 100)   # Test RMSE
        self.tree.setColumnWidth(9, 100)   # Test MAXE
        self.tree.setColumnWidth(10, 70)   # MAPE column
        self.tree.setColumnWidth(11, 60)   # R² column
        self.tree.setColumnWidth(12, 100)   # Plot button column (Narrow)

        self.main_layout.addWidget(self.tree)

        # Status Label (below the tree view)
        self.status_label = QLabel("Preparing test data...")  # Initial status
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 12px; font-weight: bold; color: #004d99;")
        self.main_layout.addWidget(self.status_label)

        # Progress bar (below status label)
        self.progress = QProgressBar(self)
        self.progress.setMaximum(100)
        self.progress.setValue(0)
        self.main_layout.addWidget(self.progress)

        # Button to open results folder
        self.open_results_button = QPushButton("Open Job Folder", self)
        self.open_results_button.setStyleSheet("""
            background-color: #0b6337;  /* Matches the green color */
            font-weight: bold;
            padding: 10px 20px;  /* Adds padding inside the button */
            color: white;  /* Set the text color to white */
        """)
        self.open_results_button.setFixedHeight(40)  # Ensure consistent height
        self.open_results_button.setMinimumWidth(150)  # Set minimum width to ensure consistency
        self.open_results_button.setMaximumWidth(300)  # Set a reasonable maximum width
        self.open_results_button.clicked.connect(self.open_job_folder)
        # Center the button using a layout
        open_button_layout = QHBoxLayout()
        open_button_layout.addStretch(1)  # Add stretchable space before the button
        open_button_layout.addWidget(self.open_results_button, alignment=Qt.AlignCenter)
        open_button_layout.addStretch(1)  # Add stretchable space after the button

        # Add padding around the button by setting the margins
        open_button_layout.setContentsMargins(50, 20, 50, 20)  # Add margins (left, top, right, bottom)

        # Add the button layout to the main layout
        self.main_layout.addLayout(open_button_layout)

        # Initially hide the button
        self.open_results_button.hide()

    def open_job_folder(self):
        """Open the job folder in the file explorer."""
        job_folder = self.job_manager.get_job_folder()
        if job_folder and os.path.exists(job_folder):
            QDesktopServices.openUrl(QUrl.fromLocalFile(job_folder))
        else:
            QMessageBox.critical(self, "Error", f"Results folder not found: {job_folder}")

    def display_hyperparameters(self, params):
        print(f"Displaying hyperparameters: {params}")
        
        # Check if params is empty
        if not params:
            print("No hyperparameters to display.")
            return

        # Clear any existing widgets in the hyperparam_frame
        if self.hyperparam_frame.layout() is not None:
            while self.hyperparam_frame.layout().count():
                item = self.hyperparam_frame.layout().takeAt(0)
                widget = item.widget()
                if widget:
                    widget.setParent(None)  # Immediately remove widget from layout

        # Set the grid layout for hyperparam_frame if not already set
        grid_layout = QGridLayout()
        self.hyperparam_frame.setLayout(grid_layout)

        # Get the parameter items (mapping them to the correct labels)
        param_items = [(self.param_labels.get(param, param), value) for param, value in params.items()]

        # Split the parameters into five columns for better layout
        columns = [param_items[i::5] for i in range(5)]  # Split into 5 columns

        # Display each column with labels
        for col_num, column in enumerate(columns):
            for row, (param, value) in enumerate(column):
                value_str = str(value)

                # Truncate long comma-separated values for display
                if "," in value_str:
                    values = value_str.split(",")
                    display_value = f"{values[0]},{values[1]},..." if len(values) > 2 else value_str
                else:
                    display_value = value_str

                # Create parameter label and value label
                param_label = QLabel(f"{param}: ")
                param_label.setStyleSheet("font-size: 10pt; font-weight: bold;")
                value_label = QLabel(f"{display_value}")
                value_label.setStyleSheet("font-size: 10pt;")

                # Add labels to the grid layout
                grid_layout.addWidget(param_label, row, col_num * 2)
                grid_layout.addWidget(value_label, row, col_num * 2 + 1)

        # Force a layout update and repaint to ensure changes are visible
        self.hyperparam_frame.update()
        self.hyperparam_frame.repaint()


    def update_status(self, message):
        self.status_label.setText(message)

    def add_result_row(self, result):
        """Add each test result as a row in the QTreeWidget."""
        # print(f"Adding result row: {result}") # Original verbose print
        
        task_data_for_log = result.get('task_completed', {})
        log_summary = (
            f"Sl.No: {task_data_for_log.get('sl_no', 'N/A')}, "
            f"Model: {task_data_for_log.get('model', 'N/A')}, "
            f"File: {task_data_for_log.get('file_name', 'N/A')}"
        )
        self.logger.info(f"Adding result row: {log_summary}") # More concise log

        if 'task_error' in result:
            print(f"Error in task: {result['task_error']}")
            return

        task_data = result.get('task_completed')

        if task_data:
            save_dir = task_data.get("saved_dir", "")
            task_id = task_data.get("task_id", "N/A")
            model_name = task_data.get("model", "Unknown Model")
            file_name = task_data.get("file_name", "Unknown File")
            num_learnable_params = str(task_data.get("#params", "N/A"))
            best_train_loss = task_data.get("best_train_loss", "N/A")
            best_valid_loss = task_data.get("best_valid_loss", "N/A")
            completed_epochs = task_data.get("completed_epochs", "N/A")
            
            # Dynamically determine target column and units
            target_column_name = task_data.get("target_column", "")
            predictions_file = task_data.get("predictions_file", "")

            unit_suffix = ""
            unit_display = "" # For table headers
            if "voltage" in target_column_name.lower():
                unit_suffix = "_mv"
                unit_display = "(mV)"
            elif "soc" in target_column_name.lower():
                unit_suffix = "_percent"
                unit_display = "(% SOC)"  # Match training GUI format
            elif "temperature" in target_column_name.lower() or "temp" in target_column_name.lower():
                unit_suffix = "_degC"
                unit_display = "(Deg C)"  # Match training GUI format
            
            # Get unit display from task_data if available (for consistency)
            if 'unit_display' in task_data:
                unit_display = task_data['unit_display']
            
            # Update tree headers if this is the first result
            if self.sl_no_counter == 1:
                current_headers = [self.tree.headerItem().text(i) for i in range(self.tree.columnCount())]
                current_headers[8] = f"Test RMSE {unit_display}"
                current_headers[9] = f"Test MAXE {unit_display}"
                self.tree.setHeaderLabels(current_headers)

            # Extract metrics using dynamic keys
            rms_key = f'rms_error{unit_suffix}'
            mae_key = f'mae{unit_suffix}'
            max_error_key = f'max_abs_error{unit_suffix}'
            
            # Retrieve values with proper fallbacks
            rms_error_val = task_data.get(rms_key, 'N/A')
            max_error_val = task_data.get(max_error_key, task_data.get('max_error_mv', 'N/A'))
            mape = task_data.get('mape_percent', task_data.get('mape', 'N/A'))
            r2 = task_data.get('r2', 'N/A')

            # Safe conversion to float for formatting - ensures numpy types are properly handled
            try:
                if best_train_loss != 'N/A':
                    best_train_loss = f"{float(best_train_loss):.4f}"
                if best_valid_loss != 'N/A':
                    best_valid_loss = f"{float(best_valid_loss):.4f}"

                if rms_error_val != 'N/A':
                    rms_error_val = float(rms_error_val)
                    rms_error_str = f"{rms_error_val:.2f}"
                else:
                    rms_error_str = 'N/A'
                    
                if max_error_val != 'N/A':
                    max_error_val = float(max_error_val)
                    max_error_str = f"{max_error_val:.2f}"
                else:
                    max_error_str = 'N/A'
                    
                if mape != 'N/A':
                    mape = float(mape)
                    mape_str = f"{mape:.2f}"
                else:
                    mape_str = 'N/A'
                    
                if r2 != 'N/A':
                    r2 = float(r2)
                    r2_str = f"{r2:.4f}"
                else:
                    r2_str = 'N/A'
            except (ValueError, TypeError) as e:
                # Log the error and use safe defaults
                print(f"Error converting metrics to float: {e}")
                rms_error_str = str(rms_error_val) if rms_error_val is not None else 'N/A'
                max_error_str = str(max_error_val) if max_error_val is not None else 'N/A'
                mape_str = str(mape) if mape is not None else 'N/A'
                r2_str = str(r2) if r2 is not None else 'N/A'

            # Add row data to QTreeWidget - All values must be strings for QTreeWidgetItem
            row = QTreeWidgetItem([
                str(self.sl_no_counter),
                str(task_id),
                str(model_name),
                str(file_name),
                str(num_learnable_params),
                str(best_train_loss),
                str(best_valid_loss),
                str(completed_epochs),
                str(rms_error_str),   # Ensure string type
                str(max_error_str),   # Ensure string type
                str(mape_str),        # Ensure string type
                str(r2_str)           # Ensure string type
            ])
            self.sl_no_counter += 1

            # Create button layout widget
            button_widget = QWidget()
            button_layout = QHBoxLayout(button_widget)
            button_layout.setContentsMargins(4, 0, 4, 0)  # Reduce margins

            # Create "Plot Result" button
            plot_button = QPushButton("Plot Result")
            plot_button.setStyleSheet("background-color: #800080; color: white; padding: 5px;")
            # Use predictions_file path for plotting if available
            plot_path = predictions_file if predictions_file and os.path.exists(predictions_file) else None
            if plot_path:
                plot_button.clicked.connect(lambda _, p=plot_path, s=save_dir, tcn=target_column_name: 
                                         self.plot_model_result(p, s, tcn))
                button_layout.addWidget(plot_button)
            else:
                plot_button.setDisabled(True)
                plot_button.setToolTip("Predictions file not found")
                button_layout.addWidget(plot_button)

            # Add row to tree widget
            self.tree.addTopLevelItem(row)
            self.tree.setItemWidget(row, 12, button_widget)

            # Automatically show training history plot if it exists
            training_history_path = os.path.join(save_dir, f'training_history_{task_id}.png')
            if os.path.exists(training_history_path):
                self.show_training_history_plot(training_history_path, task_id)

    def plot_model_result(self, predictions_file, save_dir, target_column_name):
        """Plot test results for a specific model with dynamic units."""
        try:
            print(f"Plotting results from predictions file: {predictions_file} with target: {target_column_name}")
            if not os.path.exists(predictions_file):
                QMessageBox.critical(self, "Error", f"Predictions file not found: {predictions_file}")
                return

            df = pd.read_csv(predictions_file)
            
            # Determine column names based on target_column_name
            true_col = None
            pred_col = None
            diff_col = None
            
            # Look for columns containing 'True Values', 'Predictions', and 'Difference'
            for col in df.columns:
                if 'True Value' in col: # Changed from 'True Values' to 'True Value'
                    true_col = col
                elif 'Predictions' in col:
                    pred_col = col
                elif 'Error' in col: # Changed from 'Difference' to 'Error'
                    diff_col = col
            
            if not true_col or not pred_col:
                QMessageBox.critical(self, "Error", f"Required columns not found in predictions file.\nAvailable columns: {list(df.columns)}")
                return
                
            # Determine unit display based on target and columns
            unit_display_short = ""
            unit_display_long = target_column_name
            is_percentage_target = False # Flag for SOC, SOE, SOP

            if "voltage" in target_column_name.lower():
                unit_display_short = "V"
                unit_display_long = "Voltage (V)"
                error_unit = "mV"
            elif "soc" in target_column_name.lower():
                unit_display_short = "% SOC"
                unit_display_long = "SOC (% SOC)"
                error_unit = "% SOC"
                is_percentage_target = True
            elif "soe" in target_column_name.lower(): # New case for SOE
                unit_display_short = "% SOE"
                unit_display_long = "SOE (% SOE)"
                error_unit = "% SOE"
                is_percentage_target = True
            elif "sop" in target_column_name.lower(): # New case for SOP
                unit_display_short = "% SOP"
                unit_display_long = "SOP (% SOP)"
                error_unit = "% SOP"
                is_percentage_target = True
            elif "temperature" in target_column_name.lower() or "temp" in target_column_name.lower():
                unit_display_short = "Deg C"
                unit_display_long = "Temperature (Deg C)"
                error_unit = "Deg C"
            else:
                # Extract from column name if possible
                if "(" in true_col and ")" in true_col:
                    unit_match = true_col.split("(")[1].split(")")[0]
                    unit_display_short = unit_match
                    unit_display_long = f"{target_column_name} ({unit_match})"
                    error_unit = unit_match
                else:
                    unit_display_short = ""
                    unit_display_long = target_column_name
                    error_unit = ""
            
            # Calculate errors for plot text, applying scaling if necessary
            # errors_for_plot_text will be used for RMS and Max error display on the plot
            errors_for_plot_text = df[diff_col] if diff_col else df[true_col] - df[pred_col]
            if "voltage" in target_column_name.lower():
                errors_for_plot_text *= 1000 # Convert V to mV for plot text
            elif is_percentage_target and np.max(np.abs(df[true_col])) <= 1.0:
                errors_for_plot_text *= 100 # Convert 0-1 to % for plot text

            rms_error = np.sqrt(np.mean(errors_for_plot_text**2))
            max_error = np.max(np.abs(errors_for_plot_text))

            # Create a new dialog for the plot
            plot_dialog = QDialog(self)
            plot_dialog.setWindowTitle(f"Test Result: {os.path.basename(predictions_file)}")
            plot_dialog.setGeometry(150, 150, 1000, 800)
            
            layout = QVBoxLayout()
            
            # Matplotlib Figure
            fig = Figure(figsize=(10, 8))
            canvas = FigureCanvas(fig)
            
            # Main plot (True vs. Predicted)
            ax1 = fig.add_subplot(2, 1, 1)
            ax1.plot(df.index, df[true_col], label='True Values', color='blue')
            ax1.plot(df.index, df[pred_col], label='Predictions', color='red', linestyle='--')
            ax1.set_title(f'Model Predictions vs. True Values for {os.path.basename(predictions_file)}')
            ax1.set_ylabel(unit_display_long)
            ax1.legend()
            ax1.grid(True)
            
            # Error plot
            ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
            ax2.plot(df.index, errors_for_plot_text, label=f'Error ({error_unit})', color='green')
            ax2.set_title('Prediction Error')
            ax2.set_xlabel('Time Steps')
            ax2.set_ylabel(f'Error ({error_unit})')
            ax2.legend()
            ax2.grid(True)
            
            # Add RMS and Max Error text to the error plot
            ax2.text(0.05, 0.95, f'RMS Error: {rms_error:.2f} {error_unit}\nMax Error: {max_error:.2f} {error_unit}',
                     transform=ax2.transAxes, fontsize=10,
                     verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
            
            fig.tight_layout()
            
            layout.addWidget(canvas)
            
            # Save button
            save_button = QPushButton("Save Plot")
            save_button.clicked.connect(lambda: self.save_plot(fig, predictions_file, save_dir))
            layout.addWidget(save_button)
            
            plot_dialog.setLayout(layout)
            plot_dialog.exec_()

        except Exception as e:
            QMessageBox.critical(self, "Plotting Error", f"An error occurred while plotting: {e}")

    def save_plot(self, fig, test_file_path, save_dir):
        """Save the current plot as a PNG image."""
        try:
            file_name = os.path.splitext(os.path.basename(test_file_path))[0]
            save_path = os.path.join(save_dir, f"{file_name}_test_plot.png")
            fig.savefig(save_path)
            QMessageBox.information(self, "Plot Saved", f"Plot saved to {save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save plot: {e}")

    def show_training_history_plot(self, plot_path, task_id):
        """Display the training history plot in a new window."""            
        try:
            dialog = QDialog(self)
            dialog.setWindowTitle(f"Training History for Task {task_id}")
            
            layout = QVBoxLayout()
            
            pixmap = QPixmap(plot_path)
            label = QLabel()
            label.setPixmap(pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
            layout.addWidget(label)
            dialog.setLayout(layout)
            
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

        # Start processing the queue for results
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
                if 'task_completed' in result:
                    self.add_result_row(result)
                elif 'all_tasks_completed' in result:
                    self.all_tests_completed()
                    return  # Stop processing after completion signal
                elif 'task_error' in result:
                    self.update_status(f"Error: {result['task_error']}")
        except Empty:
            pass  # Queue is empty, do nothing
        except Exception as e:
            self.update_status(f"Error processing queue: {e}")

        if self.timer_running:
            QTimer.singleShot(100, self.process_queue)

    def all_tests_completed(self):
        self.timer_running = False
        self.status_label.setText("All testing tasks completed. Exporting results to CSV...")
        self.progress.setValue(100)
        self.export_to_csv()
        self.cleanup_training_data()  # Call cleanup after all tests are done
        self.open_results_button.show()


    def cleanup_training_data(self):
        """
        Clean up training data folders and save file references for traceability.
        This saves significant storage space while maintaining full traceability.
        """
        try:
            job_folder = self.job_manager.get_job_folder()
            if not job_folder or not os.path.exists(job_folder):
                self.logger.warning("Job folder not found. Skipping data cleanup.")
                return
            
            self.logger.info(f"Starting automatic data cleanup for job: {job_folder}")
            
            # Check if data folders exist before cleanup
            data_folders_exist = any(
                os.path.exists(os.path.join(job_folder, folder))
                for folder in ['train_data', 'val_data', 'test_data']
            )
            
            if not data_folders_exist:
                self.logger.info("No data folders found. Cleanup may have already been performed.")
                print("Data folders not found. Cleanup may have already been performed.")
                return
            
            # Perform cleanup with better error handling
            cleanup_success = self.data_cleanup_manager.save_file_references_and_cleanup(job_folder)
            
            if cleanup_success:
                self.logger.info("Automatic data cleanup completed successfully")
                print("✓ Data cleanup completed successfully!")
                print("  • Training data folders removed to save space")
                print("  • File references saved for traceability")
                print("  • Original data files preserved in their source locations")
            else:
                self.logger.warning("Automatic data cleanup failed or was incomplete")
                print("⚠ Data cleanup encountered some issues but file references were saved.")
                print("  • Check logs for details on any folders that couldn't be removed")
                print("  • Most storage space should still be freed")
                
        except Exception as e:
            self.logger.error(f"Error during automatic data cleanup: {e}", exc_info=True)
            print(f"⚠ Warning: Automatic data cleanup encountered an error: {e}")
            print("  • File references should still be saved")
            print("  • You may need to manually delete training data folders if needed")

    def export_to_csv(self):
        """Export the contents of the QTreeWidget to a CSV file."""
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
    # Example usage:
    # params = {'some_param': 'value'}
    # task_list = [{'task_id': '1', ...}]
    # training_results = {'1': {'best_validation_loss': 0.1, 'final_train_loss_denorm': 0.2}}
    # gui = VEstimTestingGUI(params=params, task_list=task_list, training_results=training_results)
    gui = VEstimTestingGUI(params={}, task_list=[], training_results={})
    gui.show()
    sys.exit(app.exec_())
