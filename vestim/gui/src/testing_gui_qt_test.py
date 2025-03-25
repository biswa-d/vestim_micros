# ---------------------------------------------------------------------------------
# Author: Biswanath Dehury
# Date: `{{date:2023-03-02}}`
# Version: 1.0.0
# Description: Description of the script
# Descrition: 
# This is the batchtesting without padding implementation for the unscaled data where the batch-size is used for testloader preparation but the model is tested
# one sequence at a time like a running window. The first part of the test file is padded with data to avoid the size mismatch and get the final prediction the same
# shape as the test file.

# Copyright (c) 2024 Biswanath Dehury, Dr. Phil Kollmeyer's Battery Lab at McMaster University
# ---------------------------------------------------------------------------------


from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QTreeWidget, QTreeWidgetItem, QProgressBar, QDialog, QMessageBox, QGridLayout
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QUrl
from PyQt5.QtGui import QFont, QDesktopServices
import os, sys, time
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from queue import Queue, Empty
import logging  
import matplotlib.pyplot as plt

# Import your services
from vestim.gateway.src.testing_manager_qt import VEstimTestingManager
from vestim.gateway.src.job_manager_qt import JobManager
from vestim.gateway.src.training_setup_manager_qt_test import VEstimTrainingSetupManager
from vestim.gateway.src.hyper_param_manager_qt_test import VEstimHyperParamManager

class TestingThread(QThread):
    # Define the signals at the class level
    update_status_signal = pyqtSignal(str)  # Signal to send status messages
    result_signal = pyqtSignal(dict)        # Signal to send test results
    testing_complete_signal = pyqtSignal()  # Signal to indicate all tasks are complete

    def __init__(self, testing_manager, queue):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.testing_manager = testing_manager
        self.queue = queue
        self.stop_flag = False

    def run(self):
        try:
            self.testing_manager.start_testing(self.queue)  # Pass the queue to the manager
            while not self.stop_flag:
                try:
                    result = self.queue.get(timeout=1)  # Non-blocking queue retrieval
                    if result:
                        # Check for the special completion message
                        if 'all_tasks_completed' in result:
                            self.testing_complete_signal.emit()  # Emit signal to inform the GUI
                            self.stop_flag = True  # Stop the thread loop
                        else:
                            self.result_signal.emit(result)
                except Empty:
                    continue  # Continue checking until the thread finishes testing
        except Exception as e:
            self.update_status_signal.emit(f"Error: {str(e)}")
        finally:
            print("Testing thread is stopping...")
            self.quit()  # Ensure the thread stops properly
            # self.wait()  # Make sure all operations are completed before exiting


class VEstimTestingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.job_manager = JobManager()
        self.testing_manager = VEstimTestingManager()
        self.hyper_param_manager = VEstimHyperParamManager()
        self.training_setup_manager = VEstimTrainingSetupManager()

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
        title_label = QLabel("Testing LSTM Models")
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
        self.tree.setColumnCount(9)
        self.tree.setHeaderLabels(["Sl.No", "Task ID", "Model", "File Name", "#W&Bs", "RMS Error (mV)", "Max Error (mV)", "MAPE (%)", "R²", "Plot"])

        # Set optimized column widths
        self.tree.setColumnWidth(0, 50)   # Sl.No column
        self.tree.setColumnWidth(1, 100)  # Task ID column
        self.tree.setColumnWidth(2, 230)  # Model name column (Wider)
        self.tree.setColumnWidth(3, 220)  # File name column (Wider)
        self.tree.setColumnWidth(4, 70)   # Number of learnable parameters
        self.tree.setColumnWidth(5, 120)   # RMS Error column
        self.tree.setColumnWidth(6, 120)   # Max Error column
        self.tree.setColumnWidth(7, 80)   # MAPE column
        self.tree.setColumnWidth(8, 80)   # R² column
        self.tree.setColumnWidth(9, 50)   # Plot button column (Narrow)

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
        print(f"Adding result row: {result}")
        self.logger.info(f"Adding result row: {result}")

        task_data = result.get('task_completed')

        if task_data:
            save_dir = task_data.get("saved_dir", "")  
            task_id = task_data.get("task_id", "N/A")
            model_name = task_data.get("model", "Unknown Model")
            file_name = task_data.get("file_name", "Unknown File")
            num_learnable_params = str(task_data.get("#params", "N/A"))

            # Extract metrics (all in mV)
            rms_error = f"{task_data.get('rms_error_mv', 0):.2f}"
            max_error = f"{task_data.get('max_error_mv', 0):.2f}"
            mape = f"{task_data.get('mape', 0):.2f}"
            r2 = f"{task_data.get('r2', 0):.4f}"

            # Add row data to QTreeWidget
            row = QTreeWidgetItem([
                str(self.sl_no_counter), 
                task_id, 
                model_name, 
                file_name, 
                num_learnable_params, 
                rms_error,
                max_error,
                mape,
                r2
            ])

            # Create "Plot" button
            plot_button = QPushButton("Plot Result")
            plot_button.setStyleSheet("background-color: #800080; color: white; padding: 5px;")
            # Use test_file path for plotting
            test_file = task_data.get('test_file', '')
            plot_button.clicked.connect(lambda _, path=test_file: self.plot_model_results(path))

            self.tree.addTopLevelItem(row)
            self.tree.setItemWidget(row, 9, plot_button)
            self.sl_no_counter += 1

    def plot_model_results(self, predictions_file):
        """Plot test results from a predictions CSV file."""
        try:
            print(f"Plotting results for test file: {predictions_file}")
            
            if not os.path.exists(predictions_file):
                QMessageBox.critical(self, "Error", f"Predictions file not found: {predictions_file}")
                return

            # Read and plot predictions
            df = pd.read_csv(predictions_file)
            if 'True Values (V)' in df.columns and 'Predictions (V)' in df.columns:
                self.plot_predictions(df['True Values (V)'], df['Predictions (V)'])
            else:
                QMessageBox.warning(self, "Warning", "Invalid predictions file format")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error plotting results: {str(e)}")

    def start_testing(self):
        print("Starting testing...")
        self.timer_running = True  # Reset the flag
        self.progress.setValue(0)  # Reset progress bar
        self.status_label.setText("Preparing test data...")
        self.start_time = time.time()
        self.progress.show()  # Ensure progress bar is visible

        self.testing_thread = TestingThread(self.testing_manager, self.queue)
        self.testing_thread.update_status_signal.connect(self.update_status)
        self.testing_thread.result_signal.connect(self.add_result_row)
        self.testing_thread.testing_complete_signal.connect(self.all_tests_completed)  # Connect to the completion signal
        self.testing_thread.start()

        # Start the timer for updating elapsed time
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_elapsed_time)  # Call the update method every second
        self.timer.start(1000)  # 1000 milliseconds = 1 second

        # Start processing the queue after the thread starts
        self.process_queue()
    
    def update_elapsed_time(self):
        """Update the elapsed time label."""
        if self.timer_running:
            elapsed_time = time.time() - self.start_time
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.time_label.setText(f"Testing Time: {int(hours):02}h:{int(minutes):02}m:{int(seconds):02}s")


    def process_queue(self):
        try:
            # Try to get a result from the queue
            result = self.queue.get_nowait()
            print(f"Got result from queue: {result}")
            self.add_result_row(result)  # Add the result to the GUI
            self.results_list.append(result)  # Track the completed results
        except Empty:
            # If the queue is empty, wait and try again
            QTimer.singleShot(100, self.process_queue)
            return  # Return early if there's nothing new to process
        # Process all the events in the Qt event loop (force repaint of the UI)
        QApplication.processEvents()
        
        # If new result is added, update the progress bar and status
        total_tasks = len(self.testing_manager.training_setup_manager.get_task_list())
        print(f"Total tasks: {total_tasks}")
        completed_tasks = len(self.results_list)
        print(f"Completed tasks: {completed_tasks}")
        
        if total_tasks == 0:  # Avoid division by zero
            self.update_status("No tasks to process.")
            return

        # Ensure progress is an integer between 0 and 100
        progress_value = int((completed_tasks / total_tasks) * 100)
        self.progress.setValue(progress_value)  # Update progress bar

        # Update the status with the number of completed tasks
        self.update_status(f"Completed {completed_tasks}/{total_tasks} tasks")

        # Check if all tasks are completed
        if completed_tasks >= total_tasks:
            # If all tasks are complete, stop processing the queue and update UI
            self.timer_running = False
            self.update_status("All tests completed!")
            self.progress.hide()  # Hide the progress bar when finished
            self.open_results_button.show()  # Show the results button
        else:
            # Continue checking the queue if tasks are not yet complete
            QTimer.singleShot(100, self.process_queue)

    
    def all_tests_completed(self):
        # Update the status label to indicate completion
        self.status_label.setText("All tests completed successfully.")
        
        self.progress.setValue(100)
        self.progress.hide()
        
        # Show the button to open the results folder
        self.open_results_button.show()
        
        # Stop the timer
        self.timer_running = False
        self.timer.stop()  # Stop the QTimer
        
        # Optionally log or print a message
        print("All tests completed successfully.")
        self.update_status("All tests completed successfully.")
        # Ensure the thread is properly cleaned up
        if self.testing_thread.isRunning():
            self.testing_thread.quit()
            self.testing_thread.wait()  # Wait for the thread to finish

    def open_job_folder(self):
        job_folder = self.job_manager.get_job_folder()
        if os.path.exists(job_folder):
            QDesktopServices.openUrl(QUrl.fromLocalFile(job_folder))
        else:
            QMessageBox.critical(self, "Error", f"Results folder not found: {job_folder}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = VEstimTestingGUI()
    gui.show()
    sys.exit(app.exec_())
