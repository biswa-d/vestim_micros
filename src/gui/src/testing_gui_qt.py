from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QWidget, QTreeWidget, QTreeWidgetItem, QProgressBar, QDialog, QFileDialog, QMessageBox, QGridLayout
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont
import os, sys, time
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from queue import Queue, Empty

# Import your services
from src.gateway.src.testing_manager_qt import VEstimTestingManager
from src.gateway.src.job_manager import JobManager
from src.gateway.src.training_setup_manager_qt import VEstimTrainingSetupManager
from src.gateway.src.hyper_param_manager_test import VEstimHyperParamManager

class TestingThread(QThread):
    # Define the signals at the class level
    update_status_signal = pyqtSignal(str)  # Signal to send status messages
    result_signal = pyqtSignal(dict)        # Signal to send test results

    def __init__(self, testing_manager, queue):
        super().__init__()
        self.testing_manager = testing_manager
        self.queue = queue

    def run(self):
        try:
            self.testing_manager.start_testing()  # You don't need to pass `queue` or `emit` functions
            while True:
                try:
                    result = self.queue.get(timeout=1)  # Non-blocking queue retrieval
                    if result:
                        self.result_signal.emit(result)
                except Empty:
                    continue  # Continue checking until the thread finishes testing
        except Exception as e:
            self.update_status_signal.emit(f"Error: {str(e)}")
        finally:
            self.quit()  # Ensure the thread stops properly


class VEstimTestingGUI(QMainWindow):
    def __init__(self):
        super().__init__()

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
        self.tree.setColumnCount(7)
        self.tree.setHeaderLabels(["Sl.No", "Model", "RMS Error (mV)", "MAE (mV)", "MAPE (%)", "R²", "Plot"])
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
        self.open_results_button = QPushButton("Open Results Folder")
        self.open_results_button.clicked.connect(self.open_results_folder)
        self.main_layout.addWidget(self.open_results_button)
        self.open_results_button.hide()  # Hide the button by default

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
        # Add each result as a row in the QTreeWidget
        print(f"Adding result row: {result}")
        task_data = result.get('task_completed')
        if task_data:
            sl_no = task_data.get("sl_no")
            model_name = task_data.get("model")
            rms_error = f"{task_data.get('rms_error_mv', 0):.4f}"
            mae = f"{task_data.get('mae_mv', 0):.4f}"
            mape = f"{task_data.get('mape', 0):.4f}"
            r2 = f"{task_data.get('r2', 0):.4f}"

            # Add row data to QTreeWidget
            row = QTreeWidgetItem([str(sl_no), model_name, rms_error, mae, mape, r2])

            # Create the "Plot" button as the last column for this row
            plot_button = QPushButton("Plot")
            plot_button.clicked.connect(lambda: self.plot_model_results(model_name))
            self.tree.addTopLevelItem(row)

            # Set widget for the "Plot" column
            self.tree.setItemWidget(row, 6, plot_button)

    def plot_model_results(self, model_name):
        """
        Plot the test results for a specific model by reading from the saved CSV file.
        """
        try:
            plot_window = QDialog(self)
            plot_window.setWindowTitle(f"Testing Results for Model: {model_name}")
            plot_window.setGeometry(200, 100, 700, 500)

            # Locate the CSV file for this model
            result_file = f"{model_name}_test_results.csv"  # Use actual file path logic

            if not os.path.exists(result_file):
                QMessageBox.critical(self, "Error", f"Test results file not found for model: {model_name}")
                return

            # Load the data from the CSV file
            df = pd.read_csv(result_file)
            true_values = df['True Values (V)']
            predictions = df['Predictions (V)']

            # Create a matplotlib figure for plotting
            fig = Figure(figsize=(7, 4), dpi=100)
            ax = fig.add_subplot(111)
            ax.plot(true_values, label='True Values (V)', color='blue')
            ax.plot(predictions, label='Predictions (V)', color='green')

            ax.set_xlabel('Index')
            ax.set_ylabel('Voltage (V)')
            ax.set_title(f"Testing Results for Model: {model_name}")
            ax.legend()

            # Embed the plot in the dialog window using FigureCanvas
            canvas = FigureCanvas(fig)
            layout = QVBoxLayout()
            layout.addWidget(canvas)
            plot_window.setLayout(layout)

            # Show the plot window
            plot_window.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while plotting results for model: {model_name}\n{str(e)}")


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
        self.testing_thread.finished.connect(self.all_tests_completed)  # Connect to the newly added method
        self.testing_thread.start()

        # Start processing the queue after the thread starts
        self.process_queue()


    def process_queue(self):
        try:
            result = self.queue.get_nowait()
            print(f"Got result from queue: {result}")
            self.add_result_row(result)
            self.results_list.append(result)
        except Empty:
            QTimer.singleShot(1000, self.process_queue)

        total_tasks = len(self.testing_manager.training_setup_manager.get_task_list())
        completed_tasks = len(self.results_list)
        # Ensure the value passed to setValue is an integer
        progress_value = int((completed_tasks / total_tasks) * 100)
        self.progress.setValue(progress_value)
        # self.progress.setValue((completed_tasks / total_tasks) * 100)

        self.update_status(f"Completed {completed_tasks}/{total_tasks} tasks")

        if completed_tasks >= total_tasks:
            self.timer_running = False
            self.update_status("All tests completed!")
            self.progress.hide()
            self.open_results_button.show()
    
    def all_tests_completed(self):
        self.testing_thread.quit()  # Stop the thread
        self.testing_thread.wait()  # Ensure it's cleaned up
        # Update the status label to indicate completion
        self.status_label.setText("All tests completed successfully.")
        
        self.progress.setValue(100)
        self.progress.hide()
        
        # Show the button to open the results folder
        self.open_results_button.show()
        
        # Stop the timer
        self.timer_running = False
        
        # Optionally log or print a message
        print("All tests completed successfully.")
        self.update_status("All tests completed successfully.")


    def open_results_folder(self):
        results_folder = self.job_manager.get_test_results_folder()
        if os.path.exists(results_folder):
            QFileDialog.getExistingDirectory(self, "Open Results Folder", results_folder)
        else:
            QMessageBox.critical(self, "Error", f"Results folder not found: {results_folder}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = VEstimTestingGUI()
    gui.show()
    sys.exit(app.exec_())
