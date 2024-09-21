from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QTreeWidget, QTreeWidgetItem, QProgressBar, QDialog, QMessageBox, QGridLayout
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QUrl
from PyQt5.QtGui import QFont, QDesktopServices
import os, sys, time
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from queue import Queue, Empty

# Import your services
from vestim.gateway.src.testing_manager_qt_test import VEstimTestingManager
from vestim.gateway.src.job_manager_qt import JobManager
from vestim.gateway.src.training_setup_manager_qt_test import VEstimTrainingSetupManager
from vestim.gateway.src.hyper_param_manager_qt import VEstimHyperParamManager

class TestingThread(QThread):
    # Define the signals at the class level
    update_status_signal = pyqtSignal(str)  # Signal to send status messages
    result_signal = pyqtSignal(dict)        # Signal to send test results
    testing_complete_signal = pyqtSignal()  # Signal to indicate all tasks are complete

    def __init__(self, testing_manager, queue):
        super().__init__()
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
        self.open_results_button = QPushButton("Open Results Folder", self)
        self.open_results_button.setStyleSheet("""
            background-color: #0b6337;  /* Matches the green color */
            font-weight: bold; 
            padding: 10px 20px;  /* Adds padding inside the button */
            color: white;  /* Set the text color to white */
        """)
        self.open_results_button.setFixedHeight(40)  # Ensure consistent height
        self.open_results_button.setMinimumWidth(150)  # Set minimum width to ensure consistency
        self.open_results_button.setMaximumWidth(300)  # Set a reasonable maximum width
        self.open_results_button.clicked.connect(self.open_results_folder)

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
        # Add each result as a row in the QTreeWidget
        print(f"Adding result row: {result}")
        task_data = result.get('task_completed')
        if task_data:
            model_name = task_data.get("model")
            rms_error = f"{task_data.get('rms_error_mv', 0):.4f}"
            mae = f"{task_data.get('mae_mv', 0):.4f}"
            mape = f"{task_data.get('mape', 0):.4f}"
            r2 = f"{task_data.get('r2', 0):.4f}"

            # Manually increment Sl.No counter
            sl_no = self.sl_no_counter
            self.sl_no_counter += 1

            # Add row data to QTreeWidget
            row = QTreeWidgetItem([str(sl_no), model_name, rms_error, mae, mape, r2])

            # Set the column widths (adjust these numbers as needed)
            self.tree.setColumnWidth(0, 50)
            self.tree.setColumnWidth(1, 310)  # Set wider width for model name
            self.tree.setColumnWidth(6, 50)  # Set smaller width for the plot button

            # Create the "Plot" button with some styling
            plot_button = QPushButton("Plot Result")
            plot_button.setStyleSheet("background-color: #800080; color: white; padding: 5px;")  # Purple background
            plot_button.clicked.connect(lambda _, name=model_name: self.plot_model_results(name))  # Pass model_name to plot
            self.tree.addTopLevelItem(row)

            # Set widget for the "Plot" column
            self.tree.setItemWidget(row, 6, plot_button)


    def plot_model_results(self, model_name):
        """
        Plot the test results for a specific model by reading from the saved CSV file.
        """
        try:
            # Create a new dialog window for plotting
            plot_window = QDialog(self)
            plot_window.setWindowTitle(f"Testing Results for Model: {model_name}")
            plot_window.setGeometry(200, 100, 700, 500)

            # Locate the CSV file for this model
            save_dir = self.job_manager.get_test_results_folder()  # Assuming this method returns the correct save directory
            result_file = os.path.join(save_dir, model_name, f"{model_name}_test_results.csv")

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

            # Plot the true values and predictions
            ax.plot(true_values, label='True Values (V)', color='blue', marker='o', markersize=3, linestyle='-', linewidth=0.8)
            ax.plot(predictions, label='Predictions (V)', color='green', marker='x', markersize=3, linestyle='--', linewidth=0.8)

            # Customize the labels, title, and legend
            ax.set_xlabel('Index', fontsize=12)
            ax.set_ylabel('Voltage (V)', fontsize=12)
            ax.set_title(f"Testing Results for Model: {model_name}", fontsize=14, fontweight='bold')
            ax.legend(loc='upper right', fontsize=10)

            # Fine-tune grid and ticks for better readability
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.tick_params(axis='both', which='major', labelsize=10)

            # Embed the plot in the dialog window using FigureCanvas
            canvas = FigureCanvas(fig)
            layout = QVBoxLayout()
            layout.addWidget(canvas)
            plot_window.setLayout(layout)

            # Add "Save Plot" button
            save_button = QPushButton("Save Plot")
            save_button.clicked.connect(lambda: self.save_plot(fig, model_name, save_dir))
            layout.addWidget(save_button)

            # Show the plot window
            plot_window.exec_()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while plotting results for model: {model_name}\n{str(e)}")


    def save_plot(self, fig, model_name, save_dir):
        """
        Save the current plot as a PNG image.
        
        :param fig: The figure object of the plot.
        :param model_name: The name of the model being plotted.
        :param save_dir: The directory where the plot should be saved.
        """
        # Create the file path for the saved image
        plot_file = os.path.join(save_dir, model_name, f"{model_name}_test_results_plot.png")

        # Save the figure as a PNG image
        fig.savefig(plot_file, format='png')
        QMessageBox.information(self, "Saved", f"Plot saved as: {plot_file}")
        print(f"Plot saved as: {plot_file}")

    def start_testing(self):
        print("Starting testing...")
        self.timer_running = True  # Reset the flag
        self.progress.setValue(0)  # Reset progress bar
        self.start_time = time.time()
        self.update_elapsed_time()  # Start updating the timer
        self.status_label.setText("Preparing test data...")
        self.progress.show()  # Ensure progress bar is visible

        self.testing_thread = TestingThread(self.testing_manager, self.queue)
        # self.update_elapsed_time()  # Start the timer when testing starts
        self.testing_thread.update_status_signal.connect(self.update_status)
        self.testing_thread.result_signal.connect(self.add_result_row)
        self.testing_thread.testing_complete_signal.connect(self.all_tests_completed)  # Connect to the completion signal
        self.testing_thread.start()

        # Start processing the queue after the thread starts
        self.process_queue()

    def process_queue(self):
        try:
            # Try to get a result from the queue
            result = self.queue.get_nowait()
            print(f"Got result from queue: {result}")
            self.add_result_row(result)  # Add the result to the GUI
            self.results_list.append(result)  # Track the completed results

            # If new result is added, update the progress bar and status immediately
            total_tasks = len(self.testing_manager.training_setup_manager.get_task_list())
            completed_tasks = len(self.results_list)

            # Avoid division by zero
            if total_tasks > 0:
                progress_value = int((completed_tasks / total_tasks) * 100)
                self.progress.setValue(progress_value)  # Update progress bar
                self.update_status(f"Completed {completed_tasks}/{total_tasks} tasks")

            # Check if all tasks are completed
            if completed_tasks >= total_tasks:
                self.timer_running = False
                self.update_status("All tests completed!")
                self.progress.hide()  # Hide the progress bar when finished
                self.open_results_button.show()  # Show the results button

        except Empty:
            # If the queue is empty, wait and try again
            QTimer.singleShot(100, self.process_queue)
            return  # Return early if there's nothing new to process

        # Process all the events in the Qt event loop (force repaint of the UI)
        QApplication.processEvents()

        # Continue checking the queue if tasks are not yet complete
        if self.timer_running:
            QTimer.singleShot(100, self.process_queue)

    # Working code being modified for progress bar and timer update, to be reverted back/deleted depending on the result from the modified code above
    # def process_queue(self):
    #     try:
    #         # Try to get a result from the queue
    #         result = self.queue.get_nowait()
    #         print(f"Got result from queue: {result}")
    #         self.add_result_row(result)  # Add the result to the GUI
    #         self.results_list.append(result)  # Track the completed results
    #     except Empty:
    #         # If the queue is empty, wait and try again
    #         # QTimer.singleShot(100, self.process_queue) # thi is being taken outside the block for testing
    #         return  # Return early if there's nothing new to process
    #     # Process all the events in the Qt event loop (force repaint of the UI)
    #     QTimer.singleShot(100, self.process_queue)
    #     QApplication.processEvents()
        
    #     # If new result is added, update the progress bar and status
    #     total_tasks = len(self.testing_manager.training_setup_manager.get_task_list())
    #     print(f"Total tasks: {total_tasks}")
    #     completed_tasks = len(self.results_list)
    #     print(f"Completed tasks: {completed_tasks}")
        
    #     if total_tasks == 0:  # Avoid division by zero
    #         self.update_status("No tasks to process.")
    #         return

    #     # Ensure progress is an integer between 0 and 100
    #     progress_value = int((completed_tasks / total_tasks) * 100)
    #     self.progress.setValue(progress_value)  # Update progress bar

    #     # Update the status with the number of completed tasks
    #     self.update_status(f"Completed {completed_tasks}/{total_tasks} tasks")

    #     # Check if all tasks are completed
    #     if completed_tasks >= total_tasks:
    #         # If all tasks are complete, stop processing the queue and update UI
    #         self.timer_running = False
    #         self.update_status("All tests completed!")
    #         self.progress.hide()  # Hide the progress bar when finished
    #         self.open_results_button.show()  # Show the results button
    #     else:
    #         # Continue checking the queue if tasks are not yet complete
    #         QTimer.singleShot(100, self.process_queue)

    def update_elapsed_time(self):
        if self.timer_running:
            elapsed_time = time.time() - self.start_time
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.time_label.setText(f"Testing Time: {int(hours):02}h:{int(minutes):02}m:{int(seconds):02}s")
            
            # Call the method again after 1 second
            QTimer.singleShot(1000, self.update_elapsed_time)

    def all_tests_completed(self):
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
        # Ensure the thread is properly cleaned up
        if self.testing_thread.isRunning():
            self.testing_thread.quit()
            self.testing_thread.wait()  # Wait for the thread to finish

    def open_results_folder(self):
        results_folder = self.job_manager.get_test_results_folder()
        if os.path.exists(results_folder):
            QDesktopServices.openUrl(QUrl.fromLocalFile(results_folder))
        else:
            QMessageBox.critical(self, "Error", f"Results folder not found: {results_folder}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = VEstimTestingGUI()
    gui.show()
    sys.exit(app.exec_())
