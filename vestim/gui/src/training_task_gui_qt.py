from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QHBoxLayout, QVBoxLayout, QPushButton, QWidget, QFrame, QTextEdit, QGridLayout
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
import torch
import json, time
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from queue import Queue, Empty
from threading import Thread
import logging, wandb

# Import local services
from vestim.services.model_training.src.training_task_service_test import TrainingTaskService
from vestim.gateway.src.training_task_manager_qt_test import TrainingTaskManager
from vestim.gateway.src.training_setup_manager_qt_test import VEstimTrainingSetupManager
from vestim.gateway.src.job_manager_qt import JobManager
from vestim.gui.src.testing_gui_qt_test import VEstimTestingGUI

class TrainingThread(QThread):
    # Custom signals to emit data back to the main GUI
    update_epoch_signal = pyqtSignal(dict)  # Signal to send progress data (e.g., after each epoch)
    task_completed_signal = pyqtSignal()  # Signal when the task is completed
    task_error_signal = pyqtSignal(str)  # Signal for any error during the task

    def __init__(self, task, training_task_manager):
        super().__init__()
        self.task = task
        self.training_task_manager = training_task_manager

    def run(self):
        try:
            # Process the task in the background
            self.training_task_manager.process_task(self.task, self.update_epoch_signal)
            self.task_completed_signal.emit()  # Emit signal when the task is completed
        except Exception as e:
            self.task_error_signal.emit(str(e))  # Emit error message


class VEstimTrainingTaskGUI(QMainWindow):
    def __init__(self, task_list, params):
        super().__init__()
        
        #Logger setup
        self.logger = logging.getLogger(__name__)
        # Initialize WandB flag
        self.use_wandb = False  # Set to False if WandB should not be used
        self.wandb_enabled = False
        if self.use_wandb:
            try:
                import wandb
                wandb.init(project="VEstim", config={"task_name": "LSTM Model Training"})
                self.wandb_enabled = True
            except Exception as e:
                self.wandb_enabled = False
                self.logger.error(f"Failed to initialize WandB in GUI: {e}")
        

        self.training_task_manager = TrainingTaskManager()
        self.training_setup_manager = VEstimTrainingSetupManager()
        self.training_service = TrainingTaskService()
        self.job_manager = JobManager()

        self.task_list = task_list
        self.params = params

        # Initialize variables
        self.train_loss_values = []
        self.valid_loss_values = []
        self.valid_x_values = []
        self.start_time = None
        self.queue = Queue()
        self.timer_running = True
        self.training_process_stopped = False
        self.task_completed_flag = False
        self.current_task_index = 0

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
            "REPETITIONS": "Repetitions",
            "NUM_LEARNABLE_PARAMS": "Number of Learnable Parameters",

        }

        self.initUI()
        self.build_gui(self.task_list[self.current_task_index])
        self.start_task_processing()

    def initUI(self):
        self.setWindowTitle(f"VEstim - Training Task {self.current_task_index + 1}")
        self.setGeometry(100, 100, 900, 600)

    def build_gui(self, task):
        # Create a main widget to set as central widget in QMainWindow
        container = QWidget()
        self.setCentralWidget(container)

        # Create a main layout
        self.main_layout = QVBoxLayout()

        # Title Label
        title_label = QLabel("Training LSTM Model with Hyperparameters")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 16pt; font-weight: bold;")  # Increase font size and make it bold
        self.main_layout.addWidget(title_label)

        # Display hyperparameters
        # Initialize the hyperparameter frame
        self.hyperparam_frame = QFrame(self)
        self.hyperparam_frame.setLayout(QVBoxLayout())  # Set a default layout for the frame
        self.main_layout.addWidget(self.hyperparam_frame)
        self.display_hyperparameters(task['hyperparams'])

        # Status Label
        self.status_label = QLabel("Starting training...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.status_label)

        # Time Frame, Plot Setup, and Log Window
        self.setup_time_and_plot(task)
        self.setup_log_window(task)

        # Stop button (centered and styled)
        self.stop_button = QPushButton("Stop Training")
        self.stop_button.setStyleSheet("background-color: red; color: white; font-size: 12pt; font-weight: bold;")
        self.stop_button.setFixedWidth(150)  # Set a fixed width for the button
        self.stop_button.clicked.connect(self.stop_training)

        # Layout for stop button
        stop_button_layout = QHBoxLayout()
        stop_button_layout.addStretch(1)  # Push the button to the center
        stop_button_layout.addWidget(self.stop_button)
        stop_button_layout.addStretch(1)  # Push the button to the center
        self.main_layout.addLayout(stop_button_layout)  # Add this layout to the main layout

        # Initialize the Proceed to Testing button
        self.proceed_button = QPushButton("Proceed to Testing")
        self.proceed_button.setStyleSheet("""
            background-color: #0b6337; 
            color: white; 
            font-size: 12pt; 
            font-weight: bold; 
            padding: 10px 20px;
        """)
        self.proceed_button.setVisible(False)  # Initially hidden
        self.proceed_button.clicked.connect(self.transition_to_testing_gui)

        # Layout for proceed button
        proceed_button_layout = QHBoxLayout()
        proceed_button_layout.addStretch(1)
        proceed_button_layout.addWidget(self.proceed_button, alignment=Qt.AlignCenter)
        proceed_button_layout.addStretch(1)
        self.main_layout.addLayout(proceed_button_layout)  # Add this layout to the main layout


        # Progress Label (initially hidden)
        self.progress_label = QLabel("Processing...")
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.hide()
        self.main_layout.addWidget(self.progress_label)

        # Attach the layout to the central widget
        container.setLayout(self.main_layout)

    def display_hyperparameters(self, params):
        # Clear previous widgets in the hyperparam_frame layout
        layout = self.hyperparam_frame.layout()
        
        # Remove old widgets from the layout
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        # Create a new grid layout for hyperparameters
        hyperparam_layout = QGridLayout()

        # Get the parameter items (mapping them to the correct labels if necessary)
        param_items = [(self.param_labels.get(param, param), value) for param, value in params.items()]

        # Split the parameters into five columns
        columns = [param_items[i::5] for i in range(5)]  # Split into five columns

        # Display each column with labels
        for col_num, column in enumerate(columns):
            for row_num, (param, value) in enumerate(column):
                value_str = str(value)
                if "," in value_str:
                    values = value_str.split(",")
                    if len(values) > 2:
                        display_value = f"{values[0]},{values[1]},..."
                    else:
                        display_value = value_str
                else:
                    display_value = value_str

                # Create parameter label
                param_label = QLabel(f"{param}:")
                value_label = QLabel(f"{display_value}")
                param_label.setStyleSheet("font-size: 10pt;")
                value_label.setStyleSheet("font-size: 10pt; font-weight: bold;")

                # Add labels to the grid layout
                hyperparam_layout.addWidget(param_label, row_num, col_num * 2)
                hyperparam_layout.addWidget(value_label, row_num, col_num * 2 + 1)

        # Now add the grid layout to the existing layout (hyperparam_frame's layout)
        layout.addLayout(hyperparam_layout)


    def setup_time_and_plot(self, task):
        # Debugging statement to check the structure of the task
        print(f"Current task: {task}")
        print(f"Hyperparameters in the task: {task['hyperparams']}")

        # Time Layout
        time_layout = QHBoxLayout()

        # Time tracking label (move it just below the title)
        time_layout = QHBoxLayout()
        self.static_text_label = QLabel("Time Since Setup Started:")
        self.static_text_label.setStyleSheet("color: blue; font-size: 10pt;")
        self.time_value_label = QLabel("00h:00m:00s")
        self.time_value_label.setStyleSheet("color: purple; font-size: 11pt; font-weight: bold;")
        # Align both the label and the value in the same row, close to each other
        time_layout.addStretch(1)  # Adds space to push both labels to the center
        time_layout.addWidget(self.static_text_label)
        time_layout.addWidget(self.time_value_label)
        time_layout.addStretch(1)  # Adds space after the labels to keep them centered
        # Add the time layout to the main layout
        self.main_layout.addLayout(time_layout)

        # Add time layout to the main layout
        self.main_layout.addLayout(time_layout)

        # Plot Setup
        max_epochs = int(task['hyperparams']['MAX_EPOCHS'])
        valid_frequency = int(task['hyperparams']['ValidFrequency'])

        # Matplotlib figure setup
        fig = Figure(figsize=(6, 2.5), dpi=100)
        self.ax = fig.add_subplot(111)
        self.ax.set_xlabel("Epoch", labelpad=0)
        self.ax.set_ylabel("Loss [% RMSE]")
        self.ax.set_xlim(1, max_epochs)

        # Set x-ticks to ensure a maximum of 10 parts or based on validation frequency
        max_ticks = 10
        if max_epochs <= max_ticks:
            xticks = list(range(1, max_epochs + 1))
        else:
            xticks = list(range(1, max_epochs + 1, max(1, max_epochs // max_ticks)))

        # Ensure the last epoch is included
        if max_epochs not in xticks:
            xticks.append(max_epochs)

        self.ax.set_xticks(xticks)
        self.ax.set_xticklabels(xticks, rotation=45, ha="right")

        self.ax.set_title(
            "Training and Validation Loss",
            fontsize=12,
            fontweight='normal',
            color='#0f0c0c',
            pad=6
        )

        # Initialize the plot lines with empty data
        self.train_line, = self.ax.plot([], [], label='Train Loss')
        self.valid_line, = self.ax.plot([], [], label='Validation Loss')
        self.ax.legend()

        # Attach the Matplotlib figure to the PyQt frame
        self.canvas = FigureCanvas(fig)
        self.canvas.setMinimumSize(600, 300)  # Adjust size if necessary

        # Add the canvas to the main layout
        self.main_layout.addWidget(self.canvas)

        # Adjust margins for the plot
        fig.subplots_adjust(bottom=0.2)

    def setup_log_window(self, task):
        # Create a QTextEdit widget for the log window
        self.log_text = QTextEdit()

        # Set properties of the log window (read-only and word-wrapping)
        self.log_text.setReadOnly(True)  # Log should not be editable by the user
        self.log_text.setLineWrapMode(QTextEdit.WidgetWidth)  # Word wrap

        # Add some padding/margins to make the text more readable
        self.log_text.setStyleSheet("""
            QTextEdit {
                font-size: 10pt;
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                padding: 10px;
            }
        """)

        # Insert initial logs with task repetition details
        self.log_text.append(f"Repetition: {task['hyperparams']['REPETITIONS']}\n")

        # Automatically scroll to the bottom of the log window
        self.log_text.moveCursor(self.log_text.textCursor().End)

        # Add the log window to the main layout
        self.main_layout.addWidget(self.log_text)

    def clear_layout(self):
        # Clear the current layout to rebuild it for new tasks
        if self.centralWidget():
            old_widget = self.centralWidget()
            old_widget.deleteLater()

    def start_task_processing(self):
        if getattr(self, 'training_process_stopped', False):
            self.status_label.setText("Training process has been stopped.")
            self.show_proceed_to_testing_button()
            return

        self.status_label.setText(f"Task {self.current_task_index + 1}/{len(self.task_list)} is running. LSTM model being trained...")
        self.status_label.setStyleSheet("font-size: 12pt; font-weight: bold; color: #004d99;")

        # Start processing tasks sequentially
        self.start_time = time.time()
        self.clear_plot()

        # Start the training task in a background thread
        self.training_thread = TrainingThread(self.task_list[self.current_task_index], self.training_task_manager)

        # Pass the thread reference to the training_task_manager
        self.training_task_manager.training_thread = self.training_thread

        # Connect signals
        self.training_thread.update_epoch_signal.connect(self.update_gui_after_epoch)
        self.training_thread.task_completed_signal.connect(lambda: self.task_completed())
        self.training_thread.task_error_signal.connect(lambda error: self.handle_error(error))

        # Start the thread
        self.training_thread.start()

        # Update the elapsed time and queue processing
        self.update_elapsed_time()
        self.process_queue()


    def update_elapsed_time(self):
        if self.timer_running:
            elapsed_time = time.time() - self.start_time
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)

            # Update the time label to show the elapsed time
            self.time_value_label.setText(f" {int(hours):02}h:{int(minutes):02}m:{int(seconds):02}s")

            # Call this method again after 1 second
            QTimer.singleShot(1000, self.update_elapsed_time)

    def clear_plot(self):
        """Clear the existing plot and reinitialize plot lines for a new task."""
        # Reset the data values for a fresh plot
        self.train_loss_values = []
        self.valid_loss_values = []
        self.valid_x_values = []

        # Ensure the plot axis 'ax' exists (for new tasks or when reinitializing)
        if hasattr(self, 'ax'):
            # Clear the plot and reset labels, titles, and lines
            self.ax.clear()
            self.ax.set_title("Training and Validation Loss", fontsize=12, fontweight='normal', color='#0f0c0c')
            self.ax.set_xlabel("Epoch")
            self.ax.set_ylabel("Loss [% RMSE]")

            # Reinitialize plot lines
            self.train_line, = self.ax.plot([], [], label='Train Loss')
            self.valid_line, = self.ax.plot([], [], label='Validation Loss')
            self.ax.legend()

            # Redraw the canvas to update the cleared plot
            self.canvas.draw()

    def process_queue(self):
        try:
            # Process queue until an exception is raised or task completes
            while True:
                progress_data = self.queue.get_nowait()  # Non-blocking queue retrieval

                # Handle error messages
                if 'task_error' in progress_data:
                    self.handle_error(progress_data['task_error'])
                    break

                # Handle task completion messages
                elif 'task_completed' in progress_data:
                    self.task_completed()
                    break

        except Empty:
            # If the queue is empty, check again after a short delay (100ms)
            QTimer.singleShot(100, self.process_queue)


    def handle_error(self, error_message):
        # Update the status label with the error message
        self.status_label.setText(f"Error: {error_message}")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")

        # Append the error message to the log text
        self.log_text.append(f"Error: {error_message}")

        # Hide the stop button since the task encountered an error
        self.stop_button.hide()


    def update_gui_after_epoch(self, progress_data):
        # Task index and dynamic status for the status label
        task_info = f"Task {self.current_task_index + 1}/{len(self.task_list)}"

        # Always show the task info and append the status
        if 'status' in progress_data:
            self.status_label.setText(f"{task_info} - {progress_data['status']}")
            self.status_label.setStyleSheet("font-size: 11pt; font-weight: bold; color: #004d99;")

        # Handle log updates
        if 'epoch' in progress_data:
            epoch = progress_data['epoch']
            train_loss = progress_data['train_loss']
            val_loss = progress_data['val_loss']
            delta_t_epoch = progress_data['delta_t_epoch']
            learning_rate = progress_data.get('learning_rate', None)
            best_val_loss = progress_data.get('best_val_loss', None)

            # Format the log message using HTML for bold text
            log_message = (
                f"Epoch: <b>{epoch}</b>, "
                f"Train Loss: <b>{train_loss:.4f}</b>, "
                f"Val Loss: <b>{val_loss:.4f}</b>, "
                f"Best Val Loss: <b>{best_val_loss:.4f}</b>, "
                f"Time Per Epoch (ΔT): <b>{delta_t_epoch}s</b>, "
                f"LR: <b>{learning_rate:.1e}</b><br>"
            )
           # WandB logging (only if enabled)
            # if self.wandb_enabled:
            #     try:
            #         wandb.log({
            #             'train_loss': progress_data['train_loss'],
            #             'val_loss': progress_data['val_loss'],
            #             'epoch': progress_data['epoch']
            #         })
            #     except Exception as e:
            #         self.logger.error(f"Failed to log to WandB: {e}")
            self.logger.info(f"Epoch {progress_data['epoch']} | Train Loss: {progress_data['train_loss']} | Val Loss: {progress_data['val_loss']}")

            # Append the log message to the log text widget using rich text format
            self.log_text.append(log_message)

            # Ensure the log scrolls to the bottom
            self.log_text.moveCursor(self.log_text.textCursor().End)
            print(f"Epoch: {epoch}, Train Loss: {train_loss}, Validation Loss: {val_loss}")
            # Update the plot with the new data
            self.train_loss_values.append(train_loss)
            self.valid_loss_values.append(val_loss)
            self.valid_x_values.append(epoch)
            print(f"Valid X Values: {self.valid_x_values}")
            print(f"Train Loss Values: {self.train_loss_values}")
            print(f"Valid Loss Values: {self.valid_loss_values}")

            # Update plot lines with the new data
            self.train_line.set_data(self.valid_x_values, self.train_loss_values)
            self.valid_line.set_data(self.valid_x_values, self.valid_loss_values)

            # Adjust y-axis limits dynamically based on the new data
            self.ax.relim()  # Recompute the limits
            self.ax.autoscale_view(scalex=False, scaley=True)  # Autoscale y-axis only

            # Set fixed x-limits to ensure they remain constant
            max_epochs = int(self.task_list[self.current_task_index]['hyperparams']['MAX_EPOCHS'])
            self.ax.set_xlim(1, max_epochs)

            # Redraw the plot
            self.canvas.draw_idle()
            print("Redrawing the plot")
            # Redraw the plot
            self.canvas.draw_idle()

    def stop_training(self):
        print("Stop training button clicked")

        # Stop the timer
        self.timer_running = False

        # Send stop request to the task manager
        self.training_task_manager.stop_task()
        print("Stop request sent to training task manager")

        # Immediate GUI update to reflect the stopping state
        self.status_label.setText("Stopping Training...")
        self.status_label.setStyleSheet("color: #e75480; font-size: 16pt; font-weight: bold;")  # Pinkish-red text

        # Change stop button appearance and text during the process
        self.stop_button.setText("Stopping...")  # Update button text
        self.stop_button.setStyleSheet("background-color: #ffcccb; color: white; font-size: 12pt; font-weight: bold;")  # Lighter red

        # Set flag to prevent further tasks
        self.training_process_stopped = True
        print(f"Training process stopped flag is now {self.training_process_stopped}")

        # Check if the training thread has finished
        QTimer.singleShot(100, self.check_if_stopped)



    def check_if_stopped(self):
        if self.training_thread and self.training_thread.isRunning():
            # Keep checking until the thread has stopped
            QTimer.singleShot(100, self.check_if_stopped)
        else:
            # Once the thread is confirmed to be stopped, proceed to task completion
            print("Training thread has stopped.")
            
            # Update status to indicate training has stopped early (if it was stopped manually)
            if getattr(self, 'training_process_stopped', False):
                self.status_label.setText("Training stopped early.")
                self.status_label.setStyleSheet("color: #b22222; font-size: 14pt; font-weight: bold;")  # Subtle red color and larger font
            else:
                # In case training completed naturally
                self.status_label.setText("Training completed.")
                self.status_label.setStyleSheet("color: green; font-size: 12pt; font-weight: bold;")

            # Show the "Proceed to Testing" button once the training has stopped
            if not self.task_completed_flag:
                print("Calling task_completed() after training thread has stopped.")
                self.task_completed()
            else:
                print("task_completed() was already called, skipping.")


    def task_completed(self):
        if self.task_completed_flag:
            return  # Exit if this method has already been called for this task
        self.task_completed_flag = True  # Set the flag to True on the first call

        self.timer_running = False

        if self.isVisible():  # Check if the window still exists
            total_training_time = time.time() - self.start_time
            total_hours, total_remainder = divmod(total_training_time, 3600)
            total_minutes, total_seconds = divmod(total_remainder, 60)
            formatted_total_time = f"{int(total_hours):02}h:{int(total_minutes):02}m:{int(total_seconds):02}s"

            # Update time label
            self.static_text_label.setText("Total Training Time:")
            self.static_text_label.setStyleSheet("color: blue; font-size: 12pt; font-weight: bold;")
            self.time_value_label.setText(formatted_total_time)
            self.time_value_label.setStyleSheet("color: purple; font-size: 12pt; font-weight: bold;")

            # Check if the training process was stopped early
            if getattr(self, 'training_process_stopped', False):
                self.status_label.setText("Training stopped early. Saving model to task folder...")
                self.status_label.setStyleSheet("color: #b22222; font-size: 14pt; font-weight: bold;")  # Reddish color
            else:
                self.status_label.setText("All Training Tasks Completed!")
                self.status_label.setStyleSheet("color: green; font-size: 12pt; font-weight: bold;")

            # Ensure the "Proceed to Testing" button is displayed in both cases
            self.stop_button.hide()
            self.show_proceed_to_testing_button()

        # Handle the case where the window has been destroyed
        else:
            print("Task completed method was called after the window was destroyed.")

        # Check if there are more tasks to process
        if self.current_task_index < len(self.task_list) - 1:
            print(f"Completed task {self.current_task_index + 1}/{len(self.task_list)}.")
            self.current_task_index += 1
            self.task_completed_flag = False  # Reset the flag for the next task
            self.build_gui(self.task_list[self.current_task_index])
            self.start_task_processing()
        else:
            # Handle the case when all tasks are completed
            total_training_time = time.time() - self.start_time
            total_hours, total_remainder = divmod(total_training_time, 3600)
            total_minutes, total_seconds = divmod(total_remainder, 60)
            formatted_total_time = f"{int(total_hours):02}h:{int(total_minutes):02}m:{int(total_seconds):02}s"

            self.static_text_label.setText("Total Training Time:")
            self.time_value_label.setText(formatted_total_time)

            self.status_label.setText("All Training Tasks Completed!")
            self.show_proceed_to_testing_button()

    def wait_for_thread_to_stop(self):
        if self.worker and self.worker.isRunning():
            # Continue checking until the thread has stopped
            QTimer.singleShot(100, self.wait_for_thread_to_stop)
        else:
            # Once the thread is confirmed to be stopped
            print("Training thread has stopped, now closing the window.")
            self.close()  # Close the window

    def on_closing(self):
        if self.worker and self.worker.isRunning():
            print("Stopping training before closing...")
            self.stop_training()  # Stop the training thread
            QTimer.singleShot(100, self.wait_for_thread_to_stop)
        else:
            self.close()  # Close the window
    
    def show_proceed_to_testing_button(self):
        # Ensure the button is shown
        self.stop_button.hide()
        self.proceed_button.show()

    def transition_to_testing_gui(self):
        self.close()  # Close the current window
        self.testing_gui = VEstimTestingGUI()  # Initialize the testing GUI
        self.testing_gui.show()  # Show the testing GUI

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    task_list = []  # Replace with actual task list
    params = {}  # Replace with actual parameters
    sys.exit(app.exec_())
