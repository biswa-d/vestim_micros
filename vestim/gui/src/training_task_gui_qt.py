from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QHBoxLayout, QVBoxLayout, QPushButton, 
    QWidget, QFrame, QTextEdit, QGridLayout, QGroupBox, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal

# Make PyTorch optional in the frontend
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available in frontend. This is normal - PyTorch is only required in the backend.")

import json, time
import numpy as np
from matplotlib.figure import Figure
import matplotlib.ticker as ticker
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from queue import Queue, Empty
from threading import Thread
import logging, wandb
import os
import matplotlib.pyplot as plt

# Import local services
from vestim.gui.src.api_gateway import APIGateway
from vestim.gui.src.testing_gui_qt import VEstimTestingGUI

class TrainingThread(QThread):
    update_signal = pyqtSignal(dict)  # Signal to send progress data
    finished_signal = pyqtSignal(dict)  # Signal when the task is completed
    error_signal = pyqtSignal(str)  # Signal for any error during the task

    def __init__(self, job_id, task_id, api_gateway):
        super().__init__()
        self.job_id = job_id
        self.task_id = task_id
        self.api_gateway = api_gateway
        self.is_cancelled = False
        self.logger = logging.getLogger(__name__)

    def run(self):
        try:
            # Start the training task via API
            response = self.api_gateway.post(f"jobs/{self.job_id}/tasks/{self.task_id}/start")
            if response.get("status") != "success":
                self.error_signal.emit(f"Failed to start task: {response.get('message', 'Unknown error')}")
                return
                
            # Poll for updates until task is completed
            while not self.is_cancelled:
                status_response = self.api_gateway.get(f"jobs/{self.job_id}/tasks/{self.task_id}/status")
                
                # Send update signal with the latest status
                self.update_signal.emit(status_response)
                
                # Check if task is complete
                task_status = status_response.get("status")
                if task_status in ["completed", "failed", "stopped"]:
                    self.finished_signal.emit(status_response)
                    break
                    
                # Wait before polling again
                time.sleep(2)
                
        except Exception as e:
            self.logger.error(f"Error in training thread: {e}", exc_info=True)
            self.error_signal.emit(str(e))

    def cancel(self):
        self.is_cancelled = True

class VEstimTrainingTaskGUI(QMainWindow):
    def __init__(self, api_gateway: APIGateway, job_id: str):
        super().__init__()
        self.api = api_gateway
        self.job_id = job_id
        self.logger = logging.getLogger(__name__)
        self.train_loss_values = []
        self.valid_loss_values = []
        self.epoch_points = []
        self.current_task_index = 0  # Add missing attribute
        
        # Verify job exists and get training tasks
        try:
            job_info = self.api.get(f"jobs/{self.job_id}")
            self.job_info = job_info
            self.hyperparams = job_info.get("details", {}).get("hyperparameters", {})
            
            # Check if we have training tasks
            self.training_tasks = job_info.get("details", {}).get("training_tasks", [])
            if not self.training_tasks:
                self.logger.warning(f"No training tasks found for job {self.job_id}")
                
            self.logger.info(f"Successfully initialized training task GUI for job {self.job_id}")
        except Exception as e:
            self.logger.error(f"Error retrieving job {self.job_id} information: {e}", exc_info=True)
            self.job_info = {}
            self.hyperparams = {}
            self.training_tasks = []        
        self.initUI()
        self.start_polling()

    def initUI(self):
        self.setWindowTitle(f"VEstim - Training Job: {self.job_id}")
        self.setGeometry(100, 100, 900, 700)
        
        container = QWidget()
        self.setCentralWidget(container)
        self.main_layout = QVBoxLayout(container)

        # Title
        title_label = QLabel("Training Job Progress")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 16pt; font-weight: bold;")
        self.main_layout.addWidget(title_label)

        # Status
        self.status_label = QLabel("Connecting...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.status_label)

        # Plot
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.train_line, = self.ax.plot([], [], 'r-', label='Train Loss')
        self.valid_line, = self.ax.plot([], [], 'b-', label='Validation Loss')
        self.ax.legend()
        self.main_layout.addWidget(self.canvas)

        # Log
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.main_layout.addWidget(self.log_text)

        # Stop Button
        self.stop_button = QPushButton("Stop Training")
        self.stop_button.clicked.connect(self.stop_training)
        self.main_layout.addWidget(self.stop_button)
        
        # Initialize the Proceed to Testing button
        self.proceed_button = QPushButton("Proceed to Testing")
        self.proceed_button.setStyleSheet("""
            background-color: #0b6337;
            color: white;
            font-size: 12pt;
            font-weight: bold;
            padding: 10px 20px;
        """)
        self.proceed_button.hide()
        self.proceed_button.clicked.connect(self.transition_to_testing_gui)
          # Layout for proceed button
        proceed_button_layout = QHBoxLayout()
        proceed_button_layout.addStretch(1)
        proceed_button_layout.addWidget(self.proceed_button)
        proceed_button_layout.addStretch(1)
        self.main_layout.addLayout(proceed_button_layout)

    def start_polling(self):
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self.update_status)
        self.poll_timer.start(2000) # Poll every 2 seconds
        self.update_status()

    def update_status(self):
        try:
            status_data = self.api.get_job(self.job_id)
            
            if not status_data:
                self.status_label.setText("Could not fetch job status.")
                return

            status = status_data.get('status')
            details = status_data.get('details', {})
            self.status_label.setText(f"Status: {status} - {details.get('message', '')}")

            if 'history' in details and details['history']:
                self.update_plot(details['history'])
                self.update_log(details['history'])

            if status in ['complete', 'error', 'stopped']:
                self.poll_timer.stop()
                self.stop_button.setEnabled(False)
                self.stop_button.setText(f"Job {status.capitalize()}")

        except Exception as e:
            self.logger.error(f"Error updating status for job {self.job_id}: {e}")
            self.status_label.setText(f"Error: {e}")
            self.poll_timer.stop()

    def update_plot(self, history):
        epochs = [item['epoch'] for item in history]
        train_loss = [item['train_loss'] for item in history]
        val_loss = [item['val_loss'] for item in history]

        self.train_line.set_data(epochs, train_loss)
        self.valid_line.set_data(epochs, val_loss)
        
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()
        
    def update_log(self, history):
        self.log_text.clear()
        for item in history:
            self.log_text.append(f"Epoch {item['epoch']}: {item['message']}")
            
    def stop_training(self):
        try:
            self.api.stop_job(self.job_id)
            self.status_label.setText("Stop signal sent. Waiting for confirmation...")
            self.stop_button.setEnabled(False)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to send stop signal: {e}")

    def closeEvent(self, event):
        # Stop the poll timer if it exists
        if hasattr(self, 'poll_timer') and self.poll_timer.isActive():
            self.poll_timer.stop()
            
        # Stop any running training threads
        if hasattr(self, 'training_thread') and self.training_thread is not None:
            try:
                if self.training_thread.isRunning():
                    self.training_thread.cancel()  # Set the cancellation flag
                    self.training_thread.quit()
                    self.training_thread.wait(3000)  # Wait up to 3 seconds for thread to finish
            except RuntimeError:
                # QThread object has already been deleted by Qt - this is normal during close
                pass
            
        super().closeEvent(event)

    def transition_to_testing_gui(self):
        """Transition to the testing GUI after training is complete."""
        try:
            testing_gui = VEstimTestingGUI(api_gateway=self.api, job_id=self.job_id)
            testing_gui.show()
            self.close()
        except Exception as e:
            self.logger.error(f"Error while transitioning to the testing screen: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open testing interface: {e}")

    def build_gui(self, task):
        # Create a main widget to set as central widget in QMainWindow
        container = QWidget()
        self.setCentralWidget(container)

        # Create a main layout
        self.main_layout = QVBoxLayout()

        # Title Label
        title_label = QLabel("Training LSTM Model with Hyperparameters")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 16pt; font-weight: bold;")
        self.main_layout.addWidget(title_label)

        # Display hyperparameters
        # Initialize the hyperparameter frame
        self.hyperparam_frame = QFrame(self)
        self.hyperparam_frame.setLayout(QVBoxLayout())
        self.main_layout.addWidget(self.hyperparam_frame)
        self.display_hyperparameters(task['hyperparams'])

        # Status Label
        self.status_label = QLabel("Starting training...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.status_label)

        # Time Frame and Plot Setup
        self.setup_time_and_plot(task)

        # Add log window
        log_group = QGroupBox("Training Log")
        log_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 12px; }")
        log_layout = QVBoxLayout()
        
        # Create the log text widget
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setLineWrapMode(QTextEdit.WidgetWidth)
        self.log_text.setStyleSheet("""
            QTextEdit {
                font-size: 10pt;
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                padding: 10px;
            }
        """)
        self.log_text.setMinimumHeight(150)  # Set minimum height for better visibility
        
        # Add initial log entry
        self.log_text.append(f"Repetition: {task['hyperparams']['REPETITIONS']}\n")
        
        # Add log widget to layout
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        self.main_layout.addWidget(log_group)

        # Stop button (centered and styled)
        self.stop_button = QPushButton("Stop Training")
        self.stop_button.setStyleSheet("background-color: red; color: white; font-size: 12pt; font-weight: bold;")
        self.stop_button.setFixedWidth(150)
        self.stop_button.clicked.connect(self.stop_training)

        # Layout for stop button
        stop_button_layout = QHBoxLayout()
        stop_button_layout.addStretch(1)
        stop_button_layout.addWidget(self.stop_button)
        stop_button_layout.addStretch(1)
        self.main_layout.addLayout(stop_button_layout)

        # Initialize the Proceed to Testing button
        self.proceed_button = QPushButton("Proceed to Testing")
        self.proceed_button.setStyleSheet("""
            background-color: #0b6337;
            color: white;
            font-size: 12pt;
            font-weight: bold;
            padding: 10px 20px;
        """)
        self.proceed_button.hide()
        self.proceed_button.clicked.connect(self.transition_to_testing_gui)
        
        # Layout for proceed button
        proceed_button_layout = QHBoxLayout()
        proceed_button_layout.addStretch(1)
        proceed_button_layout.addWidget(self.proceed_button)
        proceed_button_layout.addStretch(1)
        self.main_layout.addLayout(proceed_button_layout)

        # Set the layout
        container.setLayout(self.main_layout)

    def display_hyperparameters(self, task_params):
        # Clear previous widgets in the hyperparam_frame layout
        layout = self.hyperparam_frame.layout()
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        hyperparam_layout = QGridLayout()
        
        display_items_ordered = []
        processed_keys = set()

        # Define preferred order and sections
        # Section 1: Model Architecture
        model_arch_keys = ['MODEL_TYPE', 'LAYERS', 'HIDDEN_UNITS', 'INPUT_SIZE', 'OUTPUT_SIZE', 'NUM_LEARNABLE_PARAMS']
        # Section 2: Training Method
        train_method_keys = ['TRAINING_METHOD', 'LOOKBACK', 'BATCH_TRAINING', 'BATCH_SIZE']
        # Section 3: Training Control
        train_control_keys = ['MAX_EPOCHS', 'INITIAL_LR', 'SCHEDULER_TYPE', 'VALID_PATIENCE', 'ValidFrequency', 'REPETITIONS']
        # Section 4: Execution Environment
        exec_env_keys = ['DEVICE_SELECTION', 'MAX_TRAINING_TIME_SECONDS']

        preferred_order = model_arch_keys + train_method_keys + train_control_keys + exec_env_keys

        # Helper to format scheduler string
        def get_scheduler_display_val(params):
            scheduler_type = params.get('SCHEDULER_TYPE')
            display_val = scheduler_type if scheduler_type else "N/A"
            if scheduler_type == 'StepLR':
                period = params.get('LR_DROP_PERIOD', params.get('LR_PERIOD', 'N/A')) # Check both keys
                factor = params.get('LR_DROP_FACTOR', params.get('LR_PARAM', 'N/A'))
                display_val = f"StepLR (Period: {period}, Factor: {factor})"
            elif scheduler_type == 'ReduceLROnPlateau':
                patience = params.get('PLATEAU_PATIENCE', 'N/A')
                factor = params.get('PLATEAU_FACTOR', params.get('LR_PARAM', 'N/A')) # Check old key too
                display_val = f"ReduceLROnPlateau (Patience: {patience}, Factor: {factor})"
            return display_val

        # Add items in preferred order
        for key in preferred_order:
            if key in task_params or (key == 'DEVICE_SELECTION' and key in self.params):
                label_text = self.param_labels.get(key, key.replace("_", " ").title())
                
                if key == 'DEVICE_SELECTION':
                    value = self.params.get(key, 'N/A')
                elif key == 'MAX_TRAINING_TIME_SECONDS':
                    max_time_sec = task_params.get(key, 0)
                    if isinstance(max_time_sec, str):
                        try: max_time_sec = int(max_time_sec)
                        except ValueError: max_time_sec = 0
                    h = max_time_sec // 3600
                    m = (max_time_sec % 3600) // 60
                    s = max_time_sec % 60
                    value = f"{h:02d}H:{m:02d}M:{s:02d}S"
                elif key == 'SCHEDULER_TYPE':
                    value = get_scheduler_display_val(task_params)
                else:
                    value = task_params.get(key)

                value_str = str(value)
                if isinstance(value, list) and len(value) > 2:
                    display_value = f"[{value[0]}, {value[1]}, ...]"
                elif isinstance(value, str) and "," in value_str and key not in ['SCHEDULER_TYPE']: # Don't truncate scheduler string
                    parts = value_str.split(",")
                    if len(parts) > 2: display_value = f"{parts[0]},{parts[1]},..."
                    else: display_value = value_str
                else:
                    display_value = value_str
                
                display_items_ordered.append((label_text, display_value))
                processed_keys.add(key)
        
        # Add any remaining params not in preferred order (excluding specific scheduler sub-params)
        scheduler_sub_params = {'LR_DROP_PERIOD', 'LR_PERIOD', 'LR_PARAM', 'LR_DROP_FACTOR', 'PLATEAU_PATIENCE', 'PLATEAU_FACTOR'}
        for key, value in task_params.items():
            if key not in processed_keys and key not in scheduler_sub_params:
                label_text = self.param_labels.get(key, key.replace("_", " ").title())
                value_str = str(value)
                # Basic truncation for lists or long comma-separated strings
                if isinstance(value, list) and len(value) > 2: display_value = f"[{value[0]}, {value[1]}, ...]"
                elif isinstance(value, str) and "," in value_str and len(value_str.split(",")) > 2 : display_value = f"{value_str.split(',')[0]},{value_str.split(',')[1]},..."
                else: display_value = value_str
                display_items_ordered.append((label_text, display_value))

        # Display items in a grid
        items_per_row_display = 5
        for idx, (label, val) in enumerate(display_items_ordered):
            row = idx // items_per_row_display
            col_label = (idx % items_per_row_display) * 2
            col_value = col_label + 1
            
            param_label_widget = QLabel(f"{label}:")
            value_label_widget = QLabel(str(val))
            param_label_widget.setStyleSheet("font-size: 10pt;")
            value_label_widget.setStyleSheet("font-size: 10pt; font-weight: bold;")

            hyperparam_layout.addWidget(param_label_widget, row, col_label)
            hyperparam_layout.addWidget(value_label_widget, row, col_value)

        layout.addLayout(hyperparam_layout)


    def setup_time_and_plot(self, task):
        # Debugging statement to check the structure of the task
        # print(f"Current task: {task}") # Too verbose for regular logs
        # print(f"Hyperparameters in the task: {task['hyperparams']}") # Also verbose
        self.logger.info(f"Setting up time and plot for task_id: {task.get('task_id', 'N/A')}")

        # Time Layout
        time_layout = QHBoxLayout()

        # Time tracking label (move it just below the title)
        time_layout = QHBoxLayout()
        self.static_text_label = QLabel("Current Task Time:") # Changed label
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

        # Plot Setup
        max_epochs = int(task['hyperparams']['MAX_EPOCHS'])
        valid_frequency = int(task['hyperparams']['ValidFrequency'])

        # Matplotlib figure setup
        fig = Figure(figsize=(6, 2.5), dpi=100)
        self.ax = fig.add_subplot(111)
        self.ax.set_xlabel("Epoch", labelpad=0)
        self.ax.set_ylabel(self.current_error_unit_label) # Use dynamic label
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
        self.training_thread = TrainingThread(self.job_id, self.task_list[self.current_task_index]['task_id'], self.api_gateway)
        self.training_thread.update_epoch_signal.connect(self.update_gui_after_epoch)
        self.training_thread.task_completed_signal.connect(lambda: self.task_completed())
        self.training_thread.task_error_signal.connect(lambda error: self.handle_error(error))
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
        self.epoch_points = []

        # Ensure the plot axis 'ax' exists (for new tasks or when reinitializing)
        if hasattr(self, 'ax'):
            # Clear the plot and reset labels, titles, and lines
            self.ax.clear()
            self.ax.set_title("Training and Validation Loss", fontsize=12, fontweight='normal', color='#0f0c0c')
            self.ax.set_xlabel("Epoch")
            self.ax.set_ylabel(self.current_error_unit_label) # Use dynamic label
 
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
        self.logger.error(f"An error occurred during training: {error_message}")
        self.status_label.setText(f"Error: {error_message}")
        self.status_label.setStyleSheet("font-size: 12pt; font-weight: bold; color: red;")
        self.timer_running = False
        self.show_proceed_to_testing_button()

    def update_gui_after_epoch(self, progress_data):
        epoch = progress_data.get('epoch')
        train_loss = progress_data.get('train_loss')
        valid_loss = progress_data.get('valid_loss')
        
        if epoch is not None and train_loss is not None:
            self.train_loss_values.append(train_loss)
            self.epoch_points.append(epoch)
            self.train_line.set_data(self.epoch_points, self.train_loss_values)
            
            if valid_loss is not None:
                self.valid_loss_values.append(valid_loss)
                self.valid_x_values.append(epoch)
                self.valid_line.set_data(self.valid_x_values, self.valid_loss_values)

            self.ax.relim()
            self.ax.autoscale_view()
            self.canvas.draw()
            
            self.status_label.setText(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")
            self.log_text.append(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Valid Loss={valid_loss:.4f}")
            
            if self.wandb_enabled:
                wandb.log({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})

    def stop_training(self):
        """Stops the training process by setting a flag."""
        self.logger.info("Stop button clicked. Attempting to stop training...")
        self.training_process_stopped = True
        self.timer_running = False  # Stop the timer
        
        try:
            self.api_gateway.post(f"jobs/{self.job_id}/stop")
            self.status_label.setText("Training stopped by user.")
            self.status_label.setStyleSheet("font-size: 12pt; font-weight: bold; color: orange;")
            self.show_proceed_to_testing_button()
        except Exception as e:
            self.logger.error(f"Failed to stop training: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to stop training: {e}")
    def check_if_stopped(self):
        """
        Checks if the training process has been stopped by the user.
        This method is called periodically by the training service.
        """
        return self.training_process_stopped

    def task_completed(self):
        if self.task_completed_flag:
            return
        self.task_completed_flag = True
        self.timer_running = False
        
        try:
            final_status = self.api_gateway.get(f"jobs/{self.job_id}/status")
            if final_status and final_status.get("status") == "complete":
                self.status_label.setText("Training completed successfully.")
                self.status_label.setStyleSheet("font-size: 12pt; font-weight: bold; color: green;")
                self.show_proceed_to_testing_button()
            else:
                self.handle_error("Task failed or was stopped.")
        except Exception as e:
            self.handle_error(str(e))

    def wait_for_thread_to_stop(self):
        if hasattr(self, 'training_thread') and self.training_thread.isRunning():
            self.training_thread.quit()
            self.training_thread.wait()

    def show_proceed_to_testing_button(self):
        self.proceed_button.show()
        self.stop_button.hide()
