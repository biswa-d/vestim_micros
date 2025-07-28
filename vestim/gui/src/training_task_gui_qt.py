from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QHBoxLayout, QVBoxLayout, QPushButton, QWidget, QFrame, QTextEdit, QGridLayout, QGroupBox, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
import torch
import json, time
import numpy as np
from matplotlib.figure import Figure
import matplotlib.ticker as ticker
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from queue import Queue, Empty
from threading import Thread
import logging
import os
import matplotlib.pyplot as plt

# Import local services
from vestim.services.model_training.src.training_task_service import TrainingTaskService
from vestim.gateway.src.training_task_manager_qt import TrainingTaskManager
from vestim.gateway.src.training_setup_manager_qt import VEstimTrainingSetupManager
from vestim.gateway.src.job_manager_qt import JobManager
from vestim.gui.src.testing_gui_qt import VEstimTestingGUI
from vestim.gateway.src.testing_manager_qt import VEstimTestingManager


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
    def __init__(self, job_manager=None, task_list=None, params=None):
        super().__init__()
        
        #Logger setup
        self.logger = logging.getLogger(__name__)
        # Initialize WandB flag
        self.use_wandb = False  # WandB functionality removed
        self.wandb_enabled = False
        

        self.task_list = task_list
        self.params = params # Assign to self.params first

        self.job_manager = job_manager if job_manager else JobManager()
        self.training_task_manager = TrainingTaskManager(job_manager=self.job_manager, global_params=self.params) # Now use self.params
        self.training_setup_manager = VEstimTrainingSetupManager(job_manager=self.job_manager)
        self.training_service = TrainingTaskService()

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
        self.current_error_unit_label = "RMS Error" # Default error label
        self.training_results = {}
        self.auto_proceed_timer = QTimer(self)
        self.auto_proceed_timer.setSingleShot(True)
        self.auto_proceed_timer.timeout.connect(self.transition_to_testing_gui)
 
        self.param_labels = {
            "LAYERS": "Layers",
            "HIDDEN_UNITS": "Hidden Units",
            "HIDDEN_LAYER_SIZES": "Hidden Layers",  # For FNN
            "DROPOUT_PROB": "Dropout Prob",         # For FNN
            "BATCH_SIZE": "Batch Size",
            "MAX_EPOCHS": "Max Epochs",
            "INITIAL_LR": "Initial LR", # Shorter
            "LR_DROP_FACTOR": "LR Drop Factor", # Keep for StepLR specific
            "LR_DROP_PERIOD": "LR Drop Period", # Keep for StepLR specific
            "PLATEAU_PATIENCE": "Plateau Patience", # For ReduceLROnPlateau
            "PLATEAU_FACTOR": "Plateau Factor",   # For ReduceLROnPlateau
            "VALID_PATIENCE": "Validation Patience",
            "VALID_FREQUENCY": "Validation Freq", # Shorter
            "LOOKBACK": "Lookback", # Shorter
            "REPETITIONS": "Repetitions",
            "NUM_LEARNABLE_PARAMS": "# Params", # Shorter
            "INPUT_SIZE": "Input Size",
            "OUTPUT_SIZE": "Output Size",
            "SCHEDULER_TYPE": "LR Scheduler",
            "TRAINING_METHOD": "Training Method",
            "DEVICE_SELECTION": "Device", # Added
            "MAX_TRAINING_TIME_SECONDS": "Max Train Time (Task)", # Added
            # Add other specific keys from hyperparams if they need special display names
            # e.g. BATCH_TRAINING, MODEL_TYPE are already strings from QComboBox
        }

        self.initUI()
        self.build_gui(self.task_list[self.current_task_index])
        self.start_task_processing()

    def initUI(self):
        self.setWindowTitle(f"VEstim - Training Task {self.current_task_index + 1}")
        self.setGeometry(100, 100, 1200, 800)

        # Add a global stylesheet for disabled buttons
        self.setStyleSheet("""
            QPushButton:disabled {
                background-color: #d3d3d3;
                color: #a9a9a9;
            }
        """)

    def build_gui(self, task):
        # Create a main widget to set as central widget in QMainWindow
        container = QWidget()
        self.setCentralWidget(container)

        # Create a main layout
        self.main_layout = QVBoxLayout()

        # Title Label with dynamic model type
        model_type = self.params.get("MODEL_TYPE", "Model")  # Get model type from params
        title_label = QLabel(f"Training {model_type} Model with Hyperparameters")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #0b6337; margin-bottom: 15px;")
        self.main_layout.addWidget(title_label)

        # Display hyperparameters
        # Initialize the hyperparameter frame
        self.hyperparam_frame = QFrame(self)
        self.hyperparam_frame.setObjectName("hyperparamFrame")
        self.hyperparam_frame.setStyleSheet("""
            #hyperparamFrame {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                background-color: #ffffff;
            }
        """)
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
        hyperparam_layout.setContentsMargins(15, 15, 15, 15)
        hyperparam_layout.setHorizontalSpacing(15)
        hyperparam_layout.setVerticalSpacing(10)
        
        display_items_ordered = []
        processed_keys = set()
        
        # Get model type to determine which parameters to display
        model_type = task_params.get('MODEL_TYPE', 'LSTM')

        # Define preferred order and sections - model-type aware
        # Section 1: Model Architecture
        model_arch_keys = ['MODEL_TYPE', 'INPUT_SIZE', 'OUTPUT_SIZE', 'NUM_LEARNABLE_PARAMS']
        if model_type in ['LSTM', 'GRU']:
            model_arch_keys.extend(['LAYERS', 'HIDDEN_UNITS'])
        elif model_type == 'FNN':
            model_arch_keys.extend(['HIDDEN_LAYER_SIZES', 'DROPOUT_PROB'])
            
        # Section 2: Training Method
        train_method_keys = ['TRAINING_METHOD', 'LOOKBACK', 'BATCH_TRAINING', 'BATCH_SIZE']
        # Section 3: Training Control
        train_control_keys = ['MAX_EPOCHS', 'INITIAL_LR', 'SCHEDULER_TYPE', 'VALID_PATIENCE', 'VALID_FREQUENCY', 'REPETITIONS']
        # Section 4: Execution Environment
        exec_env_keys = ['DEVICE_SELECTION', 'MAX_TRAINING_TIME_SECONDS']
        # Section 5: Data Columns
        data_keys = ['FEATURE_COLUMNS', 'TARGET_COLUMN']
        
        self.param_labels['FEATURE_COLUMNS'] = "Feature Columns"
        self.param_labels['TARGET_COLUMN'] = "Target Column"

        preferred_order = model_arch_keys + train_method_keys + train_control_keys + exec_env_keys + data_keys

        # Helper to format scheduler string
        def get_scheduler_display_val(params):
            scheduler_type = params.get('SCHEDULER_TYPE')
            display_val = scheduler_type if scheduler_type else "N/A"
            if scheduler_type == 'StepLR':
                period = params.get('LR_DROP_PERIOD', params.get('LR_PERIOD', 'N/A')) # Check both keys
                factor_val = params.get('LR_DROP_FACTOR', params.get('LR_PARAM', 'N/A'))
                try:
                    factor_str = f"{float(factor_val):.2f}"
                except (ValueError, TypeError):
                    factor_str = str(factor_val)
                display_val = f"StepLR (Period: {period}, Factor: {factor_str})"
            elif scheduler_type == 'ReduceLROnPlateau':
                patience = params.get('PLATEAU_PATIENCE', 'N/A')
                factor_val = params.get('PLATEAU_FACTOR', params.get('LR_PARAM', 'N/A')) # Check old key too
                try:
                    factor_str = f"{float(factor_val):.2f}"
                except (ValueError, TypeError):
                    factor_str = str(factor_val)
                display_val = f"ReduceLROnPlateau (Patience: {patience}, Factor: {factor_str})"
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
                    
                    if max_time_sec > 0:
                        h = max_time_sec // 3600
                        m = (max_time_sec % 3600) // 60
                        s = max_time_sec % 60
                        value = f"{h:02d}H:{m:02d}M:{s:02d}S"
                        display_items_ordered.append((label_text, value))
                    
                    processed_keys.add(key)
                    continue # Continue to next key, skipping the generic display logic below

                elif key == 'SCHEDULER_TYPE':
                    value = get_scheduler_display_val(task_params)
                else:
                    value = task_params.get(key)
                
                if key == 'INITIAL_LR':
                    try:
                        value_str = f"{float(value):.2e}"
                    except (ValueError, TypeError):
                        value_str = str(value)
                else:
                    value_str = str(value)

                if key == 'FEATURE_COLUMNS':
                    features = task_params.get(key, [])
                    if len(features) > 4:
                        display_value = f"[{', '.join(features[:4])}, ...]"
                    else:
                        display_value = str(features)
                elif isinstance(value, list) and len(value) > 2:
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
        items_per_row_display = 4
        for idx, (label, val) in enumerate(display_items_ordered):
            row = idx // items_per_row_display
            col_label = (idx % items_per_row_display) * 2
            col_value = col_label + 1
            
            param_label_widget = QLabel(f"{label}:")
            value_label_widget = QLabel(str(val))
            param_label_widget.setStyleSheet("font-size: 9pt; color: #333;")
            value_label_widget.setStyleSheet("font-size: 9pt; color: #000000; font-weight: bold;")

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

        # Time tracking labels (current task and total job time)
        # Current Task Time
        task_time_layout = QHBoxLayout()
        self.static_text_label = QLabel("Current Task Time:") # Changed label
        self.static_text_label.setStyleSheet("color: blue; font-size: 10pt;")
        self.time_value_label = QLabel("00h:00m:00s")
        self.time_value_label.setStyleSheet("color: purple; font-size: 11pt; font-weight: bold;")
        # Align both the label and the value in the same row, close to each other
        task_time_layout.addStretch(1)  # Adds space to push both labels to the center
        task_time_layout.addWidget(self.static_text_label)
        task_time_layout.addWidget(self.time_value_label)
        task_time_layout.addStretch(1)  # Adds space after the labels to keep them centered
        
        # Total Job Time
        job_time_layout = QHBoxLayout()
        self.job_text_label = QLabel("Total Training Time:")
        self.job_text_label.setStyleSheet("color: green; font-size: 10pt;")
        self.job_time_value_label = QLabel("00h:00m:00s")
        self.job_time_value_label.setStyleSheet("color: darkgreen; font-size: 11pt; font-weight: bold;")
        # Align both the label and the value in the same row, close to each other
        job_time_layout.addStretch(1)  # Adds space to push both labels to the center
        job_time_layout.addWidget(self.job_text_label)
        job_time_layout.addWidget(self.job_time_value_label)
        job_time_layout.addStretch(1)  # Adds space after the labels to keep them centered
        
        # Add both time layouts to the main layout
        self.main_layout.addLayout(task_time_layout)
        self.main_layout.addLayout(job_time_layout)

        # Plot Setup
        max_epochs = int(task['hyperparams']['MAX_EPOCHS'])
        valid_frequency = int(task['hyperparams'].get('VALID_FREQUENCY', 1))

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
        # Only set start_time for the first task (job start time)
        if self.current_task_index == 0:
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
            # Calculate total job elapsed time for smooth real-time display
            elapsed_time = time.time() - self.start_time
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)

            # Update the job time label (Total Training Time) with smooth real-time updates
            self.job_time_value_label.setText(f" {int(hours):02}h:{int(minutes):02}m:{int(seconds):02}s")
            
            # Note: Current Task Time is updated by backend data in update_gui_after_epoch()
            # This provides accurate task-specific timing while job timer updates smoothly

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
                    self.task_completed(progress_data)
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
            # Use the new scaled RMSE values and error label from progress_data
            train_rmse_scaled = progress_data.get('train_rmse_scaled', float('nan'))
            val_rmse_scaled = progress_data.get('val_rmse_scaled', float('nan'))
            best_val_rmse_scaled = progress_data.get('best_val_rmse_scaled', float('nan'))
            error_unit_label = progress_data.get('error_unit_label', "RMS Error") # Default if not provided
            self.current_error_unit_label = error_unit_label # Update instance variable

            patience_counter = progress_data.get('patience_counter', None)
            delta_t_epoch = progress_data['delta_t_epoch']
            learning_rate = progress_data.get('learning_rate', None)

            # Format the log message using HTML for bold text
            log_message = (
                f"Epoch: <b>{epoch}</b>, "
                f"Train {error_unit_label.split('[')[0].strip()}: <b>{train_rmse_scaled:.2f}</b> {error_unit_label.split('[')[-1].replace(']','').strip()}, "
                f"Val {error_unit_label.split('[')[0].strip()}: <b>{val_rmse_scaled:.2f}</b> {error_unit_label.split('[')[-1].replace(']','').strip()}, "
                f"Best Val {error_unit_label.split('[')[0].strip()}: <b>{best_val_rmse_scaled:.2f}</b> {error_unit_label.split('[')[-1].replace(']','').strip()}, "
                f"Time Per Epoch (ΔT): <b>{delta_t_epoch}s</b>, "
                f"LR: <b>{learning_rate:.1e}</b>, "
                f"Patience Counter: <b>{patience_counter}</b><br>"
            )

            # Append the log message to the log text widget using rich text
            self.log_text.append(log_message)

            # Ensure the log scrolls to the bottom
            self.log_text.moveCursor(self.log_text.textCursor().End)
            
            # Update timer display with task elapsed time from progress data
            if 'formatted_task_time' in progress_data:
                self.time_value_label.setText(f" {progress_data['formatted_task_time']}")
                # Note: Current Task Time is updated from backend for accuracy
            
            # Job timer (Total Training Time) is updated by GUI's real-time timer for smooth updates
            # No need to update job_time_value_label here

            # Update the plot data with actual epoch numbers
            if not hasattr(self, 'epoch_points'):
                self.epoch_points = []
            self.epoch_points.append(epoch)
            self.train_loss_values.append(train_rmse_scaled if not np.isnan(train_rmse_scaled) else 0) # Plot 0 for NaN to avoid issues
            self.valid_loss_values.append(val_rmse_scaled if not np.isnan(val_rmse_scaled) else 0) # Plot 0 for NaN

            # Update the plot
            self.ax.clear()
            
            # Plot the data using actual epoch numbers
            self.ax.plot(self.epoch_points, self.train_loss_values, label='Training', color='blue', marker='.')
            self.ax.plot(self.epoch_points, self.valid_loss_values, label='Validation', color='red', marker='.')
            
            # Set y-axis to log scale
            self.ax.set_yscale('log')
            
            # Keep x-axis fixed to max_epochs
            max_epochs = int(self.task_list[self.current_task_index]['hyperparams']['MAX_EPOCHS'])
            self.ax.set_xlim(1, max_epochs)
            
            # Set x-ticks to be integers
            num_ticks = min(10, max_epochs)  # Show at most 10 ticks
            step = max(1, max_epochs // num_ticks)
            ticks = list(range(1, max_epochs + 1, step))
            if max_epochs not in ticks:
                ticks.append(max_epochs)
            self.ax.set_xticks(ticks)
            self.ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            
            # Dynamically adjust only y-axis limits
            all_values = self.train_loss_values + self.valid_loss_values
            if all_values:
                y_min_calculated = min(all_values) * 0.8
                y_max_calculated = max(all_values) * 1.2

                if self.ax.get_yscale() == 'log':
                    if y_min_calculated <= 0:
                        positive_values = [val for val in all_values if val > 0]
                        if positive_values:
                            y_min_final = min(positive_values) * 0.1 # Adjust factor as needed
                            if y_min_final <= 0: # Still non-positive, use a tiny epsilon
                                y_min_final = 1e-9
                        else: # No positive values at all (e.g. all are zero)
                            y_min_final = 1e-9 # Fallback to a tiny positive number
                    else:
                        y_min_final = y_min_calculated
                    
                    # Ensure y_max is also positive and greater than y_min for log scale
                    if y_max_calculated <= y_min_final:
                        y_max_final = y_min_final * 10 # Or some other sensible factor
                    else:
                        y_max_final = y_max_calculated
                else: # Linear scale
                    y_min_final = y_min_calculated
                    y_max_final = y_max_calculated
                
                self.ax.set_ylim(y_min_final, y_max_final)
            
            # Update labels and title
            self.ax.set_xlabel('Epoch')
            self.ax.set_ylabel(self.current_error_unit_label) # Use dynamic label
            self.ax.set_title('Training Progress')
            self.ax.legend()
            self.ax.grid(True, which="both", ls="-", alpha=0.2)
            
            # Add minor gridlines for log scale
            self.ax.grid(True, which="minor", ls=":", alpha=0.1)

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

        # Save the training plot for the current task
        try:
            task_id = self.task_list[self.current_task_index].get('task_id', f'task_{self.current_task_index + 1}')
            # Use 'task_dir' which is the specific directory for this task's artifacts
            save_dir = self.task_list[self.current_task_index].get('task_dir', self.job_manager.get_job_folder()) # Fallback to job folder if task_dir is missing
            
            # Ensure the save directory exists (it should, as it's created by TrainingSetupManager)
            os.makedirs(save_dir, exist_ok=True)

            # Create a new figure for saving
            fig = Figure(figsize=(8, 5), dpi=300)
            ax = fig.add_subplot(111)
            
            # Plot the data using actual epoch numbers
            ax.plot(self.epoch_points, self.train_loss_values, label='Training', color='blue', marker='.')
            ax.plot(self.epoch_points, self.valid_loss_values, label='Validation', color='red', marker='.')
            
            # Set y-axis to log scale
            ax.set_yscale('log')
            
            # Set labels and title
            ax.set_xlabel('Epoch')
            ax.set_ylabel(self.current_error_unit_label) # Use dynamic label for saved plot
            ax.set_title(f'Training History - Task {task_id}')
            ax.legend()
            ax.grid(True, which="both", ls="-", alpha=0.2)
            ax.grid(True, which="minor", ls=":", alpha=0.1)
            
            # Save the plot
            plot_file = os.path.join(save_dir, f'training_history_{task_id}.png')
            fig.savefig(plot_file, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Saved training history plot for task {task_id} at: {plot_file}")
        except Exception as e:
            print(f"Failed to save training history plot: {str(e)}")

        if self.isVisible():  # Check if the window still exists
            # Check if the training process was stopped early
            if getattr(self, 'training_process_stopped', False):
                self.status_label.setText("Training stopped early. Saving model to task folder...")
                self.status_label.setStyleSheet("color: #b22222; font-size: 14pt; font-weight: bold;")  # Reddish color
            else:
                # Only show completion message if this is the last task
                if self.current_task_index >= len(self.task_list) - 1:
                    self.status_label.setText("All Training Tasks Completed!")
                    self.status_label.setStyleSheet("color: green; font-size: 12pt; font-weight: bold;")
                else:
                    self.status_label.setText(f"Task {self.current_task_index + 1}/{len(self.task_list)} completed. Preparing next task...")
                    self.status_label.setStyleSheet("color: #004d99; font-size: 12pt; font-weight: bold;")

            # Only show the "Proceed to Testing" button if training is stopped or all tasks are completed
            if getattr(self, 'training_process_stopped', False) or self.current_task_index >= len(self.task_list) - 1:
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
            self.timer_running = True  # Re-enable timer for the next task
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
        self.auto_proceed_timer.start(60000)  # 60 seconds

    def transition_to_testing_gui(self):
        self.auto_proceed_timer.stop()
        training_results = self.training_task_manager.get_training_results()

        # Get the job folder from the job manager
        job_folder = self.job_manager.get_job_folder()
        hyperparams_path = os.path.join(job_folder, 'hyperparams.json')

        final_params = None
        try:
            with open(hyperparams_path, 'r') as f:
                final_params = json.load(f)
            self.logger.info("Successfully loaded definitive hyperparameters from the job folder.")
        except Exception as e:
            self.logger.error(f"Could not load hyperparams.json from {hyperparams_path} for testing transition: {e}")
            # Fallback to using parameters from the first task if loading fails
            final_params = self.task_list[0]['hyperparams']
            QMessageBox.warning(self, "Warning", f"Could not load definitive hyperparameters from the job folder. Testing may use stale data.\n\nError: {e}")

        # Ensure the singleton is populated with the definitive parameters
        self.training_setup_manager.hyper_param_manager.update_params(final_params)
        self.logger.info("Updated HyperParamManager singleton with final parameters before transitioning to testing.")

        # The VEstimTestingGUI will now correctly find the parameters in the singleton.
        self.testing_manager = VEstimTestingManager(job_manager=self.job_manager, params=final_params, task_list=self.task_list, training_results=training_results)
        self.testing_gui = VEstimTestingGUI(job_manager=self.job_manager, params=final_params, task_list=self.task_list, training_results=training_results)
        self.testing_gui.show()
        self.close()


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    task_list = []  # Replace with actual task list
    params = {}  # Replace with actual parameters
    sys.exit(app.exec_())
