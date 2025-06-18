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

# Thread for fetching initial job data asynchronously
class JobDataThread(QThread):
    data_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, api_gateway, job_id, parent=None):
        super().__init__(parent)
        self.api_gateway = api_gateway
        self.job_id = job_id
        self.logger = logging.getLogger(__name__)

    def run(self):
        try:
            self.logger.info(f"JobDataThread: Fetching job info for {self.job_id}")
            job_info = self.api_gateway.get(f"jobs/{self.job_id}")
            if job_info:
                self.logger.info(f"JobDataThread: Successfully fetched job info for {self.job_id}")
                self.data_ready.emit(job_info)
            else:
                self.logger.warning(f"JobDataThread: No job info returned for {self.job_id}")
                self.error_occurred.emit(f"Job {self.job_id} not found or no data returned.")
        except Exception as e:
            self.logger.error(f"JobDataThread: Error fetching job info for {self.job_id}: {e}", exc_info=True)
            self.error_occurred.emit(str(e))

# Thread for fetching status updates asynchronously
class StatusUpdateThread(QThread):
    status_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, api_gateway, job_id, parent=None):
        super().__init__(parent)
        self.api_gateway = api_gateway
        self.job_id = job_id
        self.logger = logging.getLogger(__name__)

    def run(self):
        try:
            status_data = self.api_gateway.get(f"jobs/{self.job_id}")
            if status_data:
                self.status_ready.emit(status_data)
            else:
                self.logger.warning(f"StatusUpdateThread: No status data returned for job {self.job_id}")
                self.error_occurred.emit(f"No status data returned for job {self.job_id}.")
        except Exception as e:
            self.logger.error(f"StatusUpdateThread: Error fetching status for job {self.job_id}: {e}", exc_info=True)
            self.error_occurred.emit(str(e))

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
            response = self.api_gateway.post(f"jobs/{self.job_id}/tasks/{self.task_id}/start_training")
            if response.get("status") != "success":
                self.error_signal.emit(f"Failed to start training task: {response.get('message', 'Unknown error')}")
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
        
        # Initialize attributes
        self.job_info = {}
        self.hyperparams = {}
        self.training_tasks = []
        self.train_loss_values = []
        self.valid_loss_values = []
        self.epoch_points = []
        self.current_task_index = 0
        self.training_thread = None
        self.status_update_thread = None
        self.job_data_thread = None
        self.poll_timer = QTimer(self)
        
        # Initialize UI first
        self.initUI()
        
        # Show loading state
        self.status_label.setText(f"Loading job {self.job_id} data...")
        
        # Fetch initial job data asynchronously
        self.job_data_thread = JobDataThread(self.api, self.job_id, self)
        self.job_data_thread.data_ready.connect(self._handle_job_data_ready)
        self.job_data_thread.error_occurred.connect(self._handle_job_data_error)
        self.job_data_thread.finished.connect(self.job_data_thread.deleteLater)
        self.job_data_thread.start()

    def _handle_job_data_ready(self, job_info):
        """Handle successful job data retrieval"""
        self.logger.info(f"Successfully retrieved job data for {self.job_id}")
        self.job_info = job_info
        self.hyperparams = job_info.get("details", {}).get("hyperparameters", {})
        self.training_tasks = job_info.get("details", {}).get("training_tasks", [])
        
        self.logger.info(f"Retrieved hyperparameters: {self.hyperparams}")
        
        # Update hyperparameter display
        self.display_hyperparameters(self.hyperparams)
        
        if not self.training_tasks:
            self.logger.warning(f"No training tasks found for job {self.job_id}")
            self.status_label.setText(f"Job {self.job_id}: No training tasks found.")
            self.stop_button.setEnabled(False)
        else:
            self.logger.info(f"Found {len(self.training_tasks)} training tasks")
            self.status_label.setText(f"Job {self.job_id} data loaded. Starting training...")
            self.start_training()
            self.start_polling()

    def _handle_job_data_error(self, error_message):
        """Handle job data retrieval error"""
        self.logger.error(f"Error retrieving job {self.job_id} information: {error_message}")
        self.status_label.setText(f"Error loading job data: {error_message}")
        self.stop_button.setEnabled(False)

    def start_training(self):
        """Start training for the current task"""
        if not self.training_tasks:
            QMessageBox.warning(self, "Warning", "No training tasks available to start.")
            return
            
        if self.current_task_index >= len(self.training_tasks):
            QMessageBox.warning(self, "Warning", "No more tasks to process.")
            return
            
        current_task = self.training_tasks[self.current_task_index]
        task_id = current_task.get('task_id', f'task_{self.current_task_index}')
        
        self.logger.info(f"Starting training for task {task_id}")
        self.status_label.setText(f"Starting task {task_id}...")
        
        # Update UI state
        self.stop_button.setEnabled(True)
        
        # Create and start training thread
        self.training_thread = TrainingThread(self.job_id, task_id, self.api)
        self.training_thread.update_signal.connect(self._handle_training_update)
        self.training_thread.finished_signal.connect(self._handle_training_finished)
        self.training_thread.error_signal.connect(self._handle_training_error)
        self.training_thread.finished.connect(self.training_thread.deleteLater)
        self.training_thread.start()

    def _handle_training_update(self, update_data):
        """Handle training progress updates"""
        self.logger.debug(f"Training update received: {update_data}")
        
        # Update status if available
        if 'status' in update_data:
            self.status_label.setText(f"Training: {update_data['status']}")
        
        # Update plot and logs if history is available
        if 'history' in update_data and update_data['history']:
            self.update_plot(update_data['history'])
            self.update_log(update_data['history'])

    def _handle_training_finished(self, final_data):
        """Handle training completion"""
        self.logger.info(f"Training finished for task {self.current_task_index}")
        self.current_task_index += 1

        if self.current_task_index < len(self.training_tasks):
            self.logger.info(f"Starting next training task: {self.current_task_index}")
            self.status_label.setText(f"Task {self.current_task_index} complete. Starting next task...")
            self.log_text.append("\n" + "="*20 + f" Starting Task {self.current_task_index + 1} " + "="*20 + "\n")
            self.start_training()
        else:
            self.logger.info("All training tasks finished.")
            self.status_label.setText("Training completed successfully!")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
            
            self.stop_button.setEnabled(False)
            
            if self.job_info.get("details", {}).get("testing_config"):
                self.proceed_button.show()

    def _handle_training_error(self, error_message):
        """Handle training errors"""
        self.logger.error(f"Training error: {error_message}")
        self.status_label.setText(f"Training error: {error_message}")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        
        # Update UI state
        self.stop_button.setEnabled(False)
        
        QMessageBox.critical(self, "Training Error", f"Training failed: {error_message}")

    def initUI(self):
        self.setWindowTitle(f"VEstim - Training Task {self.current_task_index + 1}")
        self.setGeometry(100, 100, 900, 600)
        self.build_gui()

    def build_gui(self):
        """Build the GUI components after job data is loaded"""
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
        self.hyperparam_frame.setLayout(QGridLayout())
        self.main_layout.addWidget(self.hyperparam_frame)
        
        # Status Label
        self.status_label = QLabel("Loading job data...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.status_label)

        # Time Frame and Plot Setup
        self.setup_time_and_plot()

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
        self.log_text.setMinimumHeight(150)
        
        # Add log widget to layout
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        self.main_layout.addWidget(log_group)

        # Control buttons
        control_buttons = QHBoxLayout()
        
        # Stop button
        self.stop_button = QPushButton("Stop Training")
        self.stop_button.setStyleSheet("background-color: red; color: white; font-size: 12pt; font-weight: bold;")
        self.stop_button.setFixedWidth(150)
        self.stop_button.clicked.connect(self.stop_training)
        self.stop_button.setEnabled(False)

        control_buttons.addStretch(1)
        control_buttons.addWidget(self.stop_button)
        control_buttons.addStretch(1)
        self.main_layout.addLayout(control_buttons)

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
        layout = self.hyperparam_frame.layout()
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        if not task_params:
            layout.addWidget(QLabel("No hyperparameters available."), 0, 0)
            return

        display_items_ordered = []
        processed_keys = set()

        param_labels = {
            "LAYERS": "Layers", "HIDDEN_UNITS": "Hidden Units", "BATCH_SIZE": "Batch Size",
            "MAX_EPOCHS": "Max Epochs", "INITIAL_LR": "Initial LR", "LR_DROP_FACTOR": "LR Drop Factor",
            "LR_DROP_PERIOD": "LR Drop Period", "PLATEAU_PATIENCE": "Plateau Patience",
            "PLATEAU_FACTOR": "Plateau Factor", "VALID_PATIENCE": "Validation Patience",
            "ValidFrequency": "Validation Freq", "LOOKBACK": "Lookback", "REPETITIONS": "Repetitions",
            "NUM_LEARNABLE_PARAMS": "# Params", "INPUT_SIZE": "Input Size", "OUTPUT_SIZE": "Output Size",
            "SCHEDULER_TYPE": "LR Scheduler", "TRAINING_METHOD": "Training Method",
            "DEVICE_SELECTION": "Device", "MAX_TRAINING_TIME_SECONDS": "Max Train Time (Task)"
        }

        model_arch_keys = ['MODEL_TYPE', 'LAYERS', 'HIDDEN_UNITS', 'INPUT_SIZE', 'OUTPUT_SIZE', 'NUM_LEARNABLE_PARAMS']
        train_method_keys = ['TRAINING_METHOD', 'LOOKBACK', 'BATCH_TRAINING', 'BATCH_SIZE']
        train_control_keys = ['MAX_EPOCHS', 'INITIAL_LR', 'SCHEDULER_TYPE', 'VALID_PATIENCE', 'ValidFrequency', 'REPETITIONS']
        exec_env_keys = ['DEVICE_SELECTION', 'MAX_TRAINING_TIME_SECONDS']
        preferred_order = model_arch_keys + train_method_keys + train_control_keys + exec_env_keys

        def get_scheduler_display_val(params):
            scheduler_type = params.get('SCHEDULER_TYPE')
            display_val = scheduler_type if scheduler_type else "N/A"
            if scheduler_type == 'StepLR':
                period = params.get('LR_DROP_PERIOD', params.get('LR_PERIOD', 'N/A'))
                factor = params.get('LR_DROP_FACTOR', params.get('LR_PARAM', 'N/A'))
                display_val = f"StepLR (Period: {period}, Factor: {factor})"
            elif scheduler_type == 'ReduceLROnPlateau':
                patience = params.get('PLATEAU_PATIENCE', 'N/A')
                factor = params.get('PLATEAU_FACTOR', params.get('LR_PARAM', 'N/A'))
                display_val = f"ReduceLROnPlateau (Patience: {patience}, Factor: {factor})"
            return display_val

        for key in preferred_order:
            if key in task_params:
                label_text = param_labels.get(key, key.replace("_", " ").title())
                
                if key == 'MAX_TRAINING_TIME_SECONDS':
                    max_time_sec = task_params.get(key, 0)
                    try: max_time_sec = int(max_time_sec)
                    except (ValueError, TypeError): max_time_sec = 0
                    h, m, s = max_time_sec // 3600, (max_time_sec % 3600) // 60, max_time_sec % 60
                    value = f"{h:02d}H:{m:02d}M:{s:02d}S"
                elif key == 'SCHEDULER_TYPE':
                    value = get_scheduler_display_val(task_params)
                else:
                    value = task_params.get(key)

                display_items_ordered.append((label_text, str(value)))
                processed_keys.add(key)
        
        scheduler_sub_params = {'LR_DROP_PERIOD', 'LR_PERIOD', 'LR_PARAM', 'LR_DROP_FACTOR', 'PLATEAU_PATIENCE', 'PLATEAU_FACTOR'}
        for key, value in task_params.items():
            if key not in processed_keys and key not in scheduler_sub_params:
                label_text = param_labels.get(key, key.replace("_", " ").title())
                display_items_ordered.append((label_text, str(value)))

        items_per_row_display = 5
        for idx, (label, val) in enumerate(display_items_ordered):
            row = idx // items_per_row_display
            col_label = (idx % items_per_row_display) * 2
            col_value = col_label + 1
            
            param_label_widget = QLabel(f"{label}:")
            value_label_widget = QLabel(str(val))
            param_label_widget.setStyleSheet("font-size: 10pt;")
            value_label_widget.setStyleSheet("font-size: 10pt; font-weight: bold;")
            layout.addWidget(param_label_widget, row, col_label)
            layout.addWidget(value_label_widget, row, col_value)

    def setup_time_and_plot(self):
        # Time Layout
        time_layout = QHBoxLayout()
        self.static_text_label = QLabel("Current Task Time:")
        self.static_text_label.setStyleSheet("color: blue; font-size: 10pt;")
        self.time_value_label = QLabel("00h:00m:00s")
        self.time_value_label.setStyleSheet("color: purple; font-size: 11pt; font-weight: bold;")
        time_layout.addStretch(1)
        time_layout.addWidget(self.static_text_label)
        time_layout.addWidget(self.time_value_label)
        time_layout.addStretch(1)
        self.main_layout.addLayout(time_layout)

        # Plot Setup
        fig = Figure(figsize=(6, 2.5), dpi=100)
        self.ax = fig.add_subplot(111)
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.train_line, = self.ax.plot([], [], label='Train Loss')
        self.valid_line, = self.ax.plot([], [], label='Validation Loss')
        self.ax.legend()
        self.canvas = FigureCanvas(fig)
        self.main_layout.addWidget(self.canvas)
        fig.subplots_adjust(bottom=0.2)

    def start_polling(self):
        """Start polling for job status updates"""
        self.poll_timer.timeout.connect(self.request_status_update)
        self.poll_timer.start(2000)  # Poll every 2 seconds        self.request_status_update()  # Initial immediate update
        
    def request_status_update(self):
        """Request status update using background thread"""
        if hasattr(self, 'status_update_thread') and self.status_update_thread and self.status_update_thread.isRunning():
            return
        self.status_update_thread = StatusUpdateThread(self.api, self.job_id, self)
        self.status_update_thread.status_ready.connect(self._handle_status_ready)
        self.status_update_thread.error_occurred.connect(self._handle_status_update_error)
        self.status_update_thread.finished.connect(self.status_update_thread.deleteLater)
        self.status_update_thread.start()

    def _handle_status_ready(self, status_data):
        """Handle status update from background thread"""
        try:
            self.logger.debug(f"Job status data received: {status_data}")
            
            if not status_data:
                self.status_label.setText("Could not fetch job status.")
                return

            status = status_data.get('status')
            details = status_data.get('details', {})
            
            self.logger.debug(f"Job status: {status}")
            
            # Update hyperparameters if they've changed
            current_hyperparams = details.get('hyperparameters', {})
            if current_hyperparams and current_hyperparams != self.hyperparams:
                self.hyperparams = current_hyperparams
                self.display_hyperparameters(self.hyperparams)
                self.logger.debug(f"Updated hyperparameters: {self.hyperparams}")
            
            # Update training tasks if they've changed
            current_training_tasks = details.get('training_tasks', [])
            if current_training_tasks and current_training_tasks != self.training_tasks:
                self.training_tasks = current_training_tasks
                self.logger.debug(f"Updated training tasks: {self.training_tasks}")
            
            # Update status label
            status_message = details.get('message', '')
            self.status_label.setText(f"Status: {status} - {status_message}")

            # Update plot and logs if history is available
            if 'history' in details and details['history']:
                self.logger.debug(f"History data available: {len(details['history'])} entries")
                self.update_plot(details['history'])
                self.update_log(details['history'])

            # Check for terminal states
            terminal_statuses = ['complete', 'completed', 'error', 'failed', 'stopped']
            if status in terminal_statuses:
                if self.poll_timer.isActive():
                    self.poll_timer.stop()
                    self.logger.info(f"Polling stopped for job {self.job_id} due to status: {status}")
                
                self.stop_button.setEnabled(False)
                self.stop_button.setText(f"Job {status.capitalize()}")
                
                if status in ['complete', 'completed']:
                    # Check if testing is configured for this job before showing the button
                    if self.job_info.get("details", {}).get("testing_config"):
                        self.proceed_button.show()

        except Exception as e:
            self.logger.error(f"Error processing status update for job {self.job_id}: {e}", exc_info=True)
            self.status_label.setText(f"Error processing status: {e}")

    def _handle_status_update_error(self, error_message):
        """Handle status update error"""
        self.logger.error(f"Error updating status for job {self.job_id}: {error_message}")
        self.status_label.setText(f"Error fetching status: {error_message}")

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
        
        # Stop status update thread
        if hasattr(self, 'status_update_thread') and self.status_update_thread is not None:
            try:
                if self.status_update_thread.isRunning():
                    self.status_update_thread.quit()
                    self.status_update_thread.wait(2000)  # Wait up to 2 seconds
            except RuntimeError:
                pass
                
        # Stop job data thread  
        if hasattr(self, 'job_data_thread') and self.job_data_thread is not None:
            try:
                if self.job_data_thread.isRunning():
                    self.job_data_thread.quit()
                    self.job_data_thread.wait(2000)  # Wait up to 2 seconds
            except RuntimeError:
                pass
            
        super().closeEvent(event)

    def transition_to_testing_gui(self):
        """Transition to the testing GUI after training is complete."""
        try:
            testing_gui = VEstimTestingGUI(api_gateway=self.api, job_id=self.job_id)
            testing_gui.show()
            self.close()
        except Exception as e:
            self.logger.error(f"Failed to open testing GUI: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Could not open the testing GUI: {e}")
