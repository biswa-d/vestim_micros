import requests
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QFrame, QTextEdit, QHBoxLayout, QWidget
from PyQt5.QtCore import QTimer
import time
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import logging

class VEstimTrainingTaskGUI(QMainWindow):
    def __init__(self, task_list, params):
        super().__init__()

        # Initialize attributes
        self.task_list = task_list
        self.params = params
        self.task_id = None  # Store the task ID returned from the Flask server
        self.timer = QTimer(self)  # Polling timer for task status updates
        self.logger = logging.getLogger(__name__)

        # Variables for task progress tracking
        self.train_loss_values = []
        self.valid_loss_values = []
        self.valid_x_values = []
        self.start_time = None
        self.timer_running = True
        self.training_process_stopped = False
        self.current_task_index = 0

        # Initialize UI and first task display
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
        self.build_gui(self.task_list[0])  # Initialize with task_list[0]

        # Poll for task updates every 5 seconds (or adjust as needed)
        self.timer.timeout.connect(self.poll_task_status)
        self.timer.start(5000)

    def initUI(self):
        self.setWindowTitle(f"VEstim - Training Task {self.current_task_index + 1}")
        self.setGeometry(100, 100, 900, 600)

    def build_gui(self, task=None):
        """Build the initial GUI with empty placeholders or the first task information."""
        container = QWidget()
        self.setCentralWidget(container)
        self.main_layout = QVBoxLayout()

        # Title label
        title_label = QLabel("Training LSTM Model with Hyperparameters")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 16pt; font-weight: bold;")
        self.main_layout.addWidget(title_label)

        # Display hyperparameters (with empty placeholders initially)
        self.hyperparam_frame = QFrame(self)
        self.hyperparam_frame.setLayout(QVBoxLayout())
        self.main_layout.addWidget(self.hyperparam_frame)

        # Initialize the hyperparameters with placeholders if no task is provided
        hyperparams = task['hyperparams'] if task else {key: "N/A" for key in self.param_labels.keys()}
        self.display_hyperparameters(hyperparams)

        # Status label (empty placeholder initially)
        self.status_label = QLabel("Waiting for task status...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.status_label)

        # Time and plot setup (with placeholders initially)
        self.setup_time_and_plot(task)

        # Log window setup (empty initially)
        self.setup_log_window(task)

        # Stop button (enabled, but doesn't perform any function yet)
        self.stop_button = QPushButton("Stop Training")
        self.stop_button.setStyleSheet("background-color: red; color: white; font-size: 12pt; font-weight: bold;")
        self.stop_button.setFixedWidth(150)
        self.stop_button.clicked.connect(self.stop_training)
        stop_button_layout = QHBoxLayout()
        stop_button_layout.addStretch(1)
        stop_button_layout.addWidget(self.stop_button)
        stop_button_layout.addStretch(1)
        self.main_layout.addLayout(stop_button_layout)

        # Proceed button (initially hidden)
        self.proceed_button = QPushButton("Proceed to Testing")
        self.proceed_button.setStyleSheet("""
            background-color: #0b6337; 
            color: white; 
            font-size: 12pt; 
            font-weight: bold; 
            padding: 10px 20px;
        """)
        self.proceed_button.setVisible(False)
        self.main_layout.addWidget(self.proceed_button)

        # Attach layout to container
        container.setLayout(self.main_layout)

    def display_hyperparameters(self, params):
        """Display the task hyperparameters, with placeholders if no params are provided."""
        layout = self.hyperparam_frame.layout()
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        hyperparam_layout = QVBoxLayout()
        for param, value in params.items():
            param_label = QLabel(f"{self.param_labels.get(param, param)}: {value}")
            param_label.setStyleSheet("font-size: 10pt; font-weight: bold;")
            hyperparam_layout.addWidget(param_label)

        layout.addLayout(hyperparam_layout)

    def setup_time_and_plot(self, task=None):
        """Setup the plot for training and validation loss, with placeholders if task is None."""
        time_layout = QHBoxLayout()

        # Time tracking label (initially 00h:00m:00s until updated)
        self.time_value_label = QLabel("00h:00m:00s" if task is None else task.get('elapsed_time', "00h:00m:00s"))
        self.time_value_label.setStyleSheet("color: purple; font-size: 11pt; font-weight: bold;")
        time_layout.addStretch(1)
        time_layout.addWidget(self.time_value_label)
        time_layout.addStretch(1)
        self.main_layout.addLayout(time_layout)

        # Plot setup (empty initially)
        max_epochs = int(task['hyperparams']['MAX_EPOCHS']) if task else 100
        fig = Figure(figsize=(6, 2.5), dpi=100)
        self.ax = fig.add_subplot(111)
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss [% RMSE]")
        self.ax.set_xlim(1, max_epochs)
        self.ax.set_title("Training and Validation Loss", fontsize=12)

        # Plot placeholders (empty until updated)
        self.train_line, = self.ax.plot([], [], label='Train Loss')
        self.valid_line, = self.ax.plot([], [], label='Validation Loss')
        self.ax.legend()
        self.canvas = FigureCanvas(fig)
        self.main_layout.addWidget(self.canvas)

    def setup_log_window(self, task=None):
        """Setup the log window, initially empty."""
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
        self.main_layout.addWidget(self.log_text)

    def poll_task_status(self):
        """Poll the backend for the task status and update the GUI accordingly."""
        try:
            response = requests.get("http://localhost:5000/task_status")
            if response.status_code == 200:
                task_status = response.json()
                self.update_gui_with_progress(task_status)
            else:
                self.status_label.setText("Error: Unable to fetch task status.")
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")

    def update_gui_with_progress(self, task_status):
        """
        Update the GUI based on the current task status and progress data.
        The response from the Flask API contains the current task index and progress data.
        """
        # Extract task progress data from the response
        progress_data = task_status.get('progress_data', None)
        current_task_index = task_status.get('current_task_index', 0)
        current_task = self.task_list[current_task_index]

        # Display task-specific information in the GUI (like hyperparameters)
        self.display_hyperparameters(current_task['hyperparams'])

        # Update the GUI's status label with the current task and status
        task_info = f"Task {current_task_index + 1}/{len(self.task_list)}"
        status = task_status.get('status', 'running')
        self.status_label.setText(f"{task_info} - {status}")

        # Handle log updates based on progress data
        if progress_data:
            epoch = progress_data.get('epoch', 0)
            train_loss = progress_data.get('train_loss', 0)
            val_loss = progress_data.get('val_loss', 0)
            delta_t_epoch = progress_data.get('delta_t_epoch', '0s')
            learning_rate = progress_data.get('learning_rate', None)
            best_val_loss = progress_data.get('best_val_loss', 0)

            # Format and append log messages
            log_message = (
                f"Epoch: {epoch}, "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Best Val Loss: {best_val_loss:.4f}, "
                f"Time Per Epoch: {delta_t_epoch}, "
                f"LR: {learning_rate:.1e}<br>"
            )
            self.log_text.append(log_message)

            # Update the plot
            self.train_loss_values.append(train_loss)
            self.valid_loss_values.append(val_loss)
            self.valid_x_values.append(epoch)
            self.update_plot()

            # Update elapsed time
            elapsed_time = progress_data.get('elapsed_time', "00h:00m:00s")
            self.time_value_label.setText(elapsed_time)

    def update_plot(self):
        """Update the plot with the latest training and validation loss values."""
        self.train_line.set_data(self.valid_x_values, self.train_loss_values)
        self.valid_line.set_data(self.valid_x_values, self.valid_loss_values)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw_idle()

    def stop_training(self):
        """Send a request to stop the training process."""
        response = requests.post(f"http://localhost:5000/stop_task/{self.current_task_index}")
        if response.status_code == 200:
            self.status_label.setText("Stopping training...")
        else:
            self.status_label.setText("Error stopping training")

    def handle_task_completed(self):
        """Handle task completion and show the 'Proceed to Testing' button."""
        self.status_label.setText("Training completed!")
        self.stop_button.hide()
        self.proceed_button.show()

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    task_list = []  # Replace with actual task list
    params = {}  # Replace with actual parameters
    gui = VEstimTrainingTaskGUI(task_list, params)
    gui.show()
    sys.exit(app.exec_())
