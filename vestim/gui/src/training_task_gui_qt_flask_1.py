import requests
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QFrame, QTextEdit, QHBoxLayout, QWidget
from PyQt5.QtCore import QTimer
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class VEstimTrainingTaskGUI(QMainWindow):
    def __init__(self, task_list, params):
        super().__init__()

        self.task_list = task_list
        self.params = params
        self.current_task_index = None  # This will be updated after fetching task status
        self.train_loss_values = []
        self.valid_loss_values = []
        self.valid_x_values = []
        self.start_time = None

        # Initialize empty GUI placeholders
        self.initUI()
        self.build_empty_frame()

        # Make an API call to fetch the task status
        self.poll_task_status()

    def initUI(self):
        self.setWindowTitle("VEstim - Training Task")
        self.setGeometry(100, 100, 900, 600)

    def build_empty_frame(self):
        """
        Initialize the GUI with placeholders for task information.
        These will be updated once the task status is received from the backend.
        """
        container = QWidget()
        self.setCentralWidget(container)

        self.main_layout = QVBoxLayout()

        # Title Label
        title_label = QLabel("Training LSTM Model")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 16pt; font-weight: bold;")
        self.main_layout.addWidget(title_label)

        # Hyperparameters (Empty placeholders)
        self.hyperparam_frame = QFrame(self)
        self.hyperparam_frame.setLayout(QVBoxLayout())
        self.main_layout.addWidget(self.hyperparam_frame)

        # Status Label (Empty)
        self.status_label = QLabel("Waiting for task status...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.status_label)

        # Time Label (Empty)
        self.time_value_label = QLabel("00h:00m:00s")
        self.time_value_label.setStyleSheet("color: purple; font-size: 11pt; font-weight: bold;")
        self.main_layout.addWidget(self.time_value_label)

        # Plot (Empty)
        self.setup_empty_plot()

        # Log Window (Empty)
        self.setup_empty_log_window()

        # Add layout to the central widget
        container.setLayout(self.main_layout)

    def setup_empty_plot(self):
        """
        Setup an empty plot with placeholders.
        """
        fig = Figure(figsize=(6, 2.5), dpi=100)
        self.ax = fig.add_subplot(111)
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss [% RMSE]")
        self.ax.set_title("Training and Validation Loss")
        self.train_line, = self.ax.plot([], [], label='Train Loss')
        self.valid_line, = self.ax.plot([], [], label='Validation Loss')
        self.ax.legend()

        self.canvas = FigureCanvas(fig)
        self.canvas.setMinimumSize(600, 300)
        self.main_layout.addWidget(self.canvas)

    def setup_empty_log_window(self):
        """
        Setup an empty log window.
        """
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
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
        """
        Poll the Flask backend for the current task's status and update the GUI.
        """
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
            learning_rate = progress_data.get('learning_rate', 0)
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

    def display_hyperparameters(self, hyperparams):
        """
        Display the hyperparameters in the GUI once they are fetched from the backend.
        """
        layout = self.hyperparam_frame.layout()
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        # Display hyperparameters
        for param, value in hyperparams.items():
            param_label = QLabel(f"{param}: {value}")
            layout.addWidget(param_label)

    def update_plot(self):
        """
        Update the plot with the latest training and validation loss values.
        """
        self.train_line.set_data(self.valid_x_values, self.train_loss_values)
        self.valid_line.set_data(self.valid_x_values, self.valid_loss_values)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw_idle()

if __name__ == "__main__":
    app = QApplication([])
    task_list = []  # Empty task list initially (will be updated after API call)
    params = {}
    gui = VEstimTrainingTaskGUI(task_list, params)
    gui.show()
    sys.exit(app.exec_())
