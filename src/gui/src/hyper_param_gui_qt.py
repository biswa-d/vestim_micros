import os
import json
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, 
    QLineEdit, QFileDialog, QMessageBox, QDialog
)
from PyQt5.QtCore import Qt, QPropertyAnimation
from PyQt5.QtGui import QIcon

from src.gateway.src.job_manager_qt import JobManager
from src.gateway.src.hyper_param_manager_qt import VEstimHyperParamManager
from src.gui.src.training_setup_gui_qt import VEstimTrainSetupGUI

# Initialize the JobManager
job_manager = JobManager()

class VEstimHyperParamGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.params = {}  # Initialize an empty params dictionary
        self.job_manager = job_manager  # Use the shared JobManager instance
        self.hyper_param_manager = VEstimHyperParamManager()  # Initialize HyperParamManager
        self.param_entries = {}  # To store the entry widgets for parameters

        self.setup_window()
        self.build_gui()

    def setup_window(self):
        self.setWindowTitle("VEstim")
        self.setGeometry(100, 100, 900, 600)
        resources_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources')
        icon_path = os.path.join(resources_path, 'icon.ico')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        else:
            print("Icon file not found. Make sure 'icon.ico' is in the correct directory.")

    def build_gui(self):
        layout = QVBoxLayout()

        # Title Label
        title_label = QLabel("Select Hyperparameters for LSTM Model")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 16pt; font-weight: bold;")
        layout.addWidget(title_label)

        # Hyperparameter Guide Button, centered and smaller size
        guide_button = QPushButton("Open Hyperparameter Guide")
        guide_button.setFixedWidth(200)  # Half the length of the title
        guide_button.setFixedHeight(30)  # Smaller height
        guide_button.setStyleSheet("font-size: 10pt;")
        guide_button.clicked.connect(self.open_guide)

        # Create a centered layout for the guide button
        guide_button_layout = QHBoxLayout()
        guide_button_layout.addStretch(1)
        guide_button_layout.addWidget(guide_button)
        guide_button_layout.addStretch(1)
        guide_button_layout.setContentsMargins(0, 10, 0, 10)  # Add padding
        layout.addLayout(guide_button_layout)

        # Instructional text below the guide button
        instructions_label = QLabel("Please enter comma-separated values for multiple inputs.\n"
                                    "For detailed explanations of each hyperparameter, refer to the guide above.")
        instructions_label.setAlignment(Qt.AlignCenter)
        instructions_label.setStyleSheet("font-size: 10pt; color: gray;")
        layout.addWidget(instructions_label)

        # Hyperparameters grid layout with increased label and input sizes
        param_layout = QGridLayout()

        # Add parameter widgets
        self.add_param_widgets(param_layout)

        # Make the grid take more vertical space
        layout.addStretch(1)
        layout.addLayout(param_layout)
        layout.addStretch(1)

        # Buttons at the bottom in vertical arrangement and centered
        button_layout = QVBoxLayout()

        # Load Params button centered and styled
        load_button = QPushButton("Load Params from File")
        load_button.setFixedWidth(200)  # 1/4th of the window width
        load_button.setStyleSheet("font-size: 12pt;")
        load_button.clicked.connect(self.load_params_from_json)
        button_layout.addWidget(load_button, alignment=Qt.AlignCenter)
        
        # Create Training Tasks button centered and styled
        start_button = QPushButton("Create Training Tasks")
        start_button.setFixedWidth(200)  # 1/4th of the window width
        start_button.setStyleSheet("background-color: #0b6337; color: white; font-size: 12pt;")
        start_button.clicked.connect(self.proceed_to_training)
        button_layout.addWidget(start_button, alignment=Qt.AlignCenter)

        layout.addLayout(button_layout)

        self.setLayout(layout)


    def add_param_widgets(self, layout):
        # Increase size of labels and entry boxes
        hyperparameters = [
            {"label": "Layers:", "default": self.params.get("LAYERS", "1"), "tooltip": "Number of LSTM layers in the model", "param": "LAYERS"},
            {"label": "Hidden Units:", "default": self.params.get("HIDDEN_UNITS", "10"), "tooltip": "Number of units in each LSTM layer", "param": "HIDDEN_UNITS"},
            {"label": "Mini-batches:", "default": self.params.get("BATCH_SIZE", "100"), "tooltip": "Number of mini-batches to use during training", "param": "BATCH_SIZE"},
            {"label": "Max Epochs:", "default": self.params.get("MAX_EPOCHS", "5000"), "tooltip": "Maximum number of epochs to train the model", "param": "MAX_EPOCHS"},
            {"label": "Initial LR:", "default": self.params.get("INITIAL_LR", "0.00001"), "tooltip": "The starting learning rate for the optimizer", "param": "INITIAL_LR"},
            {"label": "LR Drop Factor:", "default": self.params.get("LR_DROP_FACTOR", "0.5"), "tooltip": "Factor by which the learning rate is reduced after Drop Period", "param": "LR_DROP_FACTOR"},
            {"label": "LR Drop Period:", "default": self.params.get("LR_DROP_PERIOD", "10"), "tooltip": "The number of epochs after which the learning rate drops", "param": "LR_DROP_PERIOD"},
            {"label": "Validation Patience:", "default": self.params.get("VALID_PATIENCE", "10"), "tooltip": "Number of epochs to wait for validation improvement before early stopping", "param": "VALID_PATIENCE"},
            {"label": "Validation Freq:", "default": self.params.get("ValidFrequency", "3"), "tooltip": "How often (in epochs) to perform validation", "param": "ValidFrequency"},
            {"label": "Lookback:", "default": self.params.get("LOOKBACK", "400"), "tooltip": "Number of previous time steps to consider for each timestep", "param": "LOOKBACK"},
            {"label": "Repetitions:", "default": self.params.get("REPETITIONS", "1"), "tooltip": "Number of times to repeat the entire training process with randomized initial parameters", "param": "REPETITIONS"},
        ]

        # Set bigger sizes for labels and entry boxes
        for idx, param in enumerate(hyperparameters):
            label_text = param["label"]
            default_value = param["default"]
            tooltip_text = param["tooltip"]
            param_name = param["param"]

            label = QLabel(label_text)
            label.setStyleSheet("font-size: 12pt; font-weight: bold;")  # Bigger label
            entry = QLineEdit(default_value)
            entry.setFixedHeight(30)  # Bigger entry box
            entry.setStyleSheet("font-size: 12pt;")  # Bigger text in the entry

            # Tooltip on hover
            label.setToolTip(tooltip_text)

            layout.addWidget(label, idx // 2, (idx % 2) * 2)
            layout.addWidget(entry, idx // 2, (idx % 2) * 2 + 1)

            self.param_entries[param_name] = entry


    def proceed_to_training(self):
        try:
            # Validate parameters and save them if valid
            new_params = {param: entry.text() for param, entry in self.param_entries.items()}
            self.update_params(new_params)
            if self.job_manager.get_job_folder():
                self.hyper_param_manager.save_params()
            else:
                raise ValueError("Job folder is not set.")

            self.close()  # Close the current window immediately
            self.training_setup_gui = VEstimTrainSetupGUI(new_params)
            self.training_setup_gui.show()

        except ValueError as e:
            QMessageBox.critical(self, "Error", f"Invalid parameter input: {str(e)}")

    def show_training_setup_gui(self):
        # Initialize the next GUI after fade-out is complete
        self.training_setup_gui = VEstimTrainSetupGUI(self.params)
        current_geometry = self.geometry()
        self.training_setup_gui.setGeometry(current_geometry)
        # self.training_setup_gui.setGeometry(100, 100, 900, 600)
        self.training_setup_gui.show()
        self.close()  # Close the previous window


    def load_params_from_json(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Params", "", "JSON Files (*.json);;All Files (*)")
        if filepath:
            try:
                # Load and validate parameters using the manager
                self.params = self.hyper_param_manager.load_params(filepath)
                self.update_gui_with_loaded_params()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load parameters: {str(e)}")

    def update_params(self, new_params):
        try:
            self.hyper_param_manager.update_params(new_params)
        except ValueError as e:
            QMessageBox.critical(self, "Error", f"Invalid parameter input: {str(e)}")
        self.update_gui_with_loaded_params()

    def update_gui_with_loaded_params(self):
        for param_name, entry in self.param_entries.items():
            if param_name in self.params:
                value = ', '.join(map(str, self.params[param_name])) if isinstance(self.params[param_name], list) else str(self.params[param_name])
                entry.setText(value)

    def open_guide(self):
        resources_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources')
        pdf_path = os.path.join(resources_path, 'hyper_param_guide.pdf')
        if os.path.exists(pdf_path):
            try:
                os.startfile(pdf_path)
            except Exception as e:
                print(f"Failed to open PDF: {e}")
        else:
            print("PDF guide not found. Make sure 'hyper_param_guide.pdf' is in the correct directory.")

if __name__ == "__main__":
    app = QApplication([])
    gui = VEstimHyperParamGUI()
    gui.show()
    app.exec_()
