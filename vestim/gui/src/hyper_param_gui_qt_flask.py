import os
import json
import requests
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, 
    QLineEdit, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon

from vestim.gui.src.training_setup_gui_qt_test import VEstimTrainSetupGUI
import logging

# Flask server URL where the Flask hyperparam manager is hosted
FLASK_SERVER_URL = "http://localhost:5001"

class VEstimHyperParamGUI(QWidget):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Hyperparameter GUI")
        super().__init__()
        self.params = {}  # Empty params dictionary
        self.param_entries = {}  # To store entry widgets

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
            self.logger.warning("Icon file not found. Make sure 'icon.ico' is in the correct directory.")

    def build_gui(self):
        layout = QVBoxLayout()

        # Title Label
        title_label = QLabel("Select Hyperparameters for LSTM Model")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 16pt; font-weight: bold;")
        layout.addWidget(title_label)

        # Hyperparameter Guide Button
        guide_button = QPushButton("Open Hyperparameter Guide")
        guide_button.setFixedWidth(200)
        guide_button.setFixedHeight(30)
        guide_button.setStyleSheet("font-size: 10pt;")
        guide_button.clicked.connect(self.open_guide)

        # Center the guide button
        guide_button_layout = QHBoxLayout()
        guide_button_layout.addStretch(1)
        guide_button_layout.addWidget(guide_button)
        guide_button_layout.addStretch(1)
        layout.addLayout(guide_button_layout)

        # Instructions Label
        instructions_label = QLabel("Please enter comma-separated values for multiple inputs.\n"
                                    "For detailed explanations of each hyperparameter, refer to the guide above.")
        instructions_label.setAlignment(Qt.AlignCenter)
        instructions_label.setStyleSheet("font-size: 10pt; color: gray;")
        layout.addWidget(instructions_label)

        # Hyperparameters Grid Layout
        param_layout = QGridLayout()
        self.add_param_widgets(param_layout)
        layout.addStretch(1)
        layout.addLayout(param_layout)
        layout.addStretch(1)

        # Buttons at the bottom
        button_layout = QVBoxLayout()

        # Load Params button
        load_button = QPushButton("Load Params from File")
        load_button.setFixedWidth(200)
        load_button.setStyleSheet("font-size: 12pt;")
        load_button.clicked.connect(self.load_params_from_json)
        button_layout.addWidget(load_button, alignment=Qt.AlignCenter)
        
        # Create Training Tasks button
        start_button = QPushButton("Create Training Tasks")
        start_button.setFixedWidth(200)
        start_button.setStyleSheet("background-color: #0b6337; color: white; font-size: 12pt;")
        start_button.clicked.connect(self.proceed_to_training)
        button_layout.addWidget(start_button, alignment=Qt.AlignCenter)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def add_param_widgets(self, layout):
        hyperparameters = [
            {"label": "Layers:", "default": "1", "tooltip": "Number of LSTM layers in the model", "param": "LAYERS"},
            {"label": "Hidden Units:", "default": "10", "tooltip": "Number of units in each LSTM layer", "param": "HIDDEN_UNITS"},
            {"label": "Mini-batches:", "default": "100", "tooltip": "Number of mini-batches to use during training", "param": "BATCH_SIZE"},
            {"label": "Max Epochs:", "default": "5000", "tooltip": "Maximum number of epochs to train the model", "param": "MAX_EPOCHS"},
            {"label": "Initial LR:", "default": "0.00001", "tooltip": "The starting learning rate for the optimizer", "param": "INITIAL_LR"},
            {"label": "LR Drop Factor:", "default": "0.5", "tooltip": "Factor by which the learning rate is reduced", "param": "LR_DROP_FACTOR"},
            {"label": "LR Drop Period:", "default": "10", "tooltip": "The number of epochs after which the learning rate drops", "param": "LR_DROP_PERIOD"},
            {"label": "Validation Patience:", "default": "10", "tooltip": "Number of epochs to wait for validation improvement", "param": "VALID_PATIENCE"},
            {"label": "Validation Freq:", "default": "3", "tooltip": "How often to perform validation", "param": "ValidFrequency"},
            {"label": "Lookback:", "default": "400", "tooltip": "Number of previous time steps to consider", "param": "LOOKBACK"},
            {"label": "Repetitions:", "default": "1", "tooltip": "Number of times to repeat the training", "param": "REPETITIONS"},
        ]

        for idx, param in enumerate(hyperparameters):
            label = QLabel(param["label"])
            label.setStyleSheet("font-size: 12pt; font-weight: bold;")
            entry = QLineEdit(param["default"])
            entry.setFixedHeight(30)
            entry.setStyleSheet("font-size: 12pt;")
            label.setToolTip(param["tooltip"])

            layout.addWidget(label, idx // 2, (idx % 2) * 2)
            layout.addWidget(entry, idx // 2, (idx % 2) * 2 + 1)

            self.param_entries[param["param"]] = entry

    def proceed_to_training(self):
        try:
            # Collect parameters from the GUI
            new_params = {param: entry.text() for param, entry in self.param_entries.items()}
            self.logger.info(f"Proceeding to training with params: {new_params}")

            # Make an API call to save the parameters
            response = requests.post(f"{FLASK_SERVER_URL}/update_params", json={"params": new_params})
            if response.status_code == 200:
                # Update tool state to move to the training setup screen
                tool_state = {
                    "current_state": "training_setup",
                    "current_screen": "VEstimTrainSetupGUI"
                }
                with open("vestim/tool_state.json", "w") as f:
                    json.dump(tool_state, f)

                # Open the training setup GUI
                self.logger.info("Parameters updated successfully. Proceeding to training.")
                self.close()
                self.training_setup_gui = VEstimTrainSetupGUI(new_params)
                self.training_setup_gui.show()
            else:
                raise ValueError(f"Error updating parameters: {response.json().get('error')}")


        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to proceed to training: {str(e)}")

    def load_params_from_json(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Params", "", "JSON Files (*.json);;All Files (*)")
        if filepath:
            try:
                # Load parameters from the selected JSON file
                self.logger.info(f"Loading params from JSON file: {filepath}")
                response = requests.post(f"{FLASK_SERVER_URL}/load_params", json={"filepath": filepath})
                if response.status_code == 200:
                    self.params = response.json()
                    self.update_gui_with_loaded_params()
                else:
                    raise ValueError(f"Error loading parameters: {response.json().get('error')}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load parameters: {str(e)}")

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
            self.logger.warning("PDF guide not found. Make sure 'hyper_param_guide.pdf' is in the correct directory.")

if __name__ == "__main__":
    app = QApplication([])
    gui = VEstimHyperParamGUI()
    gui.show()
    app.exec_()
