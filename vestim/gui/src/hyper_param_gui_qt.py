import os
import json
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, 
    QLineEdit, QFileDialog, QMessageBox, QDialog, QComboBox
)
from PyQt5.QtCore import Qt, QPropertyAnimation, pyqtSignal
from PyQt5.QtGui import QIcon

from vestim.gateway.src.job_manager_qt import JobManager
from vestim.gateway.src.hyper_param_manager_qt import VEstimHyperParamManager
from vestim.gui.src.training_setup_gui_qt import VEstimTrainSetupGUI

# Initialize the JobManager
job_manager = JobManager()
import logging
class VEstimHyperParamGUI(QWidget):
    def __init__(self):
        self.logger = logging.getLogger(__name__)  # Initialize the logger within the instance
        self.logger.info("Initializing Hyperparameter GUI")
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
            self.logger.warning("Icon file not found. Make sure 'icon.ico' is in the correct directory.")

    def build_gui(self):
        layout = QVBoxLayout()

        # Title Label
        title_label = QLabel("Select Hyperparameters for Model Training")
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
        self.param_entries = {} # Resetting to ensure clean state for dynamic widgets
        
        # --- Model Type Selection ---
        model_type_label = QLabel("Model Type:")
        model_type_label.setStyleSheet("font-size: 12pt; font-weight: bold;")
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["LSTM", "FNN", "GRU"]) # Add more as they are supported
        self.model_type_combo.setFixedHeight(30)
        self.model_type_combo.setStyleSheet("font-size: 12pt;")
        self.model_type_combo.currentTextChanged.connect(self.on_model_type_changed)
        layout.addWidget(model_type_label, 0, 0)
        layout.addWidget(self.model_type_combo, 0, 1, 1, 3) # Span 3 columns for the combo box
        self.param_entries["MODEL_TYPE"] = self.model_type_combo # Store combo box itself

        # --- Training Method Selection ---
        training_method_label = QLabel("Training Method:")
        training_method_label.setStyleSheet("font-size: 12pt; font-weight: bold;")
        self.training_method_combo = QComboBox()
        self.training_method_combo.addItems(["SequenceRNN", "WholeSequenceFNN"])
        self.training_method_combo.setFixedHeight(30)
        self.training_method_combo.setStyleSheet("font-size: 12pt;")
        self.training_method_combo.currentTextChanged.connect(self.on_training_method_changed)
        layout.addWidget(training_method_label, 1, 0)
        layout.addWidget(self.training_method_combo, 1, 1, 1, 3)
        self.param_entries["TRAINING_METHOD"] = self.training_method_combo

        # --- Feature and Target Columns ---
        self.add_line_edit_param(layout, 2, "Feature Columns (Input):", "FEATURE_COLS", "e.g., SOC,Current,Temp", "Comma-separated list of input feature column names.")
        self.add_line_edit_param(layout, 3, "Target Column (Output):", "TARGET_COL", "e.g., Voltage", "Name of the single target column.")
 
        # --- Common Hyperparameters ---
        common_params_start_row = 4
        common_hyperparameters = [
            {"label": "Mini-batches:", "param": "BATCH_SIZE", "default": "32", "tooltip": "Number of samples per gradient update."},
            {"label": "Max Epochs:", "param": "MAX_EPOCHS", "default": "100", "tooltip": "Maximum number of training epochs."},
            {"label": "Initial LR:", "param": "INITIAL_LR", "default": "0.001", "tooltip": "Initial learning rate."},
            {"label": "LR Drop Factor:", "param": "LR_DROP_FACTOR", "default": "0.1", "tooltip": "Factor to reduce LR by."},
            {"label": "LR Drop Period:", "param": "LR_DROP_PERIOD", "default": "10", "tooltip": "Epochs before LR drop."},
            {"label": "Weight Decay:", "param": "WEIGHT_DECAY", "default": "0.0", "tooltip": "Weight decay (L2 penalty)."},
            {"label": "Validation Patience:", "param": "VALID_PATIENCE", "default": "10", "tooltip": "Epochs to wait for val_loss improvement before early stopping."},
            {"label": "Validation Freq:", "param": "ValidFrequency", "default": "1", "tooltip": "Frequency (in epochs) to perform validation."},
            {"label": "Repetitions:", "param": "REPETITIONS", "default": "1", "tooltip": "Number of times to repeat the training run."},
            {"label": "Num Workers:", "param": "NUM_WORKERS", "default": "4", "tooltip": "Number of workers for DataLoader."},
            {"label": "Train Split:", "param": "TRAIN_SPLIT", "default": "0.7", "tooltip": "Fraction of data for training (0.0 to 1.0)."},
            {"label": "Random Seed:", "param": "SEED", "default": "42", "tooltip": "Seed for reproducibility."}
        ]
        for idx, p_info in enumerate(common_hyperparameters):
            self.add_line_edit_param(layout, common_params_start_row + idx, p_info["label"], p_info["param"], p_info["default"], p_info["tooltip"])
        
        current_row = common_params_start_row + len(common_hyperparameters)

        # --- Model Specific Hyperparameters ---
        # These will be shown/hidden based on model_type_combo selection
        # LSTM / GRU specific
        self.lstm_gru_widgets = []
        self.lookback_entry = self._add_specific_param(layout, current_row, "Lookback:", "LOOKBACK", "50", "Window size for RNNs (LSTM/GRU).")
        self.lstm_gru_widgets.append(self.lookback_entry)
        self.param_entries["LOOKBACK"] = self.lookback_entry # Ensure it's in param_entries

        self.layers_entry = self._add_specific_param(layout, current_row + 1, "RNN Layers:", "LAYERS", "1", "Number of LSTM/GRU layers.") # Used by LSTM/GRU
        self.lstm_gru_widgets.append(self.layers_entry)
        self.param_entries["LAYERS"] = self.layers_entry # For LSTM/GRU

        self.hidden_units_entry = self._add_specific_param(layout, current_row + 2, "RNN Hidden Units:", "HIDDEN_UNITS", "64", "Number of hidden units in LSTM/GRU layers.")
        self.lstm_gru_widgets.append(self.hidden_units_entry)
        self.param_entries["HIDDEN_UNITS"] = self.hidden_units_entry # For LSTM/GRU

        self.rnn_dropout_entry = self._add_specific_param(layout, current_row + 3, "RNN Dropout Prob:", "DROPOUT_PROB", "0.0", "Dropout probability for LSTM/GRU layers.")
        self.lstm_gru_widgets.append(self.rnn_dropout_entry)
        self.param_entries["DROPOUT_PROB"] = self.rnn_dropout_entry # For LSTM/GRU
        
        self.concatenate_raw_data_entry = self._add_specific_param(layout, current_row + 4, "Concat Raw (RNN):", "CONCATENATE_RAW_DATA", "False", "For SequenceRNN: True to concat raw data before sequencing, False otherwise.")
        self.lstm_gru_widgets.append(self.concatenate_raw_data_entry)
        self.param_entries["CONCATENATE_RAW_DATA"] = self.concatenate_raw_data_entry


        # FNN specific
        self.fnn_widgets = []
        self.fnn_hidden_layers_entry = self._add_specific_param(layout, current_row, "FNN Hidden Layers:", "FNN_HIDDEN_LAYERS", "128,64", "Comma-sep list of FNN hidden layer sizes. Use ';' for multiple configs.")
        self.fnn_widgets.append(self.fnn_hidden_layers_entry)
        self.param_entries["FNN_HIDDEN_LAYERS"] = self.fnn_hidden_layers_entry
        
        self.fnn_dropout_entry = self._add_specific_param(layout, current_row + 1, "FNN Dropout Prob:", "FNN_DROPOUT_PROB", "0.0", "Dropout probability for FNN layers.")
        self.fnn_widgets.append(self.fnn_dropout_entry)
        self.param_entries["FNN_DROPOUT_PROB"] = self.fnn_dropout_entry

        # Initial UI state based on default model type
        self.on_model_type_changed(self.model_type_combo.currentText())
        self.on_training_method_changed(self.training_method_combo.currentText())


    def add_line_edit_param(self, grid_layout, row, label_text, param_name, default_value, tooltip_text):
        label = QLabel(label_text)
        label.setStyleSheet("font-size: 12pt; font-weight: bold;")
        label.setToolTip(tooltip_text)
        
        entry = QLineEdit(str(self.params.get(param_name, default_value)))
        entry.setFixedHeight(30)
        entry.setStyleSheet("font-size: 12pt;")
        
        grid_layout.addWidget(label, row, 0)
        grid_layout.addWidget(entry, row, 1, 1, 3) # Span 3 columns
        self.param_entries[param_name] = entry
        return entry

    def _add_specific_param(self, grid_layout, row, label_text, param_name, default_value, tooltip_text):
        """Helper to add a parameter that might be shown/hidden."""
        # This creates the widgets but doesn't add them to layout yet.
        # The on_model_type_changed will handle adding/removing from layout.
        label = QLabel(label_text)
        label.setStyleSheet("font-size: 12pt; font-weight: bold;")
        label.setToolTip(tooltip_text)
        
        entry = QLineEdit(str(self.params.get(param_name, default_value)))
        entry.setFixedHeight(30)
        entry.setStyleSheet("font-size: 12pt;")
        
        # Store them for easy access, actual adding to layout is conditional
        # For now, just return the entry. The calling function will manage visibility.
        return entry # We return the entry, label is implicitly handled by add_line_edit_param if we refactor to use it

    def on_model_type_changed(self, model_type):
        self.logger.info(f"Model type changed to: {model_type}")
        is_rnn = model_type in ["LSTM", "GRU"]
        is_fnn = model_type == "FNN"

        # Manage visibility of LSTM/GRU specific widgets
        # Assuming self.layers_entry, self.hidden_units_entry, self.rnn_dropout_entry are QLineEdit created by _add_specific_param
        # And their labels are also created there. We need a way to show/hide both.
        # For simplicity, let's assume _add_specific_param returns a tuple (label_widget, entry_widget)
        # Or, we manage them as pairs.
        
        # For now, a simpler approach: just enable/disable entry. Proper show/hide of label+entry is better.
        # This requires _add_specific_param to add them to layout and then we show/hide.
        # Let's adjust _add_specific_param and how it's called.

        # Simplified: just manage entry visibility for now.
        # A more robust solution would involve storing (label, entry) pairs.
        
        # LSTM/GRU specific fields
        self.layers_entry.setVisible(is_rnn)
        self.layers_entry.parent().findChild(QLabel, self.layers_entry.objectName().replace("_entry", "_label")).setVisible(is_rnn) if self.layers_entry.objectName() else None # Crude way to find label
        
        self.hidden_units_entry.setVisible(is_rnn)
        self.hidden_units_entry.parent().findChild(QLabel, self.hidden_units_entry.objectName().replace("_entry", "_label")).setVisible(is_rnn) if self.hidden_units_entry.objectName() else None

        self.rnn_dropout_entry.setVisible(is_rnn)
        self.rnn_dropout_entry.parent().findChild(QLabel, self.rnn_dropout_entry.objectName().replace("_entry", "_label")).setVisible(is_rnn) if self.rnn_dropout_entry.objectName() else None
        
        # FNN specific fields
        self.fnn_hidden_layers_entry.setVisible(is_fnn)
        self.fnn_hidden_layers_entry.parent().findChild(QLabel, self.fnn_hidden_layers_entry.objectName().replace("_entry", "_label")).setVisible(is_fnn) if self.fnn_hidden_layers_entry.objectName() else None
        
        self.fnn_dropout_entry.setVisible(is_fnn)
        self.fnn_dropout_entry.parent().findChild(QLabel, self.fnn_dropout_entry.objectName().replace("_entry", "_label")).setVisible(is_fnn) if self.fnn_dropout_entry.objectName() else None

        # Training method and lookback interaction
        self.on_training_method_changed(self.training_method_combo.currentText())


    def on_training_method_changed(self, training_method):
        self.logger.info(f"Training method changed to: {training_method}")
        is_sequence_rnn = training_method == "SequenceRNN"
        
        # Lookback is relevant for SequenceRNN
        self.lookback_entry.setEnabled(is_sequence_rnn)
        self.lookback_entry.setVisible(is_sequence_rnn) # Also hide if not relevant
        self.lookback_entry.parent().findChild(QLabel, self.lookback_entry.objectName().replace("_entry", "_label")).setVisible(is_sequence_rnn) if self.lookback_entry.objectName() else None


        # Concatenate_raw_data is relevant for SequenceRNN
        self.concatenate_raw_data_entry.setEnabled(is_sequence_rnn)
        self.concatenate_raw_data_entry.setVisible(is_sequence_rnn)
        self.concatenate_raw_data_entry.parent().findChild(QLabel, self.concatenate_raw_data_entry.objectName().replace("_entry", "_label")).setVisible(is_sequence_rnn) if self.concatenate_raw_data_entry.objectName() else None


        # If model is FNN, training method should ideally be WholeSequenceFNN
        current_model_type = self.model_type_combo.currentText()
        if current_model_type == "FNN" and training_method != "WholeSequenceFNN":
            self.logger.warning("FNN model type is selected, but Training Method is not WholeSequenceFNN. This might be unintended.")
            # Optionally, force training_method_combo to "WholeSequenceFNN" or show a warning.
        elif current_model_type != "FNN" and training_method == "WholeSequenceFNN":
            self.logger.warning("WholeSequenceFNN training method is selected, but model type is not FNN. This might be unintended.")
            # Optionally, force training_method_combo to "SequenceRNN" or show a warning.


    def proceed_to_training(self):
        try:
            # Validate parameters and save them if valid
            new_params = {}
            for param_name, widget in self.param_entries.items():
                if isinstance(widget, QLineEdit):
                    new_params[param_name] = widget.text()
                elif isinstance(widget, QComboBox):
                    new_params[param_name] = widget.currentText()
            
            # Ensure all expected params are present, even if from non-visible fields, using defaults if needed
            # This part needs to be robust, perhaps by iterating over a predefined list of all possible params
            # For now, this captures visible and combo box values.
            # VEstimHyperParamManager.update_params should handle defaults for missing keys if that's the design.
            self.logger.info(f"Proceeding to training with params: {new_params}")
            self.update_params(new_params)
            if self.job_manager.get_job_folder():
                self.hyper_param_manager.save_params()
            else:
                self.logger.error("Job folder is not set.")
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
                self.logger.info(f"Loading params from JSON file: {filepath}") 
                self.update_gui_with_loaded_params()
            except Exception as e:
                self.logger.error(f"Failed to load parameters from {filepath}: {e}")
                QMessageBox.critical(self, "Error", f"Failed to load parameters: {str(e)}")

    def update_params(self, new_params):
        try:
            self.logger.info(f"Updating parameters: {new_params}")
            self.hyper_param_manager.update_params(new_params)
        except ValueError as e:
            self.logger.error(f"Invalid parameter input: {new_params} - Error: {e}")
            QMessageBox.critical(self, "Error", f"Invalid parameter input: {str(e)}")
        self.update_gui_with_loaded_params()

    def update_gui_with_loaded_params(self):
        # This needs to handle QComboBox as well
        for param_name, widget in self.param_entries.items():
            if param_name in self.params:
                value_from_params = self.params[param_name]
                if isinstance(widget, QLineEdit):
                    # If the param value is a list (e.g. from old JSON format for HIDDEN_UNITS), join it.
                    # New params like FNN_HIDDEN_LAYERS might be stored as "128,64" string.
                    text_value = ""
                    if isinstance(value_from_params, list):
                        text_value = ','.join(map(str, value_from_params))
                    else:
                        text_value = str(value_from_params)
                    widget.setText(text_value)
                elif isinstance(widget, QComboBox):
                    index = widget.findText(str(value_from_params), Qt.MatchFixedString)
                    if index >= 0:
                        widget.setCurrentIndex(index)
                    else:
                        self.logger.warning(f"Value '{value_from_params}' for param '{param_name}' not found in QComboBox. Using default.")
                        widget.setCurrentIndex(0) # Default to first item if not found
            # else: # If param not in loaded self.params, QLineEdit already has default from add_line_edit_param
                  # For QComboBox, ensure a default is selected if not in params
                # if isinstance(widget, QComboBox):
                #    widget.setCurrentIndex(0) # Or load a default from a defaults map
        
        # Trigger the change handlers to set initial visibility correctly after loading params
        if "MODEL_TYPE" in self.param_entries:
            self.on_model_type_changed(self.model_type_combo.currentText())
        if "TRAINING_METHOD" in self.param_entries:
            self.on_training_method_changed(self.training_method_combo.currentText())

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
