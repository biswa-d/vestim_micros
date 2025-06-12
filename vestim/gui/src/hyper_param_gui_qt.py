# ---------------------------------------------------------------------------------
# Author: Biswanath Dehury
# Date: `{{date:YYYY-MM-DD}}`
# Version: 1.0.0
# Description: Description of the script
# ---------------------------------------------------------------------------------


import os
import json
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton,
    QLineEdit, QFileDialog, QMessageBox, QDialog, QGroupBox, QComboBox, QListWidget, QAbstractItemView,QFormLayout, QCheckBox, QTimeEdit
)
from PyQt5.QtCore import Qt, QPropertyAnimation, QTime
from PyQt5.QtGui import QIcon

import pandas as pd
import torch

from vestim.gui.src.api_gateway import APIGateway
from vestim.backend.src.managers.hyper_param_manager_qt import VEstimHyperParamManager
from vestim.gui.src.training_setup_gui_qt import VEstimTrainSetupGUI

import logging
class VEstimHyperParamGUI(QWidget):
    def __init__(self, api_gateway: APIGateway, job_folder: str):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        if not job_folder or not os.path.isdir(job_folder):
            raise ValueError("A valid job_folder must be provided.")
            
        self.job_folder = job_folder
        self.job_id = os.path.basename(job_folder)
        self.logger.info(f"Initializing Hyperparameter GUI for job_id: {self.job_id}")

        self.api_gateway = api_gateway
        self.params = {}
        self.param_entries = {}

        self.setup_window()
        self.build_gui()

    def setup_window(self):
        """Initial setup for the main window appearance."""
        self.setWindowTitle("VEstim - Hyperparameter Selection")
        self.setGeometry(100, 100, 950, 650)  # Adjusted size for better visibility

        # Load the application icon
        resources_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources')
        icon_path = os.path.join(resources_path, 'icon.ico')
        
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        else:
            self.logger.warning("Icon file not found. Make sure 'icon.ico' is in the correct directory.")
        
        self.setStyleSheet("QToolTip { font-weight: normal; font-size: 10pt; }")

    def build_gui(self):
        """Build the main UI layout with categorized sections for parameters."""
        main_layout = QVBoxLayout()
        # Set padding/margins around the main layout
        main_layout.setContentsMargins(20, 20, 20, 20)  # (left, top, right, bottom)

        # **📌 Title & Guide Section (Full Width)**
        title_label = QLabel("Select Hyperparameters for Model Training")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 16pt; font-weight: bold; margin-bottom: 10px;")
        main_layout.addWidget(title_label)

        guide_button = QPushButton("Open Hyperparameter Guide")
        guide_button.setFixedWidth(220)
        guide_button.setFixedHeight(30)
        guide_button.setStyleSheet("font-size: 10pt;")
        guide_button.clicked.connect(self.open_guide)

        guide_button_layout = QHBoxLayout()
        guide_button_layout.addStretch(1)
        guide_button_layout.addWidget(guide_button)
        guide_button_layout.addStretch(1)
        main_layout.addLayout(guide_button_layout)

        instructions_label = QLabel(
            "Please enter values for model parameters. Use comma-separated values for multiple inputs.\n"
            "Refer to the guide above for more details."
        )
        instructions_label.setAlignment(Qt.AlignCenter)
        instructions_label.setStyleSheet("font-size: 10pt; color: gray; margin-bottom: 10px;")
        main_layout.addWidget(instructions_label)

        # **📌 Hyperparameter Selection Section**
        hyperparam_section = QGridLayout()

        # **🔹 Feature & Target Selection (Column 1)**
        feature_target_group = QGroupBox()
        feature_target_layout = QVBoxLayout()
        self.add_feature_target_selection(feature_target_layout)
        feature_target_group.setLayout(feature_target_layout)
        hyperparam_section.addWidget(feature_target_group, 0, 0)

        # **🔹 Model Selection (Column 2)**
        model_selection_group = QGroupBox()
        model_selection_layout = QVBoxLayout()
        self.add_model_selection(model_selection_layout)
        model_selection_group.setLayout(model_selection_layout)
        hyperparam_section.addWidget(model_selection_group, 0, 1)

        # **🔹 Training Method Selection (Column 3)**
        training_method_group = QGroupBox()
        training_method_layout = QVBoxLayout()
        self.add_training_method_selection(training_method_layout)
        training_method_group.setLayout(training_method_layout)
        hyperparam_section.addWidget(training_method_group, 0, 2)

        # **🔹 Scheduler Selection (Row 2, Column 1)**
        scheduler_group = QGroupBox()
        scheduler_layout = QVBoxLayout()
        self.add_scheduler_selection(scheduler_layout)
        scheduler_group.setLayout(scheduler_layout)
        hyperparam_section.addWidget(scheduler_group, 1, 0)

        # **🔹 Validation Criteria (Row 2, Column 2)**
        validation_group = QGroupBox()
        validation_criteria_layout = QVBoxLayout()
        self.add_validation_criteria(validation_criteria_layout)
        validation_group.setLayout(validation_criteria_layout)
        hyperparam_section.addWidget(validation_group, 1, 1)

        # **🔹 Device Selection (Row 2, Column 3)**
        device_selection_group = QGroupBox()
        device_selection_layout = QVBoxLayout()
        self.add_device_selection(device_selection_layout)
        device_selection_group.setLayout(device_selection_layout)
        hyperparam_section.addWidget(device_selection_group, 1, 2)

        # ** Add Hyperparameter Sections to the Main Layout**
        main_layout.addLayout(hyperparam_section)

        # **📌 Bottom Buttons**
        button_layout = QVBoxLayout()  # Changed to vertical layout for separate rows

        load_button = QPushButton("Load Params from File")
        load_button.setFixedWidth(220)
        load_button.setFixedHeight(30)  # Reduced height
        load_button.setStyleSheet("font-size: 10pt;")
        load_button.clicked.connect(self.load_params_from_json)
        button_layout.addWidget(load_button, alignment=Qt.AlignCenter)

        start_button = QPushButton("Create Training Tasks")
        start_button.setFixedWidth(220)
        start_button.setFixedHeight(35)
        start_button.setStyleSheet("background-color: #0b6337; color: white; font-size: 10pt;")
        start_button.clicked.connect(self.proceed_to_training)
        button_layout.addWidget(start_button, alignment=Qt.AlignCenter)

        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

    def add_feature_target_selection(self, layout):
        """Adds a vertically stacked feature and target selection UI components with tooltips."""

        column_names = self.load_column_names()

        # **Feature Selection**
        feature_label = QLabel("Feature Columns (Input):")
        feature_label.setStyleSheet("font-size: 11pt; font-weight: bold;")
        feature_label.setToolTip("Select one or more columns as input features for training.")

        self.feature_list = QListWidget()
        self.feature_list.addItems(column_names)
        self.feature_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.feature_list.setFixedHeight(120)  # Reduce height for compactness
        self.feature_list.setFixedWidth(180)  # Adjust width if needed
        self.feature_list.setToolTip("Select multiple features.")

        # **Target Selection**
        target_label = QLabel("Target Column (Output):")
        target_label.setStyleSheet("font-size: 11pt; font-weight: bold;")
        target_label.setToolTip("<html><body><span style='font-weight: normal;'>Select the output column for the model to predict.</span></body></html>")

        self.target_combo = QComboBox()
        self.target_combo.addItems(column_names)
        self.target_combo.setFixedWidth(180)  # Match width with feature list
        self.target_combo.setToolTip("Select a single target column.")

        # ✅ Store references in self.param_entries for easy parameter collection
        self.param_entries["FEATURE_COLUMNS"] = self.feature_list
        self.param_entries["TARGET_COLUMN"] = self.target_combo

        # **Vertical Layout (Stacked)**
        v_layout = QVBoxLayout()
        v_layout.addWidget(feature_label)
        v_layout.addWidget(self.feature_list)
        v_layout.addWidget(target_label)
        v_layout.addWidget(self.target_combo)
        v_layout.addStretch(1)  # Ensures compact spacing

        # **Apply to Parent Layout**
        layout.addLayout(v_layout)

    def add_training_method_selection(self, layout):
        """Adds training method selection with batch size, train-validation split, tooltips, and ensures UI alignment."""

        # **Main Layout with Top Alignment**
        training_layout = QVBoxLayout()
        training_layout.setAlignment(Qt.AlignTop)  # Ensures content stays at the top

        # **Training Method Selection Dropdown**
        training_method_label = QLabel("Training Method:")
        training_method_label.setStyleSheet("font-size: 11pt; font-weight: bold;")
        training_method_label.setToolTip("Choose how training data is processed.")

        self.training_method_combo = QComboBox()
        training_options = ["Sequence-to-Sequence", "Whole Sequence"]
        self.training_method_combo.addItems(training_options)
        self.training_method_combo.setFixedWidth(200)
        self.training_method_combo.setToolTip(
            "Sequence-to-Sequence: Processes data in fixed time steps.\n"
            "Whole Sequence: Uses the entire sequence for training."
        )

        # **Lookback Parameter (Only for Sequence-to-Sequence)**
        self.lookback_label = QLabel("Lookback Window:")
        self.lookback_label.setStyleSheet("font-size: 11pt; font-weight: bold;") # Make bold
        self.lookback_label.setToolTip("Defines how many previous time steps are used for each prediction.")
        self.lookback_entry = QLineEdit(self.params.get("LOOKBACK", "400"))
        self.lookback_entry.setFixedWidth(150)

        # **Batch Training Option (Checkbox)**
        self.batch_training_checkbox = QCheckBox("Enable Batch Training")
        self.batch_training_checkbox.setChecked(True)  # Default is now checked
        self.batch_training_checkbox.setToolTip("Enable mini-batch training for sequence-based methods. Uncheck for full-batch of sequences. Irrelevant if 'Whole Sequence' is chosen for RNNs.")
        self.batch_training_checkbox.stateChanged.connect(self.update_batch_size_visibility)

        # **Batch Size Entry (Initially Enabled as checkbox is checked by default)**
        self.batch_size_label = QLabel("Batch Size:") # Made it an instance variable to hide/show
        self.batch_size_label.setStyleSheet("font-size: 11pt; font-weight: bold;") # Make bold
        self.batch_size_label.setToolTip("Number of samples per batch (if batch training is enabled and not 'Whole Sequence' RNN).")
        self.batch_size_entry = QLineEdit(self.params.get("BATCH_SIZE", "100")) # Default value 100
        self.batch_size_entry.setFixedWidth(150)
        self.batch_size_entry.setEnabled(True)  # Initially enabled

        # **Train-Validation Split**
        train_val_split_label = QLabel("Train-Valid Split:")
        train_val_split_label.setStyleSheet("font-size: 11pt; font-weight: bold;")
        train_val_split_label.setToolTip("Proportion of data allocated for training. The rest is used for validation.")
        
        self.train_val_split_entry = QLineEdit(self.params.get("TRAIN_VAL_SPLIT", "0.8"))
        self.train_val_split_entry.setFixedWidth(150)
        self.train_val_split_entry.setToolTip("Enter a value between 0 and 1 (e.g., 0.8 means 80% training, 20% validation).")

        # ✅ Store references in self.param_entries for easy parameter collection
        self.param_entries["TRAINING_METHOD"] = self.training_method_combo
        self.param_entries["LOOKBACK"] = self.lookback_entry
        self.param_entries["BATCH_TRAINING"] = self.batch_training_checkbox
        self.param_entries["BATCH_SIZE"] = self.batch_size_entry
        self.param_entries["TRAIN_VAL_SPLIT"] = self.train_val_split_entry

        # Initially hide lookback if Whole Sequence is selected
        self.lookback_label.setVisible(self.training_method_combo.currentText() == "Sequence-to-Sequence")
        self.lookback_entry.setVisible(self.training_method_combo.currentText() == "Sequence-to-Sequence")

        # **Update Visibility Based on Selection**
        self.training_method_combo.currentIndexChanged.connect(self.update_training_method)

        # **Add Widgets to Layout in Vertical Order**
        training_layout.addWidget(training_method_label)
        training_layout.addWidget(self.training_method_combo)
        training_layout.addWidget(self.lookback_label)
        training_layout.addWidget(self.lookback_entry)
        training_layout.addWidget(self.batch_training_checkbox)
        training_layout.addWidget(self.batch_size_label) # Use instance variable
        training_layout.addWidget(self.batch_size_entry)
        training_layout.addWidget(train_val_split_label)
        training_layout.addWidget(self.train_val_split_entry)

        # **Apply Layout to Parent Layout**
        layout.addLayout(training_layout)


    def update_training_method(self):
        """Toggle lookback, batch training checkbox, and batch size visibility based on training method and model type."""
        current_training_method = self.training_method_combo.currentText()
        current_model_type = self.model_combo.currentText() # Assuming self.model_combo exists and is accessible

        is_rnn_model = current_model_type in ["LSTM", "GRU"]
        is_whole_sequence_rnn = (current_training_method == "Whole Sequence" and is_rnn_model)
        is_sequence_to_sequence = (current_training_method == "Sequence-to-Sequence")

        # Lookback visibility (only for Sequence-to-Sequence)
        self.lookback_label.setVisible(is_sequence_to_sequence)
        self.lookback_entry.setVisible(is_sequence_to_sequence)
        self.lookback_entry.setEnabled(is_sequence_to_sequence)


        # Batch training checkbox and Batch size field visibility/state
        if is_whole_sequence_rnn:
            self.batch_training_checkbox.setVisible(False)
            self.batch_training_checkbox.setChecked(False) # Effectively disabled
            self.batch_training_checkbox.setEnabled(False)
            self.batch_size_label.setVisible(False)
            self.batch_size_entry.setVisible(False)
            self.batch_size_entry.setEnabled(False)
        else: # Sequence-to-Sequence or FNN with Whole Sequence (where batching might still apply)
            self.batch_training_checkbox.setVisible(True)
            self.batch_training_checkbox.setEnabled(True)
            # Batch size visibility depends on the checkbox state if the checkbox itself is visible
            self.update_batch_size_visibility() # Call this to set batch_size_entry based on checkbox

    def update_batch_size_visibility(self):
        """Enable or disable batch size input based on batch training checkbox, only if checkbox is visible."""
        if self.batch_training_checkbox.isVisible():
            is_checked = self.batch_training_checkbox.isChecked()
            self.batch_size_label.setVisible(True) # Show label if checkbox is visible
            self.batch_size_entry.setVisible(True) # Show entry if checkbox is visible
            self.batch_size_entry.setEnabled(is_checked)
        else: # If checkbox is not visible (e.g. Whole Sequence RNN), hide batch size too
            self.batch_size_label.setVisible(False)
            self.batch_size_entry.setVisible(False)
            self.batch_size_entry.setEnabled(False)


    def add_model_selection(self, layout):
        """Adds aligned model selection UI components with top alignment and tooltips."""

        # **Main Vertical Layout (Ensures Content Starts at the Top)**
        model_layout = QVBoxLayout()
        model_layout.setAlignment(Qt.AlignTop)  # Ensures text stays at the top

        # **Model Selection Dropdown**
        model_label = QLabel("Select Model:")
        model_label.setStyleSheet("font-size: 11pt; font-weight: bold;")
        model_label.setToolTip("Select a model architecture for training.")

        self.model_combo = QComboBox()
        model_options = ["LSTM", "FNN", "GRU"]
        self.model_combo.addItems(model_options)
        self.model_combo.setFixedWidth(180)
        self.model_combo.setToolTip("LSTM for time-series, CNN for feature extraction, GRU for memory-efficient training, Transformer for advanced architectures.")

        # **Model-Specific Parameters Placeholder**
        self.model_param_container = QVBoxLayout()

        # **Store reference in param_entries**
        self.param_entries["MODEL_TYPE"] = self.model_combo

        # Connect Dropdown to Update Parameters
        self.model_combo.currentIndexChanged.connect(self.update_model_params)
        self.model_combo.currentIndexChanged.connect(self.update_training_method) # Also trigger training method updates

        # **Add Widgets in Order**
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        model_layout.addLayout(self.model_param_container)  # Placeholder for model-specific params

        # **Apply Layout to Parent Layout**
        layout.addLayout(model_layout)

        # **Set Default to LSTM and Populate Parameters**
        self.model_combo.setCurrentText("LSTM")  # Ensure LSTM is selected
        self.update_model_params()  # Populate default parameters


    def update_model_params(self):
        """Dynamically updates parameter fields based on selected model and stores them in param_entries."""
        selected_model = self.model_combo.currentText()

        # --- Clear previous model-specific QLineEdit entries from self.param_entries ---
        # Define keys for model-specific parameters that might exist from a previous selection
        lstm_specific_keys = ["LAYERS", "HIDDEN_UNITS"] # Add any other LSTM specific QLineEdit keys
        gru_specific_keys = ["GRU_LAYERS"] # Add any other GRU specific QLineEdit keys
        fnn_specific_keys = ["FNN_HIDDEN_LAYERS", "FNN_DROPOUT_PROB"] # Add FNN specific QLineEdit keys
        
        all_model_specific_keys = lstm_specific_keys + gru_specific_keys + fnn_specific_keys
        
        for key_to_remove in all_model_specific_keys:
            if key_to_remove in self.param_entries:
                # We don't delete the widget here as it's handled by clearing model_param_container
                # Just remove the reference from param_entries
                del self.param_entries[key_to_remove]
        # --- End clearing stale entries ---

        # **Clear only dynamic parameter widgets (Keep Label & Dropdown)**
        while self.model_param_container.count():
            item = self.model_param_container.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        # **Ensure Param Entries Are Tracked**
        model_params = {} # This will hold QLineEdit widgets for the current model

        # **Model-Specific Parameters**
        if selected_model == "LSTM" or selected_model == "":
            lstm_layers_label = QLabel("LSTM Layers:")
            lstm_layers_label.setStyleSheet("font-size: 11pt; font-weight: bold;") # Make bold
            lstm_layers_label.setToolTip("Number of LSTM layers in the model.")

            self.lstm_layers_entry = QLineEdit(self.params.get("LAYERS", "1"))
            self.lstm_layers_entry.setFixedWidth(100)
            self.lstm_layers_entry.setToolTip("Enter the number of stacked LSTM layers.")

            hidden_units_label = QLabel("Hidden Units:")
            hidden_units_label.setStyleSheet("font-size: 11pt; font-weight: bold;") # Make bold
            hidden_units_label.setToolTip("Number of hidden units per layer.")

            self.hidden_units_entry = QLineEdit(self.params.get("HIDDEN_UNITS", "10"))
            self.hidden_units_entry.setFixedWidth(100)
            self.hidden_units_entry.setToolTip("Enter the number of hidden units per LSTM layer.")

            self.model_param_container.addWidget(lstm_layers_label)
            self.model_param_container.addWidget(self.lstm_layers_entry)
            self.model_param_container.addWidget(hidden_units_label)
            self.model_param_container.addWidget(self.hidden_units_entry)

            # ✅ Store in param_entries
            model_params["LAYERS"] = self.lstm_layers_entry
            model_params["HIDDEN_UNITS"] = self.hidden_units_entry

        # **GRU Parameters**
        elif selected_model == "GRU":
            gru_layers_label = QLabel("GRU Layers:")
            gru_layers_label.setStyleSheet("font-size: 11pt; font-weight: bold;") # Make bold
            gru_layers_label.setToolTip("Number of GRU layers in the model.")

            self.gru_layers_entry = QLineEdit(self.params.get("GRU_LAYERS", "2"))
            self.gru_layers_entry.setFixedWidth(100)
            self.gru_layers_entry.setToolTip("Enter the number of stacked GRU layers.")

            self.model_param_container.addWidget(gru_layers_label)
            self.model_param_container.addWidget(self.gru_layers_entry)

            # ✅ Store in param_entries
            model_params["GRU_LAYERS"] = self.gru_layers_entry

        elif selected_model == "FNN":
            fnn_hidden_layers_label = QLabel("FNN Hidden Layers:")
            fnn_hidden_layers_label.setStyleSheet("font-size: 11pt; font-weight: bold;") # Make bold
            fnn_hidden_layers_label.setToolTip("Define FNN hidden layer sizes. Use commas for layers in one config (e.g., 128,64,32). Use semicolons to separate multiple configs (e.g., 128,64;100,50).")
            self.fnn_hidden_layers_entry = QLineEdit(self.params.get("FNN_HIDDEN_LAYERS", "128,64"))
            self.fnn_hidden_layers_entry.setFixedWidth(200) # Increased width for longer strings
            self.fnn_hidden_layers_entry.setToolTip("E.g., '128,64,32' for one config. '128,64;100,50,25' for two configs.")
            
            fnn_dropout_label = QLabel("FNN Dropout Prob:")
            fnn_dropout_label.setStyleSheet("font-size: 11pt; font-weight: bold;") # Make bold
            fnn_dropout_label.setToolTip("Dropout probability for FNN layers (0.0 to 1.0).")
            self.fnn_dropout_entry = QLineEdit(self.params.get("FNN_DROPOUT_PROB", "0.1"))
            self.fnn_dropout_entry.setFixedWidth(100)
            self.fnn_dropout_entry.setToolTip("e.g., 0.1 for 10% dropout")

            self.model_param_container.addWidget(fnn_hidden_layers_label)
            self.model_param_container.addWidget(self.fnn_hidden_layers_entry)
            self.model_param_container.addWidget(fnn_dropout_label)
            self.model_param_container.addWidget(self.fnn_dropout_entry)

            # ✅ Store in model_params for later update to self.param_entries
            model_params["FNN_HIDDEN_LAYERS"] = self.fnn_hidden_layers_entry
            model_params["FNN_DROPOUT_PROB"] = self.fnn_dropout_entry

        # ✅ Register current model-specific QLineEdit parameters in self.param_entries
        # This ensures self.param_entries only contains widgets relevant to the *current* model type
        self.param_entries.update(model_params)



    def add_scheduler_selection(self, layout):
        """Adds learning rate scheduler selection UI components with dynamic label updates and Initial LR."""

        # **Scheduler Selection**
        scheduler_label = QLabel("Learning Rate Scheduler:")
        scheduler_label.setStyleSheet("font-size: 11pt; font-weight: bold;")
        scheduler_label.setToolTip("Select a scheduler to adjust the learning rate during training.")

        self.scheduler_combo = QComboBox()
        scheduler_options = ["StepLR", "ReduceLROnPlateau"]
        self.scheduler_combo.addItems(scheduler_options)
        self.scheduler_combo.setFixedWidth(180)
        self.scheduler_combo.setToolTip(
            "StepLR: Reduces LR at fixed intervals.\n"
            "ReduceLROnPlateau: Reduces LR when training stagnates."
        )

        # ✅ Store in self.param_entries
        self.param_entries["SCHEDULER_TYPE"] = self.scheduler_combo

        # **Initial Learning Rate (Common Parameter)**
        initial_lr_label = QLabel("Initial Learning Rate:")
        initial_lr_label.setStyleSheet("font-size: 11pt; font-weight: bold;") # Make bold
        initial_lr_label.setToolTip("The starting learning rate for the optimizer.")
        self.initial_lr_entry = QLineEdit(self.params.get("INITIAL_LR", "0.0001"))
        self.initial_lr_entry.setFixedWidth(100)
        self.initial_lr_entry.setToolTip("Lower values may stabilize training but slow convergence.")
        self.param_entries["INITIAL_LR"] = self.initial_lr_entry

        # **StepLR Parameters**
        self.lr_param_label = QLabel("LR Drop Factor:")  # Dynamic label
        self.lr_param_label.setStyleSheet("font-size: 11pt; font-weight: bold;") # Make bold
        self.lr_param_label.setToolTip("Factor by which LR is reduced (e.g., 0.1 means LR reduces by 10%).")
        self.lr_param_entry = QLineEdit(self.params.get("LR_DROP_FACTOR", "0.1"))
        self.lr_param_entry.setFixedWidth(100)
        self.lr_param_entry.setToolTip("Lower values reduce LR more aggressively.")
        self.param_entries["LR_PARAM"] = self.lr_param_entry

        self.lr_period_label = QLabel("LR Drop Period:")
        self.lr_period_label.setStyleSheet("font-size: 11pt; font-weight: bold;") # Make bold
        self.lr_period_label.setToolTip("Number of epochs before LR is reduced.")
        self.lr_period_entry = QLineEdit(self.params.get("LR_DROP_PERIOD", "5"))
        self.lr_period_entry.setFixedWidth(100)
        self.lr_period_entry.setToolTip("A smaller value means LR drops more frequently.")
        self.param_entries["LR_PERIOD"] = self.lr_period_entry

        # **ReduceLROnPlateau Parameters**
        self.patience_label = QLabel("Patience:")
        self.patience_label.setStyleSheet("font-size: 11pt; font-weight: bold;") # Make bold
        self.patience_label.setToolTip("Number of epochs with no improvement before LR is reduced.")
        self.patience_entry = QLineEdit(self.params.get("PATIENCE", "3"))
        self.patience_entry.setFixedWidth(100)
        self.patience_entry.setToolTip("Higher values make the scheduler less sensitive to short-term fluctuations.")
        self.param_entries["PATIENCE"] = self.patience_entry

        # **Connect Dropdown to Update Settings**
        self.scheduler_combo.currentIndexChanged.connect(self.update_scheduler_settings)

        # **Add Widgets to Layout**
        scheduler_layout = QVBoxLayout()
        scheduler_layout.setAlignment(Qt.AlignTop)
        scheduler_layout.addWidget(scheduler_label)
        scheduler_layout.addWidget(self.scheduler_combo)
        scheduler_layout.addWidget(initial_lr_label)
        scheduler_layout.addWidget(self.initial_lr_entry)
        scheduler_layout.addWidget(self.lr_param_label)
        scheduler_layout.addWidget(self.lr_param_entry)
        scheduler_layout.addWidget(self.lr_period_label)
        scheduler_layout.addWidget(self.lr_period_entry)
        scheduler_layout.addWidget(self.patience_label)
        scheduler_layout.addWidget(self.patience_entry)

        # **Apply to Parent Layout**
        layout.addLayout(scheduler_layout)

        # **Set Default to StepLR and Update Visibility**
        self.scheduler_combo.setCurrentText("StepLR")
        self.update_scheduler_settings()

    def update_scheduler_settings(self):
        """Updates the visibility and labels of scheduler parameters based on the selected scheduler."""
        selected_scheduler = self.scheduler_combo.currentText()

        is_step_lr = (selected_scheduler == "StepLR")
        is_reduce_lr = (selected_scheduler == "ReduceLROnPlateau")

        # **StepLR Specific**
        self.lr_param_label.setText("LR Drop Factor:" if is_step_lr else "LR Reduction Factor:")
        self.lr_param_label.setVisible(True)
        self.lr_param_entry.setVisible(True)
        self.lr_period_label.setVisible(is_step_lr)
        self.lr_period_entry.setVisible(is_step_lr)

        # **ReduceLROnPlateau Specific**
        self.patience_label.setVisible(is_reduce_lr)
        self.patience_entry.setVisible(is_reduce_lr)


    def add_validation_criteria(self, layout):
        """Adds UI components for validation criteria with top alignment and tooltips."""

        # **Main Vertical Layout**
        validation_layout = QVBoxLayout()
        validation_layout.setAlignment(Qt.AlignTop)

        # **Epochs**
        epochs_label = QLabel("Epochs:")
        epochs_label.setStyleSheet("font-size: 11pt; font-weight: bold;")
        epochs_label.setToolTip("Total number of training cycles.")
        self.epochs_entry = QLineEdit(self.params.get("EPOCHS", "10"))
        self.epochs_entry.setFixedWidth(100)
        self.epochs_entry.setToolTip("More epochs can improve accuracy but risk overfitting.")
        self.param_entries["EPOCHS"] = self.epochs_entry

        # **Early Stopping**
        early_stopping_label = QLabel("Early Stopping Patience:")
        early_stopping_label.setStyleSheet("font-size: 11pt; font-weight: bold;")
        early_stopping_label.setToolTip("Number of epochs to wait for improvement before stopping training.")
        self.early_stopping_entry = QLineEdit(self.params.get("EARLY_STOPPING_PATIENCE", "3"))
        self.early_stopping_entry.setFixedWidth(100)
        self.early_stopping_entry.setToolTip("Prevents overfitting by stopping when performance on the validation set stops improving.")
        self.param_entries["EARLY_STOPPING_PATIENCE"] = self.early_stopping_entry

        # **Validation Frequency**
        validation_freq_label = QLabel("Validation Frequency:")
        validation_freq_label.setStyleSheet("font-size: 11pt; font-weight: bold;")
        validation_freq_label.setToolTip("How often to run validation (in epochs).")
        self.validation_freq_entry = QLineEdit(self.params.get("VALIDATION_FREQ", "1"))
        self.validation_freq_entry.setFixedWidth(100)
        self.validation_freq_entry.setToolTip("e.g., '1' means validate after every epoch.")
        self.param_entries["VALIDATION_FREQ"] = self.validation_freq_entry

        # **Repetitions**
        repetitions_label = QLabel("Repetitions:")
        repetitions_label.setStyleSheet("font-size: 11pt; font-weight: bold;")
        repetitions_label.setToolTip("Number of times to repeat each training configuration (e.g., for statistical significance).")
        self.repetitions_entry = QLineEdit(self.params.get("REPETITIONS", "1"))
        self.repetitions_entry.setFixedWidth(100)
        self.repetitions_entry.setToolTip("Enter an integer value (e.g., 1, 3, 5).")
        self.param_entries["REPETITIONS"] = self.repetitions_entry

        # **Add Widgets to Layout**
        validation_layout.addWidget(epochs_label)
        validation_layout.addWidget(self.epochs_entry)
        validation_layout.addWidget(early_stopping_label)
        validation_layout.addWidget(self.early_stopping_entry)
        validation_layout.addWidget(validation_freq_label)
        validation_layout.addWidget(self.validation_freq_entry)
        validation_layout.addWidget(repetitions_label)
        validation_layout.addWidget(self.repetitions_entry)

        # **Apply to Parent Layout**
        layout.addLayout(validation_layout)

    def add_device_selection(self, layout):
        """Adds device selection UI components with top alignment and tooltips."""
        device_layout = QVBoxLayout()
        device_layout.setAlignment(Qt.AlignTop)

        device_label = QLabel("Select Device:")
        device_label.setStyleSheet("font-size: 11pt; font-weight: bold;")
        device_label.setToolTip("Select the hardware device for training.")
        device_layout.addWidget(device_label)

        self.device_combo = QComboBox()
        if torch.cuda.is_available():
            device_options = ["cuda", "cpu"]
            self.device_combo.setToolTip("CUDA-enabled GPU detected. Select 'cuda' for faster training.")
        else:
            device_options = ["cpu"]
            self.device_combo.setToolTip("No CUDA-enabled GPU detected. Training will run on the CPU.")
        self.device_combo.addItems(device_options)
        self.device_combo.setFixedWidth(120)
        device_layout.addWidget(self.device_combo)

        self.mixed_precision_checkbox = QCheckBox("Use Mixed Precision")
        self.mixed_precision_checkbox.setToolTip("Use 16-bit precision for faster training on compatible GPUs.")
        device_layout.addWidget(self.mixed_precision_checkbox)

        max_time_label = QLabel("Max Training Time (hh:mm:ss):")
        max_time_label.setStyleSheet("font-size: 11pt; font-weight: bold;")
        max_time_label.setToolTip("Set a time limit for training (hh:mm:ss). Set to 00:00:00 for no limit.")
        device_layout.addWidget(max_time_label)

        self.max_time_edit = QTimeEdit()
        self.max_time_edit.setDisplayFormat("hh:mm:ss")
        self.max_time_edit.setTime(QTime(0, 0, 0)) # Default to 00:00:00
        self.max_time_edit.setFixedWidth(100)
        device_layout.addWidget(self.max_time_edit)

        self.param_entries["DEVICE"] = self.device_combo
        self.param_entries["USE_MIXED_PRECISION"] = self.mixed_precision_checkbox
        self.param_entries["MAX_TRAINING_TIME"] = self.max_time_edit # Updated key

        layout.addLayout(device_layout)


    def proceed_to_training(self):
        """Collects parameters and sends a request to the backend to start the training job."""
        params = self.collect_parameters()
        if params is None:
            # Error message is shown in collect_parameters, just exit
            return

        task_info = {
            "hyperparams": params,
            "DEVICE_SELECTION": params.get("DEVICE_SELECTION", "cpu")
        }

        try:
            # First, save the hyperparameters
            self.api_gateway.save_hyperparameters(self.job_id, params)
            
            # Transition to the training setup GUI
            self.training_setup_gui = VEstimTrainSetupGUI(job_id=self.job_id, api_gateway=self.api_gateway)
            self.training_setup_gui.show()
            self.close()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to proceed to training setup: {e}")


    def load_column_names(self):
        """Loads column names by finding a CSV in the job's processed data folder."""
        try:
            job_details = self.api_gateway.get_job(self.job_id)
            if not job_details:
                self.logger.error("Could not retrieve job details.")
                return []

            job_folder = job_details.get('job_folder')
            train_folder = os.path.join(job_folder, 'train_data', 'processed_data')

            if not os.path.exists(train_folder):
                self.logger.warning(f"Train folder not found: {train_folder}")
                return []

            for file in os.listdir(train_folder):
                if file.lower().endswith('.csv'):
                    csv_path = os.path.join(train_folder, file)
                    df = pd.read_csv(csv_path, nrows=1)
                    return df.columns.tolist()
            
            self.logger.warning("No CSV files found in the train folder.")
            return []
        except Exception as e:
            self.logger.error(f"Failed to load column names: {e}")
            QMessageBox.critical(self, "Error", f"Could not load column names: {e}")
            return []

    def load_params_from_json(self):
        """Opens a file dialog to load hyperparameters from a JSON file."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Hyperparameters", "", "JSON Files (*.json)", options=options)
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    new_params = json.load(f)
                self.update_params(new_params)
                QMessageBox.information(self, "Success", "Parameters loaded successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load parameters: {e}")

    def update_params(self, new_params):
        """Update hyperparameters and refresh the UI."""
        try:
            self.params.update(new_params)
            self.update_gui_with_loaded_params()
            self.logger.info("Parameters updated and GUI refreshed.")
        except Exception as e:
            self.logger.error(f"Failed to update parameters: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"An error occurred while updating parameters: {e}")

    def update_gui_with_loaded_params(self):
        """Refreshes the GUI with the current self.params dictionary."""
        # Update simple QLineEdit and QComboBox widgets
        for key, widget in self.param_entries.items():
            if key in self.params:
                value = self.params[key]
                if isinstance(widget, QLineEdit):
                    widget.setText(str(value))
                elif isinstance(widget, QComboBox):
                    index = widget.findText(str(value), Qt.MatchFixedString)
                    if index >= 0:
                        widget.setCurrentIndex(index)
                elif isinstance(widget, QCheckBox):
                    widget.setChecked(bool(value))
                elif isinstance(widget, QTimeEdit) and key == "MAX_TRAINING_TIME":
                    time_value = QTime.fromString(str(value), "hh:mm:ss")
                    if time_value.isValid():
                        widget.setTime(time_value)
                    else:
                        widget.setTime(QTime(0,0,0)) # Default if parse fails

        # Update QListWidget for feature columns
        if "FEATURE_COLUMNS" in self.params and isinstance(self.param_entries["FEATURE_COLUMNS"], QListWidget):
            feature_list_widget = self.param_entries["FEATURE_COLUMNS"]
            feature_list_widget.clearSelection()
            selected_features = self.params["FEATURE_COLUMNS"]
            if isinstance(selected_features, list):
                for i in range(feature_list_widget.count()):
                    item = feature_list_widget.item(i)
                    if item.text() in selected_features:
                        item.setSelected(True)

        # Update model-specific parameters
        if "MODEL_TYPE" in self.params:
            self.model_combo.setCurrentText(self.params["MODEL_TYPE"])
            self.update_model_params() # This will rebuild the model-specific UI
            # Now, re-populate the newly created model-specific fields
            for key, widget in self.param_entries.items():
                if key in self.params and isinstance(widget, QLineEdit):
                    widget.setText(str(self.params[key]))

        # Update scheduler-specific parameters
        if "SCHEDULER_TYPE" in self.params:
            self.scheduler_combo.setCurrentText(self.params["SCHEDULER_TYPE"])
            self.update_scheduler_settings()
            # Re-populate scheduler fields
            for key, widget in self.param_entries.items():
                if key in self.params and isinstance(widget, QLineEdit):
                    widget.setText(str(self.params[key]))
        
        self.logger.info("GUI has been updated with loaded parameters.")


    def open_guide(self):
        """Opens the hyperparameter guide PDF."""
        try:
            # Correctly determine the path to the guide relative to this script
            guide_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', 'hyper_param_guide.pdf')
            if os.path.exists(guide_path):
                os.startfile(guide_path)
            else:
                QMessageBox.warning(self, "Guide Not Found", "The hyperparameter guide could not be found.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not open the guide: {e}")

    def collect_parameters(self):
        """
        Collects all parameters from the UI, validates them, and returns a dictionary.
        Returns None if validation fails.
        """
        params = {}
        try:
            # Feature and Target Columns
            feature_widget = self.param_entries["FEATURE_COLUMNS"]
            selected_features = [item.text() for item in feature_widget.selectedItems()]
            if not selected_features:
                raise ValueError("At least one feature column must be selected.")
            params["FEATURE_COLUMNS"] = selected_features

            target_widget = self.param_entries["TARGET_COLUMN"]
            params["TARGET_COLUMN"] = target_widget.currentText()
            if not params["TARGET_COLUMN"]:
                raise ValueError("A target column must be selected.")

            # Model Type
            params["MODEL_TYPE"] = self.param_entries["MODEL_TYPE"].currentText()

            # Dynamically collect other parameters
            for key, widget in self.param_entries.items():
                if key in ["FEATURE_COLUMNS", "TARGET_COLUMN", "MODEL_TYPE"]:
                    continue  # Already handled

                if not widget.isVisible():
                    continue # Skip hidden widgets

                if isinstance(widget, QLineEdit):
                    params[key] = widget.text()
                elif isinstance(widget, QComboBox):
                    params[key] = widget.currentText()
                elif isinstance(widget, QCheckBox):
                    params[key] = widget.isChecked()
                elif isinstance(widget, QTimeEdit) and key == "MAX_TRAINING_TIME":
                    params[key] = widget.time().toString("hh:mm:ss")
            
            # Final check for required fields based on visibility
            required_fields = {
                "LOOKBACK": "Lookback window is required for Sequence-to-Sequence.",
                "BATCH_SIZE": "Batch size is required for batch training.",
                "TRAIN_VAL_SPLIT": "Train-Valid split is required.",
                "INITIAL_LR": "Initial Learning Rate is required.",
                "EPOCHS": "Number of epochs is required.",
                "EARLY_STOPPING_PATIENCE": "Early stopping patience is required.",
                "VALIDATION_FREQ": "Validation frequency is required.",
                "REPETITIONS": "Number of repetitions is required."
            }

            for field, error_msg in required_fields.items():
                widget = self.param_entries.get(field)
                if widget and widget.isVisible() and not params.get(field):
                    raise ValueError(error_msg)

            self.logger.info(f"Collected parameters: {params}")
            return params

        except ValueError as ve:
            QMessageBox.warning(self, "Validation Error", str(ve))
            self.logger.error(f"Parameter validation failed: {ve}")
            return None
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An unexpected error occurred while collecting parameters: {e}")
            self.logger.error(f"Failed to collect parameters: {e}", exc_info=True)
            return None
