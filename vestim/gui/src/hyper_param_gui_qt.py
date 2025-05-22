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
    QLineEdit, QFileDialog, QMessageBox, QDialog, QGroupBox, QComboBox, QListWidget, QAbstractItemView,QFormLayout, QCheckBox
)
from PyQt5.QtCore import Qt, QPropertyAnimation
from PyQt5.QtGui import QIcon

import pandas as pd

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

        # **📌 Add Hyperparameter Sections to the Main Layout**
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
            lstm_layers_label.setToolTip("Number of LSTM layers in the model.")

            self.lstm_layers_entry = QLineEdit(self.params.get("LAYERS", "1"))
            self.lstm_layers_entry.setFixedWidth(100)
            self.lstm_layers_entry.setToolTip("Enter the number of stacked LSTM layers.")

            hidden_units_label = QLabel("Hidden Units:")
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
            fnn_hidden_layers_label.setToolTip("Define FNN hidden layer sizes. Use commas for layers in one config (e.g., 128,64,32). Use semicolons to separate multiple configs (e.g., 128,64;100,50).")
            self.fnn_hidden_layers_entry = QLineEdit(self.params.get("FNN_HIDDEN_LAYERS", "128,64"))
            self.fnn_hidden_layers_entry.setFixedWidth(200) # Increased width for longer strings
            self.fnn_hidden_layers_entry.setToolTip("E.g., '128,64,32' for one config. '128,64;100,50,25' for two configs.")
            
            fnn_dropout_label = QLabel("FNN Dropout Prob:")
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
        initial_lr_label.setToolTip("The starting learning rate for the optimizer.")
        self.initial_lr_entry = QLineEdit(self.params.get("INITIAL_LR", "0.0001"))
        self.initial_lr_entry.setFixedWidth(100)
        self.initial_lr_entry.setToolTip("Lower values may stabilize training but slow convergence.")
        self.param_entries["INITIAL_LR"] = self.initial_lr_entry

        # **StepLR Parameters**
        self.lr_param_label = QLabel("LR Drop Factor:")  # Dynamic label
        self.lr_param_label.setToolTip("Factor by which LR is reduced (e.g., 0.1 means LR reduces by 10%).")
        self.lr_param_entry = QLineEdit(self.params.get("LR_DROP_FACTOR", "0.1"))
        self.lr_param_entry.setFixedWidth(100)
        self.lr_param_entry.setToolTip("Lower values reduce LR more aggressively.")
        self.param_entries["LR_PARAM"] = self.lr_param_entry # This was LR_DROP_FACTOR in some versions
        # self.param_entries["LR_DROP_FACTOR"] = self.lr_param_entry # Ensure consistency if needed

        self.lr_period_label = QLabel("LR Drop Period:")
        self.lr_period_label.setToolTip("Number of epochs after which LR is reduced.")
        self.lr_period_entry = QLineEdit(self.params.get("LR_DROP_PERIOD", "1000"))
        self.lr_period_entry.setFixedWidth(100)
        self.lr_period_entry.setToolTip("Set higher values if you want the LR to stay stable for longer periods.")
        self.param_entries["LR_PERIOD"] = self.lr_period_entry # This was LR_DROP_PERIOD

        # **ReduceLROnPlateau Parameters**
        self.plateau_patience_label = QLabel("Plateau Patience:")
        self.plateau_patience_label.setToolTip("Number of epochs to wait before reducing LR if no improvement in validation.")
        self.plateau_patience_entry = QLineEdit(self.params.get("PLATEAU_PATIENCE", "10"))
        self.plateau_patience_entry.setFixedWidth(100)
        self.plateau_patience_entry.setToolTip("Larger values allow longer training before LR adjustment.")
        self.param_entries["PLATEAU_PATIENCE"] = self.plateau_patience_entry

        self.plateau_factor_label = QLabel("Plateau Factor:")
        self.plateau_factor_label.setToolTip("Factor by which LR is reduced when a plateau is detected (e.g., 0.1).")
        self.plateau_factor_entry = QLineEdit(self.params.get("PLATEAU_FACTOR", "0.1"))
        self.plateau_factor_entry.setFixedWidth(100)
        self.plateau_factor_entry.setToolTip("Smaller values make the LR reduction more significant.")
        self.param_entries["PLATEAU_FACTOR"] = self.plateau_factor_entry
        
        # Connect Dropdown to Update Scheduler Settings
        self.scheduler_combo.currentIndexChanged.connect(self.update_scheduler_settings)
        self.update_scheduler_settings()  # Initial call to set visibility

        # **Add Widgets to Layout**
        scheduler_form_layout = QFormLayout() # Use QFormLayout for better label-entry alignment
        scheduler_form_layout.addRow(scheduler_label, self.scheduler_combo)
        scheduler_form_layout.addRow(initial_lr_label, self.initial_lr_entry) # Add Initial LR here
        scheduler_form_layout.addRow(self.lr_param_label, self.lr_param_entry)
        scheduler_form_layout.addRow(self.lr_period_label, self.lr_period_entry)
        scheduler_form_layout.addRow(self.plateau_patience_label, self.plateau_patience_entry)
        scheduler_form_layout.addRow(self.plateau_factor_label, self.plateau_factor_entry)
        
        layout.addLayout(scheduler_form_layout)


    def update_scheduler_settings(self):
        """Shows/hides scheduler-specific parameters based on selection."""
        selected_scheduler = self.scheduler_combo.currentText()

        is_step_lr = (selected_scheduler == "StepLR")
        self.lr_param_label.setVisible(is_step_lr)
        self.lr_param_entry.setVisible(is_step_lr)
        self.lr_period_label.setVisible(is_step_lr)
        self.lr_period_entry.setVisible(is_step_lr)

        is_plateau = (selected_scheduler == "ReduceLROnPlateau")
        self.plateau_patience_label.setVisible(is_plateau)
        self.plateau_patience_entry.setVisible(is_plateau)
        self.plateau_factor_label.setVisible(is_plateau)
        self.plateau_factor_entry.setVisible(is_plateau)
        
        # Update dynamic label for LR_PARAM based on scheduler
        if is_step_lr:
            self.lr_param_label.setText("LR Drop Factor:")
        elif is_plateau:
            # This field is not used for ReduceLROnPlateau directly in this UI section,
            # but if it were, the label might change. Keeping it generic for now.
            # Or hide it if not applicable:
            # self.lr_param_label.setVisible(False) 
            # self.lr_param_entry.setVisible(False)
            pass # No direct equivalent for lr_param_label for Plateau in this setup


    def add_validation_criteria(self, layout):
        """Adds validation criteria UI components with top alignment and tooltips."""

        # **Main Vertical Layout (Ensures Content Starts at the Top)**
        validation_layout = QVBoxLayout()
        validation_layout.setAlignment(Qt.AlignTop)  # Ensures text stays at the top

        # **Validation Patience**
        valid_patience_label = QLabel("Validation Patience:")
        valid_patience_label.setStyleSheet("font-size: 11pt; font-weight: bold;")
        valid_patience_label.setToolTip("Number of epochs to wait for improvement before early stopping.")
        
        self.valid_patience_entry = QLineEdit(self.params.get("VALID_PATIENCE", "10"))
        self.valid_patience_entry.setFixedWidth(100)
        self.valid_patience_entry.setToolTip("Higher values allow more epochs without improvement before stopping.")

        # **Validation Frequency**
        valid_frequency_label = QLabel("Validation Frequency:")
        valid_frequency_label.setStyleSheet("font-size: 11pt; font-weight: bold;")
        valid_frequency_label.setToolTip("How often (in epochs) to perform validation.")
        
        self.valid_frequency_entry = QLineEdit(self.params.get("VALID_FREQUENCY", "1"))
        self.valid_frequency_entry.setFixedWidth(100)
        self.valid_frequency_entry.setToolTip("e.g., '1' means validate every epoch, '5' means every 5 epochs.")

        # **Max Epochs**
        max_epochs_label = QLabel("Max Epochs:")
        max_epochs_label.setStyleSheet("font-size: 11pt; font-weight: bold;")
        max_epochs_label.setToolTip("Maximum number of epochs to train for.")
        
        self.max_epochs_entry = QLineEdit(self.params.get("MAX_EPOCHS", "100"))
        self.max_epochs_entry.setFixedWidth(100)
        self.max_epochs_entry.setToolTip("Training will stop after this many epochs, even if not converged by patience.")

        # ✅ Store in self.param_entries
        self.param_entries["VALID_PATIENCE"] = self.valid_patience_entry
        self.param_entries["VALID_FREQUENCY"] = self.valid_frequency_entry
        self.param_entries["MAX_EPOCHS"] = self.max_epochs_entry

        # **Add Widgets to Layout**
        validation_form_layout = QFormLayout() # Use QFormLayout for better label-entry alignment
        validation_form_layout.addRow(valid_patience_label, self.valid_patience_entry)
        validation_form_layout.addRow(valid_frequency_label, self.valid_frequency_entry)
        validation_form_layout.addRow(max_epochs_label, self.max_epochs_entry)
        
        validation_layout.addLayout(validation_form_layout)
        
        # **Apply Layout to Parent Layout**
        layout.addLayout(validation_layout)


    def collect_hyperparameters(self):
        """Collects all hyperparameters from the UI fields into a dictionary."""
        collected_params = {}
        for key, widget in self.param_entries.items():
            if isinstance(widget, QLineEdit):
                collected_params[key] = widget.text()
            elif isinstance(widget, QComboBox):
                collected_params[key] = widget.currentText()
            elif isinstance(widget, QListWidget): # For feature selection
                selected_items = [item.text() for item in widget.selectedItems()]
                collected_params[key] = selected_items if selected_items else None # Store as list or None
            elif isinstance(widget, QCheckBox):
                collected_params[key] = widget.isChecked()
        
        # Handle specific cases like LOOKBACK which might be hidden
        if not self.lookback_entry.isVisible():
            collected_params["LOOKBACK"] = None # Or some other indicator that it's not applicable
        
        # Handle BATCH_SIZE if batch training is disabled or not applicable
        if not self.batch_training_checkbox.isChecked() or not self.batch_training_checkbox.isVisible():
             collected_params["BATCH_SIZE"] = None # Or indicate not applicable

        self.logger.info(f"Collected hyperparameters: {collected_params}")
        return collected_params

    def proceed_to_training(self):
        """Collects parameters, saves them, and proceeds to the training setup GUI."""
        self.logger.info("Proceed to Training button clicked.")
        try:
            current_params = self.collect_hyperparameters()
            
            # Validate that feature columns are selected
            if not current_params.get("FEATURE_COLUMNS"):
                QMessageBox.warning(self, "Input Error", "Please select at least one feature column.")
                return

            # Validate that a target column is selected
            if not current_params.get("TARGET_COLUMN"): # QComboBox returns current text, which is fine
                QMessageBox.warning(self, "Input Error", "Please select a target column.")
                return

            # Save parameters to a JSON file in the job folder
            job_folder_path = self.job_manager.get_job_folder() # Get current job folder
            if not job_folder_path:
                QMessageBox.critical(self, "Error", "No active job folder found. Please start from Data Import.")
                return

            self.hyper_param_manager.save_hyperparameters(current_params, job_folder_path)
            QMessageBox.information(self, "Success", f"Hyperparameters saved to {job_folder_path}")
            
            self.show_training_setup_gui() # Proceed to the next screen

        except Exception as e:
            self.logger.error(f"Error proceeding to training: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")
            
    def show_training_setup_gui(self):
        """Initializes and shows the Training Setup GUI."""
        self.training_setup_gui = VEstimTrainSetupGUI(job_manager.get_job_folder()) # Pass current job folder
        self.training_setup_gui.show()
        self.close() # Close the hyperparameter window

    def load_column_names(self):
        """Loads column names from a sample processed data file in the current job folder."""
        job_folder = self.job_manager.get_job_folder()
        if not job_folder:
            self.logger.warning("No job folder set. Cannot load column names.")
            return ["Default_Column1", "Default_Column2"] # Fallback

        # Path to a sample processed training file
        # Assuming 'train_data/processed_data' exists and contains CSVs
        processed_train_path = os.path.join(job_folder, "train_data", "processed_data")
        
        if not os.path.isdir(processed_train_path):
            self.logger.warning(f"Processed train data folder not found: {processed_train_path}")
            return ["No_Data_Found1", "No_Data_Found2"]

        try:
            for file_name in os.listdir(processed_train_path):
                if file_name.lower().endswith(".csv"): # Look for any CSV file
                    sample_file_path = os.path.join(processed_train_path, file_name)
                    df = pd.read_csv(sample_file_path, nrows=1) # Read only header
                    self.logger.info(f"Loaded columns from {sample_file_path}: {df.columns.tolist()}")
                    return df.columns.tolist()
            self.logger.warning(f"No CSV files found in {processed_train_path} to load column names.")
        except Exception as e:
            self.logger.error(f"Error loading column names from {processed_train_path}: {e}", exc_info=True)
        
        return ["Error_Loading_Cols1", "Error_Loading_Cols2"] # Fallback if error or no files

    def load_params_from_json(self):
        """Loads hyperparameters from a JSON file and updates the UI."""
        job_folder = self.job_manager.get_job_folder()
        if not job_folder:
            QMessageBox.warning(self, "Warning", "No job folder selected. Cannot determine where to look for 'hyperparams.json'.")
            # Optionally, allow browsing for a hyperparams.json file directly
            # options = QFileDialog.Options()
            # file_path, _ = QFileDialog.getOpenFileName(self, "Load Hyperparameters JSON", "", "JSON Files (*.json)", options=options)
            # if not file_path:
            #     return # User cancelled
            # else: # If user browses, use that path
            #     params = self.hyper_param_manager.load_hyperparameters_from_path(file_path)
            return 
        
        # Default to loading from the job folder
        params = self.hyper_param_manager.load_hyperparameters(job_folder)

        if params:
            self.update_params(params) # Update internal params
            self.update_gui_with_loaded_params() # Update UI fields
            QMessageBox.information(self, "Success", "Hyperparameters loaded successfully.")
        else:
            QMessageBox.warning(self, "Load Failed", f"Could not load hyperparameters from 'hyperparams.json' in {job_folder}. File might be missing or corrupted.")
            
    def update_params(self, new_params):
        """Update internal params and refresh the UI."""
        try:
            self.params.update(new_params)
            # Refresh UI elements based on self.params
            # This is now handled by update_gui_with_loaded_params
            # For example:
            # self.lookback_entry.setText(self.params.get("LOOKBACK", "400"))
            # self.model_combo.setCurrentText(self.params.get("MODEL_TYPE", "LSTM"))
            # ... and so on for all relevant fields.
            self.logger.info("Internal parameters updated with loaded values.")
        except Exception as e:
            self.logger.error(f"Error updating internal params: {e}", exc_info=True)


    def update_gui_with_loaded_params(self):
        """Updates all relevant GUI elements based on the currently loaded self.params."""
        self.logger.info("Updating GUI with loaded parameters...")

        # Update Feature and Target Selection
        if "FEATURE_COLUMNS" in self.params and self.params["FEATURE_COLUMNS"]:
            self.feature_list.clearSelection()
            for col_name in self.params["FEATURE_COLUMNS"]:
                items = self.feature_list.findItems(col_name, Qt.MatchExactly)
                if items:
                    items[0].setSelected(True)
        if "TARGET_COLUMN" in self.params:
            self.target_combo.setCurrentText(self.params["TARGET_COLUMN"])

        # Update Model Selection and its dynamic params
        if "MODEL_TYPE" in self.params:
            self.model_combo.setCurrentText(self.params["MODEL_TYPE"]) # This will trigger update_model_params
            # update_model_params will then populate specific fields like LAYERS, HIDDEN_UNITS
            # We need to ensure those fields are then set from self.params *after* they are created
            # This is handled by QLineEdit(self.params.get("KEY", "default")) in update_model_params

        # Update Training Method and its dynamic params
        if "TRAINING_METHOD" in self.params:
            self.training_method_combo.setCurrentText(self.params["TRAINING_METHOD"]) # Triggers update_training_method
            # update_training_method will show/hide lookback, batch_training_checkbox, batch_size
            # We need to set their values from self.params
        if "LOOKBACK" in self.params and self.lookback_entry.isVisible():
            self.lookback_entry.setText(self.params.get("LOOKBACK", ""))
        if "BATCH_TRAINING" in self.params and self.batch_training_checkbox.isVisible():
            self.batch_training_checkbox.setChecked(self.params.get("BATCH_TRAINING", True))
            # update_batch_size_visibility will be called by stateChanged if checkbox visibility itself changes
            # or by update_training_method. We might need an explicit call if only value changes.
            self.update_batch_size_visibility() # Ensure batch size field updates based on checkbox
        if "BATCH_SIZE" in self.params and self.batch_size_entry.isVisible() and self.batch_size_entry.isEnabled():
            self.batch_size_entry.setText(self.params.get("BATCH_SIZE", ""))
        if "TRAIN_VAL_SPLIT" in self.params:
            self.train_val_split_entry.setText(self.params.get("TRAIN_VAL_SPLIT", "0.8"))

        # Update Scheduler Selection and its dynamic params
        if "SCHEDULER_TYPE" in self.params:
            self.scheduler_combo.setCurrentText(self.params["SCHEDULER_TYPE"]) # Triggers update_scheduler_settings
        if "INITIAL_LR" in self.params:
            self.initial_lr_entry.setText(self.params.get("INITIAL_LR", "0.0001"))
        # For StepLR params (LR_PARAM was LR_DROP_FACTOR, LR_PERIOD was LR_DROP_PERIOD)
        if "LR_PARAM" in self.params and self.lr_param_entry.isVisible():
             self.lr_param_entry.setText(self.params.get("LR_PARAM", "0.1"))
        if "LR_PERIOD" in self.params and self.lr_period_entry.isVisible():
             self.lr_period_entry.setText(self.params.get("LR_PERIOD", "1000"))
        # For ReduceLROnPlateau params
        if "PLATEAU_PATIENCE" in self.params and self.plateau_patience_entry.isVisible():
            self.plateau_patience_entry.setText(self.params.get("PLATEAU_PATIENCE", "10"))
        if "PLATEAU_FACTOR" in self.params and self.plateau_factor_entry.isVisible():
            self.plateau_factor_entry.setText(self.params.get("PLATEAU_FACTOR", "0.1"))

        # Update Validation Criteria
        if "VALID_PATIENCE" in self.params:
            self.valid_patience_entry.setText(self.params.get("VALID_PATIENCE", "10"))
        if "VALID_FREQUENCY" in self.params:
            self.valid_frequency_entry.setText(self.params.get("VALID_FREQUENCY", "1"))
        if "MAX_EPOCHS" in self.params:
            self.max_epochs_entry.setText(self.params.get("MAX_EPOCHS", "100"))
            
        self.logger.info("GUI updated with loaded parameters.")


    def open_guide(self):
        """Opens the hyperparameter guide PDF."""
        # Determine the path to the guide relative to this script file
        # Assuming this script is in vestim/gui/src/
        # and resources is in vestim/gui/resources/
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__)) # dir of this script
            gui_dir = os.path.dirname(current_dir) # up to vestim/gui/
            guide_path = os.path.join(gui_dir, "resources", "hyper_param_guide.pdf")

            if os.path.exists(guide_path):
                if sys.platform == "win32":
                    os.startfile(guide_path)
                elif sys.platform == "darwin": # macOS
                    os.system(f"open \"{guide_path}\"")
                else: # linux variants
                    os.system(f"xdg-open \"{guide_path}\"")
            else:
                QMessageBox.warning(self, "Guide Not Found", f"Hyperparameter guide PDF not found at expected location: {guide_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error Opening Guide", f"Could not open hyperparameter guide: {e}")

# Example usage for testing this GUI directly
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
    
#     # Create a dummy job folder for testing
#     test_job_folder = "test_job_hyperparam_gui"
#     os.makedirs(os.path.join(test_job_folder, "train_data", "processed_data"), exist_ok=True)
#     # Create a dummy processed CSV file with some columns
#     dummy_data = pd.DataFrame({'Timestamp': [1, 2, 3], 'Voltage': [3.0, 3.1, 3.2], 'Current': [-1, -1, -1], 'SOC': [90, 89, 88]})
#     dummy_data.to_csv(os.path.join(test_job_folder, "train_data", "processed_data", "sample.csv"), index=False)
    
#     # Set the job folder using JobManager (simulating it was set by DataImportGUI)
#     job_manager.set_job_folder(test_job_folder) # This is how the GUI expects job_folder to be set
    
#     gui = VEstimHyperParamGUI()
#     gui.show()
#     sys.exit(app.exec_())
