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
from vestim.gateway.src.hyper_param_manager_qt_test import VEstimHyperParamManager
from vestim.gui.src.training_setup_gui_qt_test import VEstimTrainSetupGUI

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
        feature_label = QLabel("Feature Columns:")
        feature_label.setStyleSheet("font-size: 11pt; font-weight: bold;")
        feature_label.setToolTip("Select one or more columns as input features for training.")

        self.feature_list = QListWidget()
        self.feature_list.addItems(column_names)
        self.feature_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.feature_list.setFixedHeight(120)  # Reduce height for compactness
        self.feature_list.setFixedWidth(180)  # Adjust width if needed
        self.feature_list.setToolTip("Select multiple features.")

        # **Target Selection**
        target_label = QLabel("Target Column:")
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
        self.batch_training_checkbox.setChecked(False)  # Default is unchecked
        self.batch_training_checkbox.setToolTip("Enable mini-batch training instead of full-sequence training.")
        self.batch_training_checkbox.stateChanged.connect(self.update_batch_size_visibility)

        # **Batch Size Entry (Initially Disabled)**
        batch_size_label = QLabel("Batch Size:")
        batch_size_label.setToolTip("Number of samples per batch during training.")
        self.batch_size_entry = QLineEdit(self.params.get("BATCH_SIZE", "100"))
        self.batch_size_entry.setFixedWidth(150)
        self.batch_size_entry.setEnabled(False)  # Initially disabled

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
        training_layout.addWidget(batch_size_label)
        training_layout.addWidget(self.batch_size_entry)
        training_layout.addWidget(train_val_split_label)
        training_layout.addWidget(self.train_val_split_entry)

        # **Apply Layout to Parent Layout**
        layout.addLayout(training_layout)


    def update_training_method(self):
        """Toggle lookback parameter visibility based on training method selection."""
        is_seq_to_seq = self.training_method_combo.currentText() == "Sequence-to-Sequence"
        self.lookback_label.setVisible(is_seq_to_seq)
        self.lookback_entry.setVisible(is_seq_to_seq)

    def update_batch_size_visibility(self):
        """Enable or disable batch size input based on batch training checkbox."""
        self.batch_size_entry.setEnabled(self.batch_training_checkbox.isChecked())


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
        model_options = ["LSTM", "CNN", "GRU", "Transformer"]
        self.model_combo.addItems(model_options)
        self.model_combo.setFixedWidth(180)
        self.model_combo.setToolTip("LSTM for time-series, CNN for feature extraction, GRU for memory-efficient training, Transformer for advanced architectures.")

        # **Model-Specific Parameters Placeholder**
        self.model_param_container = QVBoxLayout()

        # **Store reference in param_entries**
        self.param_entries["MODEL_TYPE"] = self.model_combo

        # Connect Dropdown to Update Parameters
        self.model_combo.currentIndexChanged.connect(self.update_model_params)

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

        # **Clear only dynamic parameter widgets (Keep Label & Dropdown)**
        while self.model_param_container.count():
            item = self.model_param_container.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        # **Ensure Param Entries Are Tracked**
        model_params = {}

        # **Model-Specific Parameters (Default to LSTM)**
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

        # **CNN Parameters**
        elif selected_model == "CNN":
            cnn_layers_label = QLabel("CNN Layers:")
            cnn_layers_label.setToolTip("Number of convolutional layers.")

            self.cnn_layers_entry = QLineEdit(self.params.get("CNN_LAYERS", "3"))
            self.cnn_layers_entry.setFixedWidth(100)
            self.cnn_layers_entry.setToolTip("Enter the number of CNN layers.")

            kernel_size_label = QLabel("Kernel Size:")
            kernel_size_label.setToolTip("Size of the convolutional filter.")

            self.kernel_size_entry = QLineEdit(self.params.get("KERNEL_SIZE", "3"))
            self.kernel_size_entry.setFixedWidth(100)
            self.kernel_size_entry.setToolTip("Enter the kernel size for convolution.")

            self.model_param_container.addWidget(cnn_layers_label)
            self.model_param_container.addWidget(self.cnn_layers_entry)
            self.model_param_container.addWidget(kernel_size_label)
            self.model_param_container.addWidget(self.kernel_size_entry)

            # ✅ Store in param_entries
            model_params["CNN_LAYERS"] = self.cnn_layers_entry
            model_params["KERNEL_SIZE"] = self.kernel_size_entry

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

        # **Transformer Parameters**
        elif selected_model == "Transformer":
            attention_heads_label = QLabel("Attention Heads:")
            attention_heads_label.setToolTip("Number of attention heads in the transformer model.")

            self.attn_heads_entry = QLineEdit(self.params.get("ATTN_HEADS", "4"))
            self.attn_heads_entry.setFixedWidth(100)
            self.attn_heads_entry.setToolTip("Enter the number of attention heads.")

            self.model_param_container.addWidget(attention_heads_label)
            self.model_param_container.addWidget(self.attn_heads_entry)

            # ✅ Store in param_entries
            model_params["ATTN_HEADS"] = self.attn_heads_entry

        # ✅ Register model-specific parameters in param_entries
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
        self.param_entries["LR_PARAM"] = self.lr_param_entry

        self.lr_period_label = QLabel("LR Drop Period:")
        self.lr_period_label.setToolTip("Number of epochs after which LR is reduced.")
        self.lr_period_entry = QLineEdit(self.params.get("LR_DROP_PERIOD", "1000"))
        self.lr_period_entry.setFixedWidth(100)
        self.lr_period_entry.setToolTip("Set higher values if you want the LR to stay stable for longer periods.")
        self.param_entries["LR_PERIOD"] = self.lr_period_entry

        # **ReduceLROnPlateau Parameters**
        self.plateau_patience_label = QLabel("Plateau Patience:")
        self.plateau_patience_label.setToolTip("Number of epochs to wait before reducing LR if no improvement in validation.")
        self.plateau_patience_entry = QLineEdit(self.params.get("PLATEAU_PATIENCE", "10"))
        self.plateau_patience_entry.setFixedWidth(100)
        self.plateau_patience_entry.setToolTip("Larger values allow longer training before LR adjustment.")
        self.param_entries["PLATEAU_PATIENCE"] = self.plateau_patience_entry

        self.plateau_factor_label = QLabel("Plateau Factor:")
        self.plateau_factor_label.setToolTip("Factor by which LR is reduced when ReduceLROnPlateau is triggered.")
        self.plateau_factor_entry = QLineEdit(self.params.get("PLATEAU_FACTOR", "0.1"))
        self.plateau_factor_entry.setFixedWidth(100)
        self.plateau_factor_entry.setToolTip("Lower values make the LR decrease more significantly.")
        self.param_entries["PLATEAU_FACTOR"] = self.plateau_factor_entry

        # ✅ Initially hide ReduceLROnPlateau parameters
        self.plateau_patience_label.setVisible(False)
        self.plateau_patience_entry.setVisible(False)
        self.plateau_factor_label.setVisible(False)
        self.plateau_factor_entry.setVisible(False)

        # ✅ Connect selection change event
        self.scheduler_combo.currentIndexChanged.connect(self.update_scheduler_settings)

        # **Apply Layout**
        layout.addWidget(scheduler_label)
        layout.addWidget(self.scheduler_combo)
        layout.addWidget(initial_lr_label)
        layout.addWidget(self.initial_lr_entry)
        layout.addWidget(self.lr_param_label)
        layout.addWidget(self.lr_param_entry)
        layout.addWidget(self.lr_period_label)
        layout.addWidget(self.lr_period_entry)
        layout.addWidget(self.plateau_patience_label)
        layout.addWidget(self.plateau_patience_entry)
        layout.addWidget(self.plateau_factor_label)
        layout.addWidget(self.plateau_factor_entry)

        # **Set Default to StepLR**
        self.update_scheduler_settings()

    def update_scheduler_settings(self):
        """Updates the displayed scheduler parameters dynamically."""
        selected_scheduler = self.param_entries["SCHEDULER_TYPE"].currentText()

        if selected_scheduler == "StepLR":
            self.lr_param_label.setText("LR Drop Factor:")
            self.lr_param_label.setVisible(True)
            self.lr_param_entry.setVisible(True)
            self.lr_period_label.setVisible(True)
            self.lr_period_entry.setVisible(True)

            self.plateau_patience_label.setVisible(False)
            self.plateau_patience_entry.setVisible(False)
            self.plateau_factor_label.setVisible(False)
            self.plateau_factor_entry.setVisible(False)

        elif selected_scheduler == "ReduceLROnPlateau":
            self.lr_param_label.setText("Plateau Factor:")
            self.lr_param_label.setVisible(True)
            self.lr_param_entry.setVisible(True)
            self.lr_period_label.setVisible(False)
            self.lr_period_entry.setVisible(False)

            self.plateau_patience_label.setVisible(True)
            self.plateau_patience_entry.setVisible(True)
            self.plateau_factor_label.setVisible(True)
            self.plateau_factor_entry.setVisible(True)


    def add_validation_criteria(self, layout):
        """Adds aligned validation patience and frequency UI components with tooltips and structured alignment."""

        # **Main Layout with Top Alignment**
        validation_layout = QVBoxLayout()
        validation_layout.setAlignment(Qt.AlignTop)  # Ensures content stays at the top

        # **Validation Patience**
        patience_label = QLabel("Validation Patience:")
        patience_label.setStyleSheet("font-size: 11pt; font-weight: bold;")
        patience_label.setToolTip("Number of epochs to wait for validation improvement before early stopping.")

        self.patience_entry = QLineEdit(self.params.get("VALID_PATIENCE", "10"))
        self.patience_entry.setFixedWidth(100)
        self.patience_entry.setToolTip("Enter an integer value (e.g., 10) to define early stopping patience.")

        # **Validation Frequency**
        freq_label = QLabel("Validation Frequency:")
        freq_label.setStyleSheet("font-size: 11pt; font-weight: bold;")
        freq_label.setToolTip("How often (in epochs) to perform validation during training.")

        self.freq_entry = QLineEdit(self.params.get("ValidFrequency", "3"))
        self.freq_entry.setFixedWidth(100)
        self.freq_entry.setToolTip("Enter an integer value (e.g., 3) to specify validation frequency.")

        # ✅ Store references in self.param_entries for parameter collection
        self.param_entries["VALID_PATIENCE"] = self.patience_entry
        self.param_entries["VALID_FREQUENCY"] = self.freq_entry

        # **Ensure Proper Alignment**
        patience_layout = QHBoxLayout()
        patience_layout.addWidget(patience_label)
        patience_layout.addWidget(self.patience_entry)
        patience_layout.setAlignment(Qt.AlignLeft)  # Aligns label and entry to the left

        freq_layout = QHBoxLayout()
        freq_layout.addWidget(freq_label)
        freq_layout.addWidget(self.freq_entry)
        freq_layout.setAlignment(Qt.AlignLeft)  # Aligns label and entry to the left

        # **Add Widgets to Layout in Vertical Order**
        validation_layout.addLayout(patience_layout)
        validation_layout.addLayout(freq_layout)

        # **Apply Layout to Parent Layout**
        layout.addLayout(validation_layout)


    def get_selected_features(self):
        """Retrieve selected feature columns as a list."""
        return [item.text() for item in self.feature_list.selectedItems()]



    def proceed_to_training(self):
        try:
            # Fetch all parameters
            new_params = {}

            # Collect values from stored param entries
            for param, entry in self.param_entries.items():
                if isinstance(entry, QLineEdit):
                    new_params[param] = entry.text().strip()  # Text input
                elif isinstance(entry, QComboBox):
                    new_params[param] = entry.currentText()  # Dropdown selection
                elif isinstance(entry, QCheckBox):
                    new_params[param] = entry.isChecked()  # Boolean value
                elif isinstance(entry, QListWidget):  # Multi-select feature list
                    new_params[param] = [item.text() for item in entry.selectedItems()]

            # Ensure critical fields are selected
            if not new_params.get("FEATURE_COLUMNS"):
                QMessageBox.critical(self, "Selection Error", "Please select at least one feature column.")
                return
            if not new_params.get("TARGET_COLUMN"):
                QMessageBox.critical(self, "Selection Error", "Please select a target column.")
                return
            if not new_params.get("MODEL_TYPE"):
                QMessageBox.critical(self, "Selection Error", "Please select a model type.")
                return

            self.logger.info(f"Proceeding to training with params: {new_params}")

            # Update the parameter manager
            self.update_params(new_params)

            # Save params only if job folder is set
            if self.job_manager.get_job_folder():
                self.hyper_param_manager.save_params()
            else:
                self.logger.error("Job folder is not set.")
                raise ValueError("Job folder is not set.")

            self.close()  # Close current window
            self.training_setup_gui = VEstimTrainSetupGUI(new_params)  # Pass updated params
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

    def load_column_names(self):
        """Loads column names from a sample CSV file in the train processed data folder."""
        train_folder = self.job_manager.get_train_folder()
        if train_folder:
            try:
                csv_files = [f for f in os.listdir(train_folder) if f.endswith(".csv")]
                if not csv_files:
                    raise FileNotFoundError("No CSV files found in train processed data folder.")
                
                sample_csv_path = os.path.join(train_folder, csv_files[0])  # Pick the first CSV
                df = pd.read_csv(sample_csv_path, nrows=1)  # Load only header
                return list(df.columns)  # Return column names

            except Exception as e:
                print(f"Error loading CSV columns: {e}")
        return []
    
    def load_params_from_json(self):
        """Load hyperparameters from a JSON file and update the UI."""
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Params", "", "JSON Files (*.json);;All Files (*)")
        if filepath:
            try:
                # Load and validate parameters using the manager
                self.params = self.hyper_param_manager.load_params(filepath)
                self.logger.info(f"Successfully loaded parameters from {filepath}")

                # Update GUI elements with loaded parameters
                self.update_gui_with_loaded_params()

            except Exception as e:
                self.logger.error(f"Failed to load parameters from {filepath}: {e}")
                QMessageBox.critical(self, "Error", f"Failed to load parameters: {str(e)}")


    def update_params(self, new_params):
        """Update hyperparameters and refresh the UI."""
        try:
            self.logger.info(f"Updating parameters: {new_params}")

            # Update using the hyperparameter manager
            self.hyper_param_manager.update_params(new_params)

            # Refresh the GUI with updated parameters
            self.update_gui_with_loaded_params()

        except ValueError as e:
            self.logger.error(f"Invalid parameter input: {new_params} - Error: {e}")
            QMessageBox.critical(self, "Error", f"Invalid parameter input: {str(e)}")


    def update_gui_with_loaded_params(self):
        """Update the GUI with previously saved parameters, including features & target."""
        
        if not self.params:
            self.logger.warning("No parameters found to update the GUI.")
            return

        # ✅ Update standard hyperparameters (Text Fields, Dropdowns, Checkboxes)
        for param_name, entry in self.param_entries.items():
            if param_name in self.params:
                value = self.params[param_name]

                # ✅ Handle different widget types correctly
                if isinstance(entry, QLineEdit):
                    entry.setText(str(value))  # Convert value to string for text fields

                elif isinstance(entry, QComboBox):
                    index = entry.findText(str(value))  # Get index for dropdowns
                    if index != -1:
                        entry.setCurrentIndex(index)

                elif isinstance(entry, QCheckBox):
                    entry.setChecked(bool(value))  # Ensure checkbox reflects state

                elif isinstance(entry, QListWidget):  # Multi-Select Feature List
                    selected_items = set(value) if isinstance(value, list) else set([value])
                    for i in range(entry.count()):
                        item = entry.item(i)
                        item.setSelected(item.text() in selected_items)

        # ✅ Update Model Parameters (Ensure proper model-specific param refresh)
        if "MODEL_TYPE" in self.params:
            model_index = self.model_combo.findText(self.params["MODEL_TYPE"])
            if model_index != -1:
                self.model_combo.setCurrentIndex(model_index)
                self.update_model_params()  # Refresh model-specific parameters

        # ✅ Ensure Scheduler Parameters Update Correctly
        if "SCHEDULER_TYPE" in self.params:
            self.update_scheduler_settings()  # Refresh visibility of LR scheduler params

        self.logger.info("GUI successfully updated with loaded parameters.")



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
