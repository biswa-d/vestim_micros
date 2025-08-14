# FIXED:---------------------------------------------------------------------------------
# FIXED:Author: Biswanath Dehury
# FIXED:Date: `{{date:YYYY-MM-DD}}`
# FIXED:Version: 1.0.0
# FIXED:Description: Description of the script
# FIXED:---------------------------------------------------------------------------------


import os
import json
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, 
    QLineEdit, QFileDialog, QMessageBox, QDialog, QGroupBox, QComboBox, QListWidget, QAbstractItemView,QFormLayout, QCheckBox, QScrollArea, QDesktopWidget
)
from PyQt5.QtCore import Qt, QPropertyAnimation
from PyQt5.QtGui import QIcon, QColor, QPalette

import pandas as pd
import torch

import logging

from vestim.gateway.src.job_manager_qt import JobManager
from vestim.gateway.src.hyper_param_manager_qt import VEstimHyperParamManager
from vestim.gui.src.training_setup_gui_qt import VEstimTrainSetupGUI
from vestim.config_manager import get_default_hyperparams, update_last_used_hyperparams, load_hyperparams_from_root

# FIXED:Initialize the JobManager
import logging
class VEstimHyperParamGUI(QWidget):
    def __init__(self, job_manager=None):
        self.logger = logging.getLogger(__name__)  # FIXED:Initialize the logger within the instance
        self.logger.info("Initializing Hyperparameter GUI")
        super().__init__()
        self.params = {}  # FIXED:Initialize an empty params dictionary
        self.job_manager = job_manager if job_manager else JobManager()
        self.hyper_param_manager = VEstimHyperParamManager(job_manager=self.job_manager)
        self.param_entries = {}  # FIXED:To store the entry widgets for parameters
        self.error_fields = set() # FIXED:To track fields with validation errors

        self.setup_window()
        self.build_gui()
        
        # FIXED:Load default hyperparameters after UI is built
        self.load_default_hyperparameters()

    def setup_window(self):
        """Initial setup for the main window appearance with responsive sizing."""
        self.setWindowTitle("VEstim - Hyperparameter Selection")
        
        # Get screen geometry for responsive sizing
        screen = QApplication.desktop().screenGeometry()
        screen_width = screen.width()
        screen_height = screen.height()
        
        # Calculate responsive window size (80% of screen, but with min/max limits)
        window_width = max(1000, min(1400, int(screen_width * 0.8)))
        window_height = max(700, min(900, int(screen_height * 0.8)))
        
        # Center the window on screen
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        self.setGeometry(x, y, window_width, window_height)
        
        # Set minimum size to prevent too much shrinking
        self.setMinimumSize(900, 600)
        
        # Enable DPI scaling
        self.setAttribute(Qt.WA_AcceptTouchEvents)
        
        # FIXED:Load the application icon
        resources_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources')
        icon_path = os.path.join(resources_path, 'icon.ico')
        
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        else:
            self.logger.warning("Icon file not found. Make sure 'icon.ico' is in the correct directory.")
        
        self.setStyleSheet("""
            QToolTip { font-weight: normal; font-size: 10pt; }
            QPushButton:disabled {
                background-color: #d3d3d3;
                color: #a9a9a9;
            }
            QScrollArea {
                border: none;
            }
        """)

    def load_default_hyperparameters(self):
        """Auto-load default hyperparameters and populate the GUI with column validation"""
        try:
            # FIXED:Get default hyperparameters from config (includes last used params with features/targets)
            default_params = get_default_hyperparams()
            
            # FIXED:Validate feature/target columns against current dataset
            validated_params = self.validate_columns_against_dataset(default_params)
            
            # FIXED:Load and validate parameters using the manager (same as load_params_from_json)
            self.params = self.hyper_param_manager.validate_and_normalize_params(validated_params)
            self.logger.info("Successfully loaded default hyperparameters automatically")

            # FIXED:Update GUI elements with loaded parameters (same as load_params_from_json)
            self.update_gui_with_loaded_params()
            
        except Exception as e:
            self.logger.error(f"Failed to auto-load default hyperparameters: {e}")
            # FIXED:If auto-load fails, just continue with empty params - user can manually load

    def validate_columns_against_dataset(self, params):
        """Validate that feature and target columns exist in the current dataset"""
        try:
            # FIXED:Get available columns from the current dataset
            available_columns = self.load_column_names()
            
            if not available_columns:
                self.logger.warning("No columns available in dataset, using parameters as-is")
                return params
            
            validated_params = params.copy()
            
            # FIXED:Validate feature columns
            if "FEATURE_COLUMNS" in params and params["FEATURE_COLUMNS"]:
                original_features = params["FEATURE_COLUMNS"]
                if isinstance(original_features, list):
                    # FIXED:Filter features to only include available columns
                    valid_features = [col for col in original_features if col in available_columns]
                    
                    if not valid_features:
                        # FIXED:No valid features, use first 3 available columns as fallback
                        valid_features = available_columns[:3] if len(available_columns) >= 3 else available_columns[:-1]
                        self.logger.warning(f"No saved feature columns found in dataset. Using fallback features: {valid_features}")
                    elif len(valid_features) < len(original_features):
                        missing_features = [col for col in original_features if col not in available_columns]
                        self.logger.info(f"Some saved feature columns not found in dataset. Missing: {missing_features}. Using available: {valid_features}")
                    
                    validated_params["FEATURE_COLUMNS"] = valid_features
                else:
                    # FIXED:Handle case where FEATURE_COLUMNS is not a list
                    validated_params["FEATURE_COLUMNS"] = available_columns[:3] if len(available_columns) >= 3 else available_columns[:-1]
                    self.logger.warning("Invalid feature columns format, using fallback features")
            
            # FIXED:Validate target column
            if "TARGET_COLUMN" in params and params["TARGET_COLUMN"]:
                original_target = params["TARGET_COLUMN"]
                if original_target not in available_columns:
                    # FIXED:Use last available column as fallback target
                    fallback_target = available_columns[-1] if available_columns else ""
                    validated_params["TARGET_COLUMN"] = fallback_target
                    self.logger.warning(f"Saved target column '{original_target}' not found in dataset. Using fallback target: '{fallback_target}'")
                else:
                    # Check if target is a timestamp column
                    valid_targets = self.filter_valid_target_columns([original_target])
                    if not valid_targets:
                        # Target is a timestamp column - find a safe alternative
                        safe_targets = self.filter_valid_target_columns(available_columns)
                        if safe_targets:
                            fallback_target = safe_targets[0]
                            validated_params["TARGET_COLUMN"] = fallback_target
                            self.logger.warning(f"Target column '{original_target}' appears to be a timestamp column. Using safe fallback target: '{fallback_target}'")
                        else:
                            self.logger.warning(f"Target column '{original_target}' appears to be a timestamp column, but no safe alternatives found.")
                    else:
                        self.logger.info(f"Target column '{original_target}' found in dataset and is valid")
            
            # FIXED:Validate training method compatibility with model type
            model_type = validated_params.get("MODEL_TYPE", "LSTM")
            training_method = validated_params.get("TRAINING_METHOD", "Sequence-to-Sequence")
            
            if model_type in ["LSTM", "GRU"] and training_method == "Whole Sequence":
                validated_params["TRAINING_METHOD"] = "Sequence-to-Sequence"
                self.logger.info(f"Converted training method from 'Whole Sequence' to 'Sequence-to-Sequence' for {model_type} model")
            elif model_type == "FNN" and training_method != "WholeSequenceFNN":
                # FIXED:For FNN, ensure we use the correct method name that data loader expects
                validated_params["TRAINING_METHOD"] = "WholeSequenceFNN"
                self.logger.info(f"Set training method to 'WholeSequenceFNN' for FNN model")
            
            return validated_params
            
        except Exception as e:
            self.logger.error(f"Error validating columns against dataset: {e}")
            return params  # FIXED:Return original params if validation fails

    def build_gui(self):
        """Build the main UI layout with categorized sections for parameters."""
        # Create main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create scroll area for the main content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Create content widget that will go inside the scroll area
        content_widget = QWidget()
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(10, 10, 10, 10)

        # FIXED:Title & Guide Section
        title_label = QLabel("Select Hyperparameters for Model Training")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #0b6337; margin-bottom: 15px;")
        content_layout.addWidget(title_label)

        guide_button = QPushButton("Open Hyperparameter Guide")
        guide_button.setFixedWidth(220)
        guide_button.setFixedHeight(30)
        guide_button.setStyleSheet("font-size: 10pt;")
        guide_button.clicked.connect(self.open_guide)
        guide_button_layout = QHBoxLayout()
        guide_button_layout.addStretch(1)
        guide_button_layout.addWidget(guide_button)
        guide_button_layout.addStretch(1)
        content_layout.addLayout(guide_button_layout)

        instructions_label = QLabel(
            "Please enter values for model parameters:\n"
            "• For Grid Search: Use comma-separated values (e.g., 1,2,5) or semicolons for multiple configs ([64,128];[32,64])\n"
            "• For Optuna Search: Use boundary format [min,max] for core hyperparameters (e.g., [1,5] for layers, [0.001,0.1] for learning rate)\n"
            "• Time and validation parameters (patience, frequency) can use single values for both methods\n"
            "Refer to the guide above for more details."
        )
        instructions_label.setAlignment(Qt.AlignCenter)
        instructions_label.setStyleSheet("font-size: 10pt; color: gray; margin-bottom: 10px;")
        instructions_label.setWordWrap(True)  # Allow text wrapping
        content_layout.addWidget(instructions_label)

        # FIXED:Hyperparameter Selection Section
        hyperparam_section = QGridLayout()
        hyperparam_section.setSpacing(10)  # Add spacing between grid items
        group_box_style = "QGroupBox { font-size: 10pt; font-weight: bold; }"

        # FIXED:--- Row 0 ---
        data_selection_group = QGroupBox("Data Selection")
        data_selection_group.setStyleSheet(group_box_style)
        data_selection_layout = QVBoxLayout()
        self.add_feature_target_selection(data_selection_layout)
        data_selection_group.setLayout(data_selection_layout)

        device_optimizer_group = QGroupBox("Device and Optimizer")
        device_optimizer_group.setStyleSheet(group_box_style)
        device_optimizer_layout = QVBoxLayout()
        self.add_device_selection(device_optimizer_layout)
        device_optimizer_group.setLayout(device_optimizer_layout)

        # FIXED:--- Row 1 (Left) & Row 2 (Left) ---
        model_training_group = QGroupBox("Model and Training Method")
        model_training_group.setStyleSheet(group_box_style)
        model_training_layout = QVBoxLayout() # FIXED:Changed to QVBoxLayout

        self.add_model_selection(model_training_layout)
        self.add_training_method_selection(model_training_layout)
        
        model_training_group.setLayout(model_training_layout)

        # FIXED:--- Row 1 (Right) ---
        validation_group = QGroupBox("Validation Training")
        validation_group.setStyleSheet(group_box_style)
        validation_criteria_layout = QVBoxLayout()
        self.add_validation_criteria(validation_criteria_layout)
        validation_group.setLayout(validation_criteria_layout)

        # FIXED:--- Row 2 (Right) ---
        lr_group = QGroupBox("Learning Rate Scheduler")
        lr_group.setStyleSheet(group_box_style)
        lr_layout = QHBoxLayout()

        scheduler_group = QGroupBox("LR Scheduler")
        scheduler_group.setStyleSheet("QGroupBox { font-size: 9pt; font-weight: bold; }")
        scheduler_layout = QVBoxLayout()
        self.add_scheduler_selection(scheduler_layout)
        scheduler_group.setLayout(scheduler_layout)

        exploit_lr_group = QGroupBox("Exploit LR")
        exploit_lr_group.setStyleSheet("QGroupBox { font-size: 9pt; font-weight: bold; }")
        exploit_lr_layout = QFormLayout()
        self.add_exploit_lr_widgets(exploit_lr_layout)
        exploit_lr_group.setLayout(exploit_lr_layout)

        lr_layout.addWidget(scheduler_group)
        lr_layout.addWidget(exploit_lr_group)
        lr_group.setLayout(lr_layout)

        # FIXED:Add widgets to grid
        hyperparam_section.addWidget(data_selection_group, 0, 0)
        hyperparam_section.addWidget(device_optimizer_group, 0, 1)
        hyperparam_section.addWidget(model_training_group, 1, 0, 2, 1)
        hyperparam_section.addWidget(validation_group, 1, 1)
        hyperparam_section.addWidget(lr_group, 2, 1)

        content_layout.addLayout(hyperparam_section)

        # FIXED:Bottom Buttons
        button_layout = QVBoxLayout()
        load_button = QPushButton("Load Params from File")
        load_button.setFixedWidth(220)
        load_button.setFixedHeight(30)
        load_button.setStyleSheet("font-size: 10pt;")
        load_button.clicked.connect(self.load_params_from_json)
        button_layout.addWidget(load_button, alignment=Qt.AlignCenter)

        search_method_layout = QHBoxLayout()
        search_method_layout.setAlignment(Qt.AlignCenter)
        
        auto_search_button = QPushButton("Auto Search (Optuna)")
        auto_search_button.setFixedWidth(180)
        auto_search_button.setFixedHeight(35)
        auto_search_button.setStyleSheet("background-color: #2E86AB; color: white; font-size: 10pt;")
        auto_search_button.setToolTip("Use Optuna for automatic hyperparameter optimization.\nRequires boundary format [min,max] for core hyperparameters (layers, hidden units, learning rate, epochs).\nTime and validation parameters can use single values.\nExample: [1,5] for layers, [0.001,0.1] for learning rate")
        auto_search_button.clicked.connect(self.proceed_to_auto_search)
        self.auto_search_button = auto_search_button
        
        grid_search_button = QPushButton("Exhaustive Grid Search")
        grid_search_button.setFixedWidth(180)
        grid_search_button.setFixedHeight(35)
        grid_search_button.setStyleSheet("background-color: #0b6337; color: white; font-size: 10pt;")
        grid_search_button.setToolTip("Use traditional exhaustive grid search.\nRequires comma-separated values: 1,2,5 or semicolon for multiple configs: [64,128];[32,64]")
        grid_search_button.clicked.connect(self.proceed_to_grid_search)
        
        search_method_layout.addWidget(auto_search_button)
        search_method_layout.addWidget(grid_search_button)
        button_layout.addLayout(search_method_layout)

        content_layout.addLayout(button_layout)
        
        # Set up the scroll area
        content_widget.setLayout(content_layout)
        scroll_area.setWidget(content_widget)
        
        # Add scroll area to main layout
        main_layout.addWidget(scroll_area)
        self.setLayout(main_layout)

    def add_feature_target_selection(self, layout):
        """Adds a vertically stacked feature and target selection UI components with tooltips."""

        column_names = self.load_column_names()
        
        # Filter out timestamp/time-related columns for target selection
        valid_target_columns = self.filter_valid_target_columns(column_names)

        # FIXED:**Feature Selection**
        feature_label = QLabel("Feature Columns (Input):")
        feature_label.setStyleSheet("font-size: 9pt;")
        feature_label.setToolTip("Select one or more columns as input features for training.")

        self.feature_list = QListWidget()
        self.feature_list.addItems(column_names)  # Features can include all columns
        self.feature_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.feature_list.setFixedHeight(100)
        self.feature_list.setToolTip("Select multiple features.")

        # FIXED:**Target Selection**
        target_label = QLabel("Target Column (Output):")
        target_label.setStyleSheet("font-size: 9pt;")
        target_label.setToolTip("<html><body><span style='font-weight: normal;'>Select the output column for the model to predict.<br><b>Note:</b> Timestamp/time columns are filtered out.</span></body></html>")

        self.target_combo = QComboBox()
        self.target_combo.addItems(valid_target_columns)  # Only valid targets
        self.target_combo.setToolTip("Select a single target column (timestamp columns excluded).")

        # FIXED:✅ Store references in self.param_entries for easy parameter collection
        self.param_entries["FEATURE_COLUMNS"] = self.feature_list
        self.param_entries["TARGET_COLUMN"] = self.target_combo

        # FIXED:**Form Layout for Alignment**
        form_layout = QFormLayout()
        form_layout.addRow(feature_label, self.feature_list)
        form_layout.addRow(target_label, self.target_combo)

        # FIXED:**Apply to Parent Layout**
        layout.addLayout(form_layout)

    def add_training_method_selection(self, layout):
        """Adds training method selection with batch size, train-validation split, tooltips, and ensures UI alignment."""

        # FIXED:**Main Layout with Top Alignment**
        training_layout = QVBoxLayout()
        training_layout.setAlignment(Qt.AlignTop)  # FIXED:Ensures content stays at the top

        # FIXED:**Training Method Selection Dropdown**
        self.training_method_label = QLabel("Training Method:")
        self.training_method_label.setStyleSheet("font-size: 9pt;")
        self.training_method_label.setToolTip("Choose how training data is processed.")

        self.training_method_combo = QComboBox()
        training_options = ["Sequence-to-Sequence", "Whole Sequence"]
        self.training_method_combo.addItems(training_options)
        self.training_method_combo.setToolTip(
            "Sequence-to-Sequence: Processes data in fixed time steps.\n"
            "Whole Sequence: Uses the entire sequence for training."
        )

        # FIXED:**Lookback Parameter (Only for Sequence-to-Sequence)**
        self.lookback_label = QLabel("Lookback Window:")
        self.lookback_label.setStyleSheet("font-size: 9pt;")
        self.lookback_label.setToolTip("Defines how many previous time steps are used for each prediction.")
        self.lookback_entry = QLineEdit(self.params.get("LOOKBACK", "400"))
        self.lookback_entry.textChanged.connect(self.on_param_text_changed)

        # FIXED:**Batch Training Option (Checkbox)**
        self.batch_training_checkbox = QCheckBox("Enable Batch Training")
        self.batch_training_checkbox.setChecked(True)  # FIXED:Default is now checked
        self.batch_training_checkbox.setToolTip("Enable mini-batch training. This is required for FNN and recommended for sequence-based methods.")
        self.batch_training_checkbox.stateChanged.connect(self.update_batch_size_visibility)

        # FIXED:**Batch Size Entry (Initially Enabled as checkbox is checked by default)**
        self.batch_size_label = QLabel("Batch Size:") # FIXED:Made it an instance variable to hide/show
        self.batch_size_label.setStyleSheet("font-size: 9pt;")
        self.batch_size_label.setToolTip("Number of samples per batch.")
        self.batch_size_entry = QLineEdit(self.params.get("BATCH_SIZE", "100")) # FIXED:Default value 100
        self.batch_size_entry.textChanged.connect(self.on_param_text_changed)
        self.batch_size_entry.setEnabled(True)  # FIXED:Initially enabled

        # FIXED:✅ Store references in self.param_entries for easy parameter collection
        self.param_entries["TRAINING_METHOD"] = self.training_method_combo
        self.param_entries["LOOKBACK"] = self.lookback_entry
        self.param_entries["BATCH_TRAINING"] = self.batch_training_checkbox
        self.param_entries["BATCH_SIZE"] = self.batch_size_entry

        # FIXED:Initially hide lookback if Whole Sequence is selected
        self.lookback_label.setVisible(self.training_method_combo.currentText() == "Sequence-to-Sequence")
        self.lookback_entry.setVisible(self.training_method_combo.currentText() == "Sequence-to-Sequence")

        # FIXED:**Update Visibility Based on Selection**
        self.training_method_combo.currentIndexChanged.connect(self.update_training_method)

        # FIXED:**Add Widgets to Layout in Vertical Order**
        training_layout.addWidget(self.training_method_label)
        training_layout.addWidget(self.training_method_combo)
        training_layout.addWidget(self.lookback_label)
        training_layout.addWidget(self.lookback_entry)
        training_layout.addWidget(self.batch_training_checkbox)
        training_layout.addWidget(self.batch_size_label) # FIXED:Use instance variable
        training_layout.addWidget(self.batch_size_entry)

        # FIXED:**Apply Layout to Parent Layout**
        layout.addLayout(training_layout)


    def update_training_method(self):
        """Toggle lookback, batch training checkbox, and batch size visibility based on training method and model type."""
        current_training_method = self.training_method_combo.currentText()
        current_model_type = self.model_combo.currentText() # FIXED:Assuming self.model_combo exists and is accessible

        is_rnn_model = current_model_type in ["LSTM", "GRU"]
        is_fnn_model = current_model_type == "FNN"
        is_whole_sequence_rnn = (current_training_method == "Whole Sequence" and is_rnn_model)
        is_sequence_to_sequence = (current_training_method == "Sequence-to-Sequence")

        # FIXED:For FNN, hide sequence-related options
        if is_fnn_model:
            self.lookback_label.setVisible(False)
            self.lookback_entry.setVisible(False)
            self.training_method_label.setVisible(False)
            self.training_method_combo.setVisible(False)
            
            # FIXED:FNN requires batch training
            self.batch_training_checkbox.setChecked(True)
            self.batch_training_checkbox.setEnabled(True)
            if self.batch_size_entry.text().strip() in ["", "100"]:
                self.batch_size_entry.setText("5000")
        
        # FIXED:For RNN models, show sequence-related options
        else:
            self.training_method_label.setVisible(True)
            self.training_method_combo.setVisible(True)
            self.lookback_label.setVisible(is_sequence_to_sequence)
            self.lookback_entry.setVisible(is_sequence_to_sequence)
            
            # FIXED:Allow user to toggle batch training for RNN
            self.batch_training_checkbox.setEnabled(True)
            if self.batch_size_entry.text().strip() == "5000":
                self.batch_size_entry.setText("100")

        # FIXED:Update batch size visibility based on checkbox state
        self.update_batch_size_visibility()

    def update_batch_size_visibility(self):
        """Enable or disable batch size input based on batch training checkbox, only if checkbox is visible."""
        if self.batch_training_checkbox.isVisible():
            is_checked = self.batch_training_checkbox.isChecked()
            is_checked = self.batch_training_checkbox.isChecked()
            self.batch_size_entry.setEnabled(is_checked)
            self.batch_size_label.setVisible(True)
            self.batch_size_entry.setVisible(True)


    def add_model_selection(self, layout):
        """Adds aligned model selection UI components with top alignment and tooltips."""

        # FIXED:**Main Vertical Layout (Ensures Content Starts at the Top)**
        model_layout = QVBoxLayout()
        model_layout.setAlignment(Qt.AlignTop)  # FIXED:Ensures text stays at the top

        # FIXED:**Model Selection Dropdown**
        model_label = QLabel("Select Model:")
        model_label.setStyleSheet("font-size: 9pt;")
        model_label.setToolTip("Select a model architecture for training.")

        self.model_combo = QComboBox()
        model_options = ["LSTM", "FNN", "GRU"]
        self.model_combo.addItems(model_options)
        self.model_combo.setToolTip("LSTM for time-series, CNN for feature extraction, GRU for memory-efficient training, Transformer for advanced architectures.")

        # FIXED:**Model-Specific Parameters Placeholder**
        self.model_param_container = QVBoxLayout()

        # FIXED:**Store reference in param_entries**
        self.param_entries["MODEL_TYPE"] = self.model_combo

        # FIXED:Connect Dropdown to Update Parameters
        self.model_combo.currentIndexChanged.connect(self.update_model_params)
        self.model_combo.currentIndexChanged.connect(self.update_training_method) # FIXED:Also trigger training method updates

        # FIXED:**Add Widgets in Order**
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        model_layout.addLayout(self.model_param_container)  # FIXED:Placeholder for model-specific params

        # FIXED:**Apply Layout to Parent Layout**
        layout.addLayout(model_layout)

        # FIXED:**Set Default to LSTM and Populate Parameters**
        self.model_combo.setCurrentText("LSTM")  # FIXED:Ensure LSTM is selected
        self.update_model_params()  # FIXED:Populate default parameters


    def update_model_params(self):
        """Dynamically updates parameter fields based on selected model and stores them in param_entries."""
        selected_model = self.model_combo.currentText()

        # FIXED:--- Clear previous model-specific QLineEdit entries from self.param_entries ---
        # FIXED:Define keys for model-specific parameters that might exist from a previous selection
        lstm_specific_keys = ["LAYERS", "HIDDEN_UNITS"] # FIXED:Add any other LSTM specific QLineEdit keys
        gru_specific_keys = ["GRU_LAYERS", "GRU_HIDDEN_UNITS"] # FIXED:Add any other GRU specific QLineEdit keys
        fnn_specific_keys = ["FNN_HIDDEN_LAYERS", "FNN_DROPOUT_PROB"] # FIXED:Add FNN specific QLineEdit keys
        
        all_model_specific_keys = lstm_specific_keys + gru_specific_keys + fnn_specific_keys
        
        for key_to_remove in all_model_specific_keys:
            if key_to_remove in self.param_entries:
                # FIXED:We don't delete the widget here as it's handled by clearing model_param_container
                # FIXED:Just remove the reference from param_entries
                del self.param_entries[key_to_remove]
        # FIXED:--- End clearing stale entries ---

        # FIXED:**Clear only dynamic parameter widgets (Keep Label & Dropdown)**
        while self.model_param_container.count():
            item = self.model_param_container.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        # FIXED:**Ensure Param Entries Are Tracked**
        model_params = {} # FIXED:This will hold QLineEdit widgets for the current model

        # FIXED:**Model-Specific Parameters**
        if selected_model == "LSTM" or selected_model == "":
            lstm_layers_label = QLabel("LSTM Layers:")
            lstm_layers_label.setStyleSheet("font-size: 9pt;")
            lstm_layers_label.setToolTip("Number of LSTM layers in the model.")

            self.lstm_layers_entry = QLineEdit(self.params.get("LAYERS", "1"))
            self.lstm_layers_entry.setToolTip("Enter the number of stacked LSTM layers.\nGrid Search: 1,2,3 | Optuna: [1,5]")
            self.lstm_layers_entry.textChanged.connect(self.on_param_text_changed)

            hidden_units_label = QLabel("Hidden Units:")
            hidden_units_label.setStyleSheet("font-size: 9pt;")
            hidden_units_label.setToolTip("Number of hidden units per layer.")

            self.hidden_units_entry = QLineEdit(self.params.get("HIDDEN_UNITS", "10"))
            self.hidden_units_entry.setToolTip("Enter the number of hidden units per LSTM layer.\nGrid Search: 10,20,50 | Optuna: [10,100]")
            self.hidden_units_entry.textChanged.connect(self.on_param_text_changed)

            self.model_param_container.addWidget(lstm_layers_label)
            self.model_param_container.addWidget(self.lstm_layers_entry)
            self.model_param_container.addWidget(hidden_units_label)
            self.model_param_container.addWidget(self.hidden_units_entry)

            # FIXED:✅ Store in param_entries
            model_params["LAYERS"] = self.lstm_layers_entry
            model_params["HIDDEN_UNITS"] = self.hidden_units_entry

        # FIXED:**GRU Parameters**
        elif selected_model == "GRU":
            gru_layers_label = QLabel("GRU Layers:")
            gru_layers_label.setStyleSheet("font-size: 9pt;")
            gru_layers_label.setToolTip("Number of GRU layers in the model.")

            self.gru_layers_entry = QLineEdit(self.params.get("GRU_LAYERS", "1"))
            self.gru_layers_entry.setToolTip("Enter the number of stacked GRU layers.")
            self.gru_layers_entry.textChanged.connect(self.on_param_text_changed)

            gru_hidden_units_label = QLabel("GRU Hidden Units:")
            gru_hidden_units_label.setStyleSheet("font-size: 9pt;")
            gru_hidden_units_label.setToolTip("Number of hidden units per GRU layer.")

            self.gru_hidden_units_entry = QLineEdit(self.params.get("GRU_HIDDEN_UNITS", "10"))
            self.gru_hidden_units_entry.setToolTip("Enter the number of hidden units per GRU layer.")
            self.gru_hidden_units_entry.textChanged.connect(self.on_param_text_changed)

            self.model_param_container.addWidget(gru_layers_label)
            self.model_param_container.addWidget(self.gru_layers_entry)
            self.model_param_container.addWidget(gru_hidden_units_label)
            self.model_param_container.addWidget(self.gru_hidden_units_entry)

            # FIXED:✅ Store in param_entries
            model_params["GRU_LAYERS"] = self.gru_layers_entry
            model_params["GRU_HIDDEN_UNITS"] = self.gru_hidden_units_entry

        elif selected_model == "FNN":
            fnn_hidden_layers_label = QLabel("FNN Hidden Layers:")
            fnn_hidden_layers_label.setStyleSheet("font-size: 9pt;")
            fnn_hidden_layers_label.setToolTip("Define FNN hidden layer sizes. Single config: 128,64,32. Multiple configs: use semicolons (128,64;100,50,25) or brackets ([128,64,32], [100,50,25]).")
            self.fnn_hidden_layers_entry = QLineEdit(self.params.get("FNN_HIDDEN_LAYERS", "128,64"))
            self.fnn_hidden_layers_entry.setToolTip("Single: '128,64,32' | Multiple with semicolons: '128,64;100,50,25' | Multiple with brackets: '[128,64,32], [100,50,25]'")
            self.fnn_hidden_layers_entry.textChanged.connect(self.on_param_text_changed)
            
            fnn_dropout_label = QLabel("FNN Dropout Prob:")
            fnn_dropout_label.setStyleSheet("font-size: 9pt;")
            fnn_dropout_label.setToolTip("Dropout probability for FNN layers (0.0 to 1.0).")
            self.fnn_dropout_entry = QLineEdit(self.params.get("FNN_DROPOUT_PROB", "0.1"))
            self.fnn_dropout_entry.setToolTip("e.g., 0.1 for 10% dropout")
            self.fnn_dropout_entry.textChanged.connect(self.on_param_text_changed)

            self.model_param_container.addWidget(fnn_hidden_layers_label)
            self.model_param_container.addWidget(self.fnn_hidden_layers_entry)
            self.model_param_container.addWidget(fnn_dropout_label)
            self.model_param_container.addWidget(self.fnn_dropout_entry)

            # FIXED:✅ Store in model_params for later update to self.param_entries
            model_params["FNN_HIDDEN_LAYERS"] = self.fnn_hidden_layers_entry
            model_params["FNN_DROPOUT_PROB"] = self.fnn_dropout_entry

        # FIXED:✅ Register current model-specific QLineEdit parameters in self.param_entries
        # FIXED:This ensures self.param_entries only contains widgets relevant to the *current* model type
        self.param_entries.update(model_params)



    def add_exploit_lr_widgets(self, layout):
        """Adds exploit LR widgets to the given QFormLayout."""
        # FIXED:Add Exploit LR QLineEdit
        exploit_lr_label = QLabel("Exploit LR:")
        exploit_lr_label.setStyleSheet("font-size: 9pt;")
        exploit_lr_label.setToolTip("Learning rate to use after patience is reached and the best model is reloaded.")
        self.exploit_lr_entry = QLineEdit(self.params.get("EXPLOIT_LR", "1e-5"))
        self.exploit_lr_entry.setToolTip("After patience is exhausted, the best model is reloaded and trained with this learning rate.")
        self.exploit_lr_entry.textChanged.connect(self.on_param_text_changed)
        self.param_entries["EXPLOIT_LR"] = self.exploit_lr_entry

        # FIXED:Add Exploit Patience QLineEdit
        exploit_epochs_label = QLabel("Exploit Epochs:")
        exploit_epochs_label.setStyleSheet("font-size: 9pt;")
        exploit_epochs_label.setToolTip("Number of epochs for the Cosine Annealing exploit phase.")
        self.exploit_epochs_entry = QLineEdit(self.params.get("EXPLOIT_EPOCHS", "5"))
        self.exploit_epochs_entry.setToolTip("Number of epochs to train from the best state using a Cosine Annealing schedule.")
        self.exploit_epochs_entry.textChanged.connect(self.on_param_text_changed)
        self.param_entries["EXPLOIT_EPOCHS"] = self.exploit_epochs_entry

        # FIXED:Add Exploit Factor QLineEdit
        exploit_repetitions_label = QLabel("Exploit Repetitions:")
        exploit_repetitions_label.setStyleSheet("font-size: 9pt;")
        exploit_repetitions_label.setToolTip("Number of times to repeat the exploit phase.")
        self.exploit_repetitions_entry = QLineEdit(self.params.get("EXPLOIT_REPETITIONS", "1"))
        self.exploit_repetitions_entry.setToolTip("Number of times to repeat the exploit phase if no new best model is found.")
        self.exploit_repetitions_entry.textChanged.connect(self.on_param_text_changed)
        self.param_entries["EXPLOIT_REPETITIONS"] = self.exploit_repetitions_entry

        final_lr_label = QLabel("Final LR:")
        final_lr_label.setStyleSheet("font-size: 9pt;")
        final_lr_label.setToolTip("The final learning rate for the Cosine Annealing scheduler.")
        self.final_lr_entry = QLineEdit(self.params.get("FINAL_LR", "1e-7"))
        self.final_lr_entry.setToolTip("The minimum learning rate at the end of the Cosine Annealing cycle.")
        self.final_lr_entry.textChanged.connect(self.on_param_text_changed)
        self.param_entries["FINAL_LR"] = self.final_lr_entry

        layout.addRow(exploit_lr_label, self.exploit_lr_entry)
        layout.addRow(final_lr_label, self.final_lr_entry)
        layout.addRow(exploit_epochs_label, self.exploit_epochs_entry)
        layout.addRow(exploit_repetitions_label, self.exploit_repetitions_entry)

    def add_scheduler_selection(self, layout):
        """Adds learning rate scheduler selection UI components with dynamic label updates and Initial LR."""

        # FIXED:**Scheduler Selection**
        scheduler_label = QLabel("LR Scheduler:")
        scheduler_label.setStyleSheet("font-size: 9pt;")
        scheduler_label.setToolTip("Select a scheduler to adjust the learning rate during training.")

        self.scheduler_combo = QComboBox()
        scheduler_options = ["StepLR", "ReduceLROnPlateau"]
        self.scheduler_combo.addItems(scheduler_options)
        self.scheduler_combo.setToolTip(
            "StepLR: Reduces LR at fixed intervals.\n"
            "ReduceLROnPlateau: Reduces LR when training stagnates."
        )
        self.param_entries["SCHEDULER_TYPE"] = self.scheduler_combo

        # FIXED:**Initial Learning Rate (Common Parameter)**
        initial_lr_label = QLabel("Initial LR:")
        initial_lr_label.setStyleSheet("font-size: 9pt;")
        initial_lr_label.setToolTip("The starting learning rate for the optimizer.")
        self.initial_lr_entry = QLineEdit(self.params.get("INITIAL_LR", "0.0001"))
        self.initial_lr_entry.setToolTip("Lower values may stabilize training but slow convergence.")
        self.initial_lr_entry.textChanged.connect(self.on_param_text_changed)
        self.param_entries["INITIAL_LR"] = self.initial_lr_entry

        # FIXED:**StepLR Parameters**
        self.lr_param_label = QLabel("LR Drop Factor:")  # FIXED:Dynamic label
        self.lr_param_label.setStyleSheet("font-size: 9pt;")
        self.lr_param_label.setToolTip("Factor by which LR is reduced (e.g., 0.1 means LR reduces by 10%).")
        self.lr_param_entry = QLineEdit(self.params.get("LR_DROP_FACTOR", "0.1"))
        self.lr_param_entry.setToolTip("Lower values reduce LR more aggressively.")
        self.lr_param_entry.textChanged.connect(self.on_param_text_changed)
        self.param_entries["LR_PARAM"] = self.lr_param_entry

        self.lr_period_label = QLabel("LR Drop Period:")
        self.lr_period_label.setStyleSheet("font-size: 9pt;")
        self.lr_period_label.setToolTip("Number of epochs after which LR is reduced.")
        self.lr_period_entry = QLineEdit(self.params.get("LR_DROP_PERIOD", "10"))
        self.lr_period_entry.setToolTip("Set higher values if you want the LR to stay stable for longer periods.")
        self.lr_period_entry.textChanged.connect(self.on_param_text_changed)
        self.param_entries["LR_PERIOD"] = self.lr_period_entry

        # FIXED:**ReduceLROnPlateau Parameters**
        self.plateau_patience_label = QLabel("Plateau Patience:")
        self.plateau_patience_label.setStyleSheet("font-size: 9pt;")
        self.plateau_patience_label.setToolTip("Number of epochs to wait before reducing LR if no improvement in validation.")
        self.plateau_patience_entry = QLineEdit(self.params.get("PLATEAU_PATIENCE", "10"))
        self.plateau_patience_entry.setToolTip("Larger values allow longer training before LR adjustment.")
        self.plateau_patience_entry.textChanged.connect(self.on_param_text_changed)
        self.param_entries["PLATEAU_PATIENCE"] = self.plateau_patience_entry

        self.plateau_factor_label = QLabel("Plateau Factor:")
        self.plateau_factor_label.setStyleSheet("font-size: 9pt;")
        self.plateau_factor_label.setToolTip("Factor by which LR is reduced when ReduceLROnPlateau is triggered.")
        self.plateau_factor_entry = QLineEdit(self.params.get("PLATEAU_FACTOR", "0.1"))
        self.plateau_factor_entry.setToolTip("Lower values make the LR decrease more significantly.")
        self.plateau_factor_entry.textChanged.connect(self.on_param_text_changed)
        self.param_entries["PLATEAU_FACTOR"] = self.plateau_factor_entry

        # FIXED:Initially hide all scheduler-specific parameters
        self.plateau_patience_label.setVisible(False)
        self.plateau_patience_entry.setVisible(False)
        self.plateau_factor_label.setVisible(False)
        self.plateau_factor_entry.setVisible(False)

        # FIXED:Connect selection change event
        self.scheduler_combo.currentIndexChanged.connect(self.update_scheduler_settings)

        # FIXED:Use QFormLayout for neat alignment
        form_layout = QFormLayout()
        form_layout.addRow(scheduler_label, self.scheduler_combo)
        form_layout.addRow(initial_lr_label, self.initial_lr_entry)
        form_layout.addRow(self.lr_period_label, self.lr_period_entry)
        form_layout.addRow(self.lr_param_label, self.lr_param_entry)
        form_layout.addRow(self.plateau_patience_label, self.plateau_patience_entry)
        form_layout.addRow(self.plateau_factor_label, self.plateau_factor_entry)
        
        layout.addLayout(form_layout)

        # FIXED:Set Default to StepLR
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

        # FIXED:**Main Layout with Top Alignment**
        validation_layout = QVBoxLayout()
        validation_layout.setAlignment(Qt.AlignTop)

        validation_form_layout = QFormLayout()

        # FIXED:Add maximum training epochs
        max_epochs_label = QLabel("Max Training Epochs:")
        max_epochs_label.setStyleSheet("font-size: 9pt;")
        max_epochs_label.setToolTip("Enter maximum training epochs. Use commas for multiple values (e.g., 100,200,500)")
        self.max_epochs_entry = QLineEdit(self.params.get("MAX_EPOCHS", "500"))
        self.max_epochs_entry.setToolTip("Enter maximum training epochs.\nGrid Search: 100,200,500 | Optuna: [100,1000]")
        self.max_epochs_entry.textChanged.connect(self.on_param_text_changed)
        validation_form_layout.addRow(max_epochs_label, self.max_epochs_entry)

        # FIXED:**Validation Patience**
        patience_label = QLabel("Validation Patience:")
        patience_label.setStyleSheet("font-size: 9pt;")
        patience_label.setToolTip("Enter validation patience. Use commas for multiple values (e.g., 5,10,15)")
        self.patience_entry = QLineEdit(self.params.get("VALID_PATIENCE", "10"))
        self.patience_entry.setToolTip("Enter validation patience. Use commas for multiple values (e.g., 5,10,15)")
        self.patience_entry.textChanged.connect(self.on_param_text_changed)
        validation_form_layout.addRow(patience_label, self.patience_entry)

        # FIXED:**Validation Frequency**
        freq_label = QLabel("Validation Frequency:")
        freq_label.setStyleSheet("font-size: 9pt;")
        freq_label.setToolTip("Enter validation frequency. Use commas for multiple values (e.g., 1,3,5)")
        self.freq_entry = QLineEdit(self.params.get("ValidFrequency", "1"))
        self.freq_entry.setToolTip("Enter validation frequency. Use commas for multiple values (e.g., 1,3,5)")
        self.freq_entry.textChanged.connect(self.on_param_text_changed)
        validation_form_layout.addRow(freq_label, self.freq_entry)

        # FIXED:Add Repetitions QLineEdit
        repetitions_label = QLabel("Repetitions:")
        repetitions_label.setStyleSheet("font-size: 9pt;")
        repetitions_label.setToolTip("Number of times to repeat each training task with the same hyperparameters.")
        self.repetitions_entry = QLineEdit(str(self.params.get("REPETITIONS", "1"))) # FIXED:Default to "1"
        self.repetitions_entry.setToolTip("Enter an integer (e.g., 1, 2, 3).")
        self.repetitions_entry.textChanged.connect(self.on_param_text_changed)
        validation_form_layout.addRow(repetitions_label, self.repetitions_entry)

        # FIXED:✅ Store references in self.param_entries for parameter collection
        self.param_entries["VALID_PATIENCE"] = self.patience_entry
        self.param_entries["VALID_FREQUENCY"] = self.freq_entry
        self.param_entries["MAX_EPOCHS"] = self.max_epochs_entry
        self.param_entries["REPETITIONS"] = self.repetitions_entry # FIXED:Add to param_entries

        # FIXED:**Max Training Time**
        max_time_label = QLabel("Max Training Time:")
        max_time_label.setStyleSheet("font-size: 9pt;")
        max_time_label.setToolTip("Set a maximum duration for the training process (HH:MM:SS).")
        
        time_layout = QHBoxLayout()
        self.max_time_hours_entry = QLineEdit(self.params.get("MAX_TRAIN_HOURS", "0"))
        self.max_time_hours_entry.setPlaceholderText("HH")
        self.max_time_hours_entry.textChanged.connect(self.on_param_text_changed)
        time_layout.addWidget(self.max_time_hours_entry)
        time_layout.addWidget(QLabel("H :"))
        
        self.max_time_minutes_entry = QLineEdit(self.params.get("MAX_TRAIN_MINUTES", "0"))
        self.max_time_minutes_entry.setPlaceholderText("MM")
        self.max_time_minutes_entry.textChanged.connect(self.on_param_text_changed)
        time_layout.addWidget(self.max_time_minutes_entry)
        time_layout.addWidget(QLabel("M :"))

        self.max_time_seconds_entry = QLineEdit(self.params.get("MAX_TRAIN_SECONDS", "0"))
        self.max_time_seconds_entry.setPlaceholderText("SS")
        self.max_time_seconds_entry.textChanged.connect(self.on_param_text_changed)
        time_layout.addWidget(self.max_time_seconds_entry)
        time_layout.addWidget(QLabel("S"))
        time_layout.addStretch()
        validation_form_layout.addRow(max_time_label, time_layout)

        self.param_entries["MAX_TRAIN_HOURS"] = self.max_time_hours_entry
        self.param_entries["MAX_TRAIN_MINUTES"] = self.max_time_minutes_entry
        self.param_entries["MAX_TRAIN_SECONDS"] = self.max_time_seconds_entry

        layout.addLayout(validation_form_layout)
        
    def add_device_selection(self, layout):
        """Adds device selection UI components."""
        form_layout = QFormLayout()

        device_label = QLabel("Device Selection:")
        device_label.setStyleSheet("font-size: 9pt;")
        device_label.setToolTip("Select the device for training (CPU or specific CUDA GPU).")
        
        self.device_combo = QComboBox()
        device_options = ["CPU"]
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device_options.append(f"cuda:{i}")
        self.device_combo.addItems(device_options)
        self.device_combo.setToolTip("Select CPU for compatibility, GPU for faster training.")
        default_device = "cuda:0" if torch.cuda.is_available() else "CPU"
        if default_device in device_options:
            self.device_combo.setCurrentText(default_device)
        elif "CPU" in device_options:
            self.device_combo.setCurrentText("CPU")
        self.param_entries["DEVICE_SELECTION"] = self.device_combo
        form_layout.addRow(device_label, self.device_combo)

        if torch.cuda.is_available():
            self.mixed_precision_checkbox = QCheckBox("Use Mixed Precision Training")
            self.mixed_precision_checkbox.setChecked(True)
            self.mixed_precision_checkbox.setToolTip("Enable automatic mixed precision (AMP) to accelerate GPU training.")
            self.param_entries["USE_MIXED_PRECISION"] = self.mixed_precision_checkbox
            form_layout.addRow(self.mixed_precision_checkbox)

        optimizer_label = QLabel("Optimizer:")
        optimizer_label.setStyleSheet("font-size: 9pt;")
        optimizer_label.setToolTip("Select one or more optimization algorithms for grid search.")
        self.optimizer_list = QListWidget()
        self.optimizer_list.addItems(["Adam", "SGD", "RMSprop"])
        self.optimizer_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.optimizer_list.setFixedHeight(60)
        self.param_entries["OPTIMIZER_TYPE"] = self.optimizer_list
        form_layout.addRow(optimizer_label, self.optimizer_list)

        weight_decay_label = QLabel("Weight Decay:")
        weight_decay_label.setStyleSheet("font-size: 9pt;")
        weight_decay_label.setToolTip("L2 penalty (regularization term). Helps prevent overfitting.")
        self.weight_decay_entry = QLineEdit(self.params.get("WEIGHT_DECAY", "0.0"))
        self.weight_decay_entry.setToolTip("Enter a float value (e.g., 0.01).")
        self.weight_decay_entry.textChanged.connect(self.on_param_text_changed)
        self.param_entries["WEIGHT_DECAY"] = self.weight_decay_entry
        form_layout.addRow(weight_decay_label, self.weight_decay_entry)

        # FIXED:Data Loading Optimization Section
        data_loading_label = QLabel("Data Loading:")
        data_loading_label.setStyleSheet("font-size: 9pt; font-weight: bold; color: #2E86AB; margin-top: 10px;")
        form_layout.addRow(data_loading_label)

        # FIXED:NUM_WORKERS with user-friendly label
        cpu_threads_label = QLabel("# FIXED:CPU Threads:")
        cpu_threads_label.setStyleSheet("font-size: 9pt;")
        cpu_threads_label.setToolTip(
            "Number of CPU processes for data loading (improves GPU utilization).\n"
            "Recommended: 4-8 for most systems.\n"
            "Higher values = faster data loading but more CPU RAM usage.\n"
            "Your system (32GB RAM): Can safely use 6-8 threads."
        )
        import os
        cpu_cores = os.cpu_count()
        default_workers = min(6, cpu_cores) if cpu_cores else 4
        self.num_workers_entry = QLineEdit(self.params.get("NUM_WORKERS", str(default_workers)))
        self.num_workers_entry.setToolTip(
            f"Recommended for your system: {default_workers} threads\n"
            f"Your CPU cores: {cpu_cores}\n"
            "Higher values improve GPU utilization but use more CPU RAM.\n"
            "Set to 0 for single-threaded loading (uses less RAM)."
        )
        self.num_workers_entry.textChanged.connect(self.on_param_text_changed)
        self.param_entries["NUM_WORKERS"] = self.num_workers_entry
        form_layout.addRow(cpu_threads_label, self.num_workers_entry)

        # FIXED:PIN_MEMORY checkbox
        self.pin_memory_checkbox = QCheckBox("Fast CPU-GPU Transfer")
        self.pin_memory_checkbox.setChecked(self.params.get("PIN_MEMORY", True))
        self.pin_memory_checkbox.setToolTip("Enable pinned memory for faster CPU-GPU data transfers (recommended)")
        self.param_entries["PIN_MEMORY"] = self.pin_memory_checkbox
        form_layout.addRow(self.pin_memory_checkbox)

        # FIXED:PREFETCH_FACTOR
        prefetch_label = QLabel("Batch Pre-loading:")
        prefetch_label.setStyleSheet("font-size: 9pt;")
        prefetch_label.setToolTip("Number of batches to pre-load in memory for smoother training.")
        self.prefetch_factor_entry = QLineEdit(self.params.get("PREFETCH_FACTOR", "2"))
        self.prefetch_factor_entry.setToolTip("2-4 recommended. Higher values use more RAM but reduce GPU waiting time.")
        self.prefetch_factor_entry.textChanged.connect(self.on_param_text_changed)
        self.param_entries["PREFETCH_FACTOR"] = self.prefetch_factor_entry
        form_layout.addRow(prefetch_label, self.prefetch_factor_entry)

        layout.addLayout(form_layout)

    def get_selected_features(self):
        """Retrieve selected feature columns as a list."""
        return [item.text() for item in self.feature_list.selectedItems()]

    def proceed_to_training(self):
        """Legacy method for backward compatibility - uses grid search"""
        self.proceed_to_grid_search()


    def show_training_setup_gui(self):
        # FIXED:Initialize the next GUI after fade-out is complete
        self.training_setup_gui = VEstimTrainSetupGUI(self.params)
        current_geometry = self.geometry()
        self.training_setup_gui.setGeometry(current_geometry)
        # FIXED:self.training_setup_gui.setGeometry(100, 100, 900, 600)
        self.training_setup_gui.show()
        self.close()  # FIXED:Close the previous window

    def load_column_names(self):
        """Loads column names from a sample CSV file in the train processed data folder."""
        train_folder = self.job_manager.get_train_folder()
        if train_folder:
            try:
                csv_files = [f for f in os.listdir(train_folder) if f.endswith(".csv")]
                if not csv_files:
                    raise FileNotFoundError("No CSV files found in train processed data folder.")
                
                sample_csv_path = os.path.join(train_folder, csv_files[0])  # FIXED:Pick the first CSV
                df = pd.read_csv(sample_csv_path, nrows=1)  # FIXED:Load only header
                return list(df.columns)  # FIXED:Return column names

            except Exception as e:
                print(f"Error loading CSV columns: {e}")
        return []
    
    def filter_valid_target_columns(self, column_names):
        """Filter out timestamp/time-related columns that shouldn't be used as targets."""
        # Define patterns for timestamp/time-related columns
        # Note: 'status' columns are NOT filtered out as they can be valid targets
        timestamp_patterns = [
            'time', 'timestamp', 'date', 'datetime', 'epoch', 'unix',
            'year', 'month', 'day', 'hour', 'minute', 'second',
            '_time', '_timestamp', '_date', '_datetime',
            'time_', 'timestamp_', 'date_', 'datetime_'
        ]
        
        valid_targets = []
        filtered_out = []
        
        for col in column_names:
            col_lower = col.lower()
            # Check if column name contains any timestamp pattern (case-insensitive)
            is_timestamp = any(pattern in col_lower for pattern in timestamp_patterns)
            
            if is_timestamp:
                filtered_out.append(col)
            else:
                valid_targets.append(col)
        
        # Log what was filtered out
        if filtered_out:
            self.logger.info(f"Filtered out timestamp columns from target selection: {filtered_out}")
        
        # Safety check - ensure we have at least one valid target
        if not valid_targets:
            self.logger.warning("No valid target columns found after filtering. Including all columns as fallback.")
            return column_names
            
        return valid_targets
    
    def load_params_from_json(self):
        """Load hyperparameters from a JSON file and update the UI."""
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Params", "", "JSON Files (*.json);;All Files (*)")
        if filepath:
            try:
                # FIXED:Load parameters from file
                with open(filepath, 'r') as f:
                    loaded_params = json.load(f)
                
                # FIXED:Validate feature/target columns against current dataset
                validated_params = self.validate_columns_against_dataset(loaded_params)
                
                # FIXED:Load and validate parameters using the manager
                self.params = self.hyper_param_manager.validate_and_normalize_params(validated_params)
                self.logger.info(f"Successfully loaded parameters from {filepath}")

                # FIXED:Update GUI elements with loaded parameters
                self.update_gui_with_loaded_params()

            except Exception as e:
                self.logger.error(f"Failed to load parameters from {filepath}: {e}")
                QMessageBox.critical(self, "Error", f"Failed to load parameters: {str(e)}")


    def update_params(self, new_params):
        """Update hyperparameters and refresh the UI."""
        try:
            self.logger.info(f"Updating parameters: {new_params}")

            # FIXED:Update using the hyperparameter manager
            self.hyper_param_manager.update_params(new_params)

            # FIXED:Refresh the GUI with updated parameters
            self.update_gui_with_loaded_params()

        except ValueError as e:
            self.logger.error(f"Invalid parameter input: {new_params} - Error: {e}")
            QMessageBox.critical(self, "Error", f"Invalid parameter input: {str(e)}")


    def update_gui_with_loaded_params(self):
        """Update the GUI with previously saved parameters, including features & target."""
        
        if not self.params:
            self.logger.warning("No parameters found to update the GUI.")
            return

        # FIXED:✅ Update standard hyperparameters (Text Fields, Dropdowns, Checkboxes)
        for param_name, entry in self.param_entries.items():
            if param_name in self.params:
                value = self.params[param_name]

                # FIXED:✅ Handle different widget types correctly
                if isinstance(entry, QLineEdit):
                    entry.setText(str(value))  # FIXED:Convert value to string for text fields

                elif isinstance(entry, QComboBox):
                    index = entry.findText(str(value))  # FIXED:Get index for dropdowns
                    if index != -1:
                        entry.setCurrentIndex(index)

                elif isinstance(entry, QCheckBox):
                    entry.setChecked(bool(value))  # FIXED:Ensure checkbox reflects state

                elif isinstance(entry, QListWidget):  # FIXED:Multi-Select Feature List
                    selected_items = set(value) if isinstance(value, list) else set([value])
                    for i in range(entry.count()):
                        item = entry.item(i)
                        item.setSelected(item.text() in selected_items)

        # FIXED:✅ Update Model Parameters (Ensure proper model-specific param refresh)
        if "MODEL_TYPE" in self.params:
            model_index = self.model_combo.findText(self.params["MODEL_TYPE"])
            if model_index != -1:
                self.model_combo.setCurrentIndex(model_index)

        # FIXED:✅ Ensure Scheduler Parameters Update Correctly
        if "SCHEDULER_TYPE" in self.params:
            scheduler_index = self.scheduler_combo.findText(self.params["SCHEDULER_TYPE"])
            if scheduler_index != -1:
                self.scheduler_combo.setCurrentIndex(scheduler_index)

        # FIXED:✅ Ensure Training Method and dependent fields update correctly
        if "TRAINING_METHOD" in self.params:
            method_index = self.training_method_combo.findText(self.params["TRAINING_METHOD"])
            if method_index != -1:
                self.training_method_combo.setCurrentIndex(method_index)
        
        # FIXED:Explicitly call update methods after setting combo boxes to ensure UI consistency
        # FIXED:especially if loaded params match current combo box text (so currentIndexChanged doesn't fire)
        self.update_model_params()
        self.update_scheduler_settings()
        self.update_training_method() # FIXED:This will also handle batch size visibility

        # FIXED:Populate Max Training Time H, M, S fields from MAX_TRAINING_TIME_SECONDS
        if "MAX_TRAINING_TIME_SECONDS" in self.params:
            try:
                total_seconds = int(self.params["MAX_TRAINING_TIME_SECONDS"])
                if total_seconds >= 0:
                    hours = total_seconds // 3600
                    minutes = (total_seconds % 3600) // 60
                    seconds = total_seconds % 60
                    
                    if hasattr(self, 'max_time_hours_entry'):
                        self.max_time_hours_entry.setText(str(hours))
                    if hasattr(self, 'max_time_minutes_entry'):
                        self.max_time_minutes_entry.setText(str(minutes))
                    if hasattr(self, 'max_time_seconds_entry'):
                        self.max_time_seconds_entry.setText(str(seconds))
                    self.logger.info(f"Populated Max Training Time H:M:S from loaded MAX_TRAINING_TIME_SECONDS ({total_seconds}s).")
                else:
                    if hasattr(self, 'max_time_hours_entry'):
                        self.max_time_hours_entry.setText("0")
                    if hasattr(self, 'max_time_minutes_entry'):
                        self.max_time_minutes_entry.setText("0")
                    if hasattr(self, 'max_time_seconds_entry'):
                        self.max_time_seconds_entry.setText("0")
            except (ValueError, TypeError) as e:
                self.logger.warning(f"Could not parse MAX_TRAINING_TIME_SECONDS ('{self.params.get('MAX_TRAINING_TIME_SECONDS')}') for GUI: {e}. Setting H:M:S to defaults.")
                if hasattr(self, 'max_time_hours_entry'):
                    self.max_time_hours_entry.setText("0")
                if hasattr(self, 'max_time_minutes_entry'):
                    self.max_time_minutes_entry.setText("0")
                if hasattr(self, 'max_time_seconds_entry'):
                    self.max_time_seconds_entry.setText("0")
        # FIXED:If MAX_TRAINING_TIME_SECONDS is not in params, the QLineEdit defaults (set during creation) will be used.

        self.logger.info("GUI successfully updated with loaded parameters.")
        # FIXED:self.update_auto_search_button_state()

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

    def proceed_to_auto_search(self):
        """Handle auto search (Optuna) button click with validation."""
        self.reset_field_styles()
        
        try:
            new_params = self._collect_basic_params()
            if new_params is None:
                return

            # FIXED:Perform all validations
            is_valid_optuna, error_optuna = self.hyper_param_manager.validate_for_optuna(new_params)
            is_valid_gui, error_gui = self.hyper_param_manager.validate_hyperparameters_for_gui(new_params, search_mode='optuna')

            # FIXED:Combine error messages
            all_errors = []
            if not is_valid_optuna:
                all_errors.append(error_optuna)
            if not is_valid_gui:
                all_errors.append(error_gui)
            
            if all_errors:
                combined_error_message = "\n\n".join(all_errors)
                self.highlight_error_fields(combined_error_message)
                QMessageBox.warning(self, "Validation Error", combined_error_message)
                return
            
            self.hyper_param_manager.update_params(new_params)
            self.hyper_param_manager.save_params()

            from vestim.gui.src.optuna_optimization_gui_qt import VEstimOptunaOptimizationGUI
            
            self.close()
            self.optuna_gui = VEstimOptunaOptimizationGUI(base_params=new_params, job_manager=self.job_manager)
            self.optuna_gui.show()
            
        except Exception as e:
            self.logger.error(f"Error proceeding to auto search: {e}")
            QMessageBox.critical(self, "Error", f"Error proceeding to auto search: {str(e)}")

    def proceed_to_grid_search(self):
        """Handle grid search button click with validation."""
        self.reset_field_styles()
        
        try:
            new_params = self._collect_basic_params()
            if new_params is None:
                return

            # FIXED:Perform all validations
            is_valid_grid, error_grid = self.hyper_param_manager.validate_for_grid_search(new_params)
            is_valid_gui, error_gui = self.hyper_param_manager.validate_hyperparameters_for_gui(new_params, search_mode='grid')

            # FIXED:Combine error messages
            all_errors = []
            if not is_valid_grid:
                all_errors.append(error_grid)
            if not is_valid_gui:
                all_errors.append(error_gui)

            if all_errors:
                combined_error_message = "\n\n".join(all_errors)
                self.highlight_error_fields(combined_error_message)
                QMessageBox.warning(self, "Validation Error", combined_error_message)
                return
            
            self.hyper_param_manager.update_params(new_params)
            self.hyper_param_manager.save_params()

            # FIXED:Proceed directly to training setup with grid search logic
            self.close()  # FIXED:Close current window
            self.training_setup_gui = VEstimTrainSetupGUI(job_manager=self.job_manager, params=new_params)
            self.training_setup_gui.show()
            
        except Exception as e:
            self.logger.error(f"Error proceeding to grid search: {e}")
            QMessageBox.critical(self, "Error", f"Error proceeding to grid search: {str(e)}")

    def _collect_basic_params(self):
        """Collect basic parameters that are common to both Optuna and grid search."""
        new_params = {}

        # FIXED:Collect values from stored param entries
        for param, entry in self.param_entries.items():
            if isinstance(entry, QLineEdit):
                new_params[param] = entry.text().strip()  # FIXED:Text input
            elif isinstance(entry, QComboBox):
                new_params[param] = entry.currentText()  # FIXED:Dropdown selection
            elif isinstance(entry, QListWidget) and param == "OPTIMIZER_TYPE":
                selected_optimizers = [item.text() for item in entry.selectedItems()]
                new_params[param] = ",".join(selected_optimizers)
            elif isinstance(entry, QCheckBox):
                new_params[param] = entry.isChecked()
            elif isinstance(entry, QListWidget):
                new_params[param] = [item.text() for item in entry.selectedItems()]

        # FIXED:Debug: Log collected parameter values
        self.logger.info("Collected parameters from GUI:")
        for param, value in new_params.items():
            self.logger.info(f"  {param}: '{value}' (type: {type(value).__name__})")
        
        # FIXED:Debug: Check scheduler-specific parameters
        if hasattr(self, 'scheduler_combo'):
            selected_scheduler = self.scheduler_combo.currentText()
            self.logger.info(f"Selected scheduler: {selected_scheduler}")
            
            if selected_scheduler == "ReduceLROnPlateau":
                if hasattr(self, 'plateau_factor_entry'):
                    plateau_gui_value = self.plateau_factor_entry.text().strip()
                    self.logger.info(f"Plateau Factor GUI field direct read: '{plateau_gui_value}'")
                    self.logger.info(f"Plateau Factor field visible: {self.plateau_factor_entry.isVisible()}")
                    self.logger.info(f"Plateau Factor field enabled: {self.plateau_factor_entry.isEnabled()}")
                
                if hasattr(self, 'lr_param_entry'):
                    lr_param_gui_value = self.lr_param_entry.text().strip()
                    self.logger.info(f"LR Param GUI field direct read: '{lr_param_gui_value}'")
                    self.logger.info(f"LR Param field visible: {self.lr_param_entry.isVisible()}")
                    self.logger.info(f"LR Param field enabled: {self.lr_param_entry.isEnabled()}")

        # FIXED:Ensure critical fields are selected
        if not new_params.get("FEATURE_COLUMNS"):
            QMessageBox.critical(self, "Selection Error", "Please select at least one feature column.")
            return None
        if not new_params.get("TARGET_COLUMN"):
            QMessageBox.critical(self, "Selection Error", "Please select a target column.")
            return None
            
        # Validate that target column is not a timestamp column
        target_column = new_params.get("TARGET_COLUMN")
        if target_column:
            valid_targets = self.filter_valid_target_columns([target_column])
            if not valid_targets:
                QMessageBox.critical(self, "Invalid Target Column", 
                    f"The selected target column '{target_column}' appears to be a timestamp/time column.\n\n"
                    "Timestamp columns cannot be used as regression targets as they would cause NaN losses.\n\n"
                    "Please select a different target column (like 'Voltage', 'Current', etc.).")
                return None
                
        if not new_params.get("MODEL_TYPE"):
            QMessageBox.critical(self, "Selection Error", "Please select a model type.")
            return None

        # FIXED:Validate REPETITIONS specifically as it's a QLineEdit now
        if "REPETITIONS" in new_params:
            try:
                repetitions_val = int(new_params["REPETITIONS"])
                if repetitions_val < 1:
                    QMessageBox.warning(self, "Invalid Input", "Repetitions must be at least 1.")
                    return None
                new_params["REPETITIONS"] = repetitions_val # FIXED:Store as int after validation
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Repetitions (in Validation Criteria) must be a valid integer.")
                return None
        else:
            QMessageBox.warning(self, "Missing Information", "Please fill in the 'REPETITIONS' field.")
            return None

        # FIXED:Special handling for FNN models - ensure training method is set correctly
        if new_params.get("MODEL_TYPE") == "FNN":
            # FIXED:For FNN, always use "WholeSequenceFNN" training method (data loader expects this)
            new_params["TRAINING_METHOD"] = "WholeSequenceFNN"
            # FIXED:Ensure batch training is enabled for FNN
            new_params["BATCH_TRAINING"] = True
            # FIXED:Ensure batch size has a proper default for FNN
            if not new_params.get("BATCH_SIZE") or new_params.get("BATCH_SIZE").strip() == "":
                new_params["BATCH_SIZE"] = "5000"
        
        # FIXED:Special handling for LSTM/GRU models - ensure compatible training method
        elif new_params.get("MODEL_TYPE") in ["LSTM", "GRU"]:
            # FIXED:For RNN models, only "Sequence-to-Sequence" is supported by data loader
            if new_params.get("TRAINING_METHOD") == "Whole Sequence":
                new_params["TRAINING_METHOD"] = "Sequence-to-Sequence"
                self.logger.info("Converted 'Whole Sequence' to 'Sequence-to-Sequence' for LSTM/GRU model")

        return new_params


    def parse_boundary_format(self, value):
        """Parse boundary format [min,max] and return min, max values."""
        if not value or not value.strip():
            return None, None
        
        value = value.strip()
        if value.startswith('[') and value.endswith(']'):
            try:
                inner = value[1:-1].strip()
                parts = [part.strip() for part in inner.split(',')]
                if len(parts) == 2:
                    min_val = float(parts[0])
                    max_val = float(parts[1])
                    return min_val, max_val
            except ValueError:
                pass
        return None, None

    def convert_params_for_optuna(self, params):
        """Convert parameter dictionary with boundary format to Optuna-compatible format."""
        optuna_params = {}
        
        # FIXED:Define which parameters should be treated as integers vs floats
        integer_params = {
            "LAYERS", "HIDDEN_UNITS", "GRU_LAYERS", "GRU_HIDDEN_UNITS", 
            "MAX_EPOCHS", "VALID_PATIENCE", "VALID_FREQUENCY", "LOOKBACK",
            "BATCH_SIZE", "LR_PERIOD", "PLATEAU_PATIENCE", "REPETITIONS"
        }
        
        for key, value in params.items():
            if isinstance(value, str) and '[' in value and ']' in value:
                min_val, max_val = self.parse_boundary_format(value)
                if min_val is not None and max_val is not None:
                    optuna_params[key] = {
                        'type': 'int' if key in integer_params else 'float',
                        'low': int(min_val) if key in integer_params else min_val,
                        'high': int(max_val) if key in integer_params else max_val
                    }
                else:
                    # FIXED:Keep original value if parsing fails
                    optuna_params[key] = value
            else:
                # FIXED:Keep non-boundary parameters as-is
                optuna_params[key] = value
                
        return optuna_params

    def _convert_time_fields_to_seconds(self, params):
        """Convert MAX_TRAIN_HOURS, MAX_TRAIN_MINUTES, MAX_TRAIN_SECONDS to MAX_TRAINING_TIME_SECONDS."""
        try:
            hours = int(params.get("MAX_TRAIN_HOURS", "0"))
            minutes = int(params.get("MAX_TRAIN_MINUTES", "0"))
            seconds = int(params.get("MAX_TRAIN_SECONDS", "0"))
            
            total_seconds = hours * 3600 + minutes * 60 + seconds
            params["MAX_TRAINING_TIME_SECONDS"] = total_seconds
            
            # FIXED:Remove the individual time components as they're now consolidated
            for key in ["MAX_TRAIN_HOURS", "MAX_TRAIN_MINUTES", "MAX_TRAIN_SECONDS"]:
                if key in params:
                    del params[key]
                    
            self.logger.info(f"Converted time fields to total seconds: {total_seconds}")
            
        except ValueError as e:
            self.logger.warning(f"Error converting time fields: {e}. Setting total time to 0.")
            params["MAX_TRAINING_TIME_SECONDS"] = 0


    def on_param_text_changed(self, text=None):
        """Resets the style of a QLineEdit when its text is changed."""
        sender = self.sender()
        if sender in self.error_fields:
            sender.setStyleSheet("")
            self.error_fields.remove(sender)

    def reset_field_styles(self):
        """Resets the stylesheet for all QLineEdit widgets to default."""
        for entry in self.param_entries.values():
            if isinstance(entry, QLineEdit):
                entry.setStyleSheet("")
        self.error_fields.clear()

    def highlight_error_fields(self, error_message):
        """Highlights QLineEdit fields that are mentioned in the error message."""
        import re
        # FIXED:Regex to find capitalized keys like 'LAYERS', 'FNN_HIDDEN_LAYERS', etc.
        # FIXED:It looks for single-quoted uppercase words with underscores.
        error_keys = re.findall(r"'([A-Z_]+)'", error_message)
        
        for key in error_keys:
            if key in self.param_entries and isinstance(self.param_entries[key], QLineEdit):
                self.param_entries[key].setStyleSheet("border: 1px solid red;")
                self.error_fields.add(self.param_entries[key])
