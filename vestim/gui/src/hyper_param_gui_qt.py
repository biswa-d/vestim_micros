"""
Hyperparameter selection GUI for VEstim.

Provides controls to select model, training, and data-related parameters.
Comments are concise and professional; unnecessary markers removed.
"""


import os
import json
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, 
    QLineEdit, QFileDialog, QMessageBox, QDialog, QGroupBox, QComboBox, QListWidget, QAbstractItemView,QFormLayout, QCheckBox, QScrollArea, QDesktopWidget
)
from PyQt5.QtCore import Qt, QPropertyAnimation
from PyQt5.QtGui import QIcon, QColor, QPalette

import pandas as pd
from vestim.utils.gpu_setup import _safe_import_torch

import logging

from vestim.gateway.src.job_manager_qt import JobManager
from vestim.gateway.src.hyper_param_manager_qt import VEstimHyperParamManager
from vestim.gui.src.training_setup_gui_qt import VEstimTrainSetupGUI
from vestim.config_manager import get_default_hyperparams, update_last_used_hyperparams, load_hyperparams_from_root
from vestim.gui.src.adaptive_gui_utils import scale_font, scale_widget_size, get_adaptive_stylesheet

# Initialize the JobManager
import logging
class VEstimHyperParamGUI(QWidget):
    def __init__(self, job_manager=None):
        self.logger = logging.getLogger(__name__)  # Initialize the module logger
        self.logger.info("Initializing Hyperparameter GUI")
        super().__init__()
        self.params = {}  # Parameters loaded from defaults or user selection
        self.job_manager = job_manager if job_manager else JobManager()
        self.hyper_param_manager = VEstimHyperParamManager(job_manager=self.job_manager)
        self.param_entries = {}  # Mapping of parameter names to input widgets
        self.error_fields = set() # Track fields with validation errors

        self.setup_window()
        self.build_gui()
        
        # Load default hyperparameters after UI is built
        self.load_default_hyperparameters()

    def setup_window(self):
        """Initial setup for the main window appearance with responsive sizing."""
        self.setWindowTitle("VEstim - Hyperparameter Selection")
        
        # Adapt window size to screen
        self.setGeometry(100, 100, scale_widget_size(1200), scale_widget_size(900))
        self.setMinimumSize(scale_widget_size(900), scale_widget_size(600))
        
        # Enable DPI scaling
        self.setAttribute(Qt.WA_AcceptTouchEvents)
        self.setMouseTracking(True)  # Enable mouse tracking for hover effects
        
        # Load the application icon
        resources_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources')
        icon_path = os.path.join(resources_path, 'icon.ico')
        
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        else:
            self.logger.warning("Icon file not found. Make sure 'icon.ico' is in the correct directory.")
        
        self.setStyleSheet("""
            QToolTip { font-weight: normal; font-size: 10pt; }
            QPushButton:disabled {
                background-color: #d3d3d3 !important;
                color: #a9a9a9 !important;
            }
            QScrollArea {
                border: none;
            }
        """)

    def load_default_hyperparameters(self):
        """Load default hyperparameters and populate the GUI with column validation."""
        import traceback
        try:
            # Get default hyperparameters from config (includes last used params with features/targets)
            default_params = get_default_hyperparams()
            self.logger.info(f"DEBUG: Type of default_params: {type(default_params)}")
            self.logger.info(f"DEBUG: Contents of default_params: {default_params}")

            # Validate feature/target columns against current dataset
            validated_params = self.validate_columns_against_dataset(default_params)
            self.logger.info(f"DEBUG: Type of validated_params: {type(validated_params)}")
            self.logger.info(f"DEBUG: Contents of validated_params: {validated_params}")

            # Load and validate parameters using the manager (same as load_params_from_json)
            self.params = self.hyper_param_manager.validate_and_normalize_params(validated_params)
            self.logger.info("Successfully loaded default hyperparameters automatically")

            # Update GUI elements with loaded parameters (same as load_params_from_json)
            self.update_gui_with_loaded_params()

        except Exception as e:
            self.logger.error(f"Failed to auto-load default hyperparameters: {type(e).__name__}: {e}")
            self.logger.error(traceback.format_exc())
            # If auto-load fails, continue with empty params; user can load manually

    def validate_columns_against_dataset(self, params):
        """Validate that feature and target columns exist in the current dataset"""
        try:
            available_columns = self.load_column_names()
            if not available_columns:
                self.logger.warning("No columns available in dataset, using parameters as-is")
                return params

            validated_params = params.copy()

            # Only remove missing features, keep the rest
            if "FEATURE_COLUMNS" in params and params["FEATURE_COLUMNS"]:
                original_features = params["FEATURE_COLUMNS"]
                if isinstance(original_features, list):
                    valid_features = [col for col in original_features if col in available_columns]
                    missing_features = [col for col in original_features if col not in available_columns]
                    if missing_features:
                        self.logger.info(f"Some saved feature columns not found in dataset. Missing: {missing_features}. Using available: {valid_features}")
                    validated_params["FEATURE_COLUMNS"] = valid_features
                else:
                    validated_params["FEATURE_COLUMNS"] = []
                    self.logger.warning("Invalid feature columns format, clearing features.")

            # If target column missing, clear it (don't force a default)
            if "TARGET_COLUMN" in params and params["TARGET_COLUMN"]:
                original_target = params["TARGET_COLUMN"]
                if original_target not in available_columns:
                    validated_params["TARGET_COLUMN"] = ""
                    self.logger.warning(f"Saved target column '{original_target}' not found in dataset. Clearing target.")

            return validated_params
        except Exception as e:
            self.logger.error(f"Error validating columns against dataset: {e}")
            return params
            
            return validated_params
            
        except Exception as e:
            self.logger.error(f"Error validating columns against dataset: {e}")
            return params  # Return original params if validation fails

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

        # Title and guidance section
        title_label = QLabel("Select Hyperparameters for Model Training")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(get_adaptive_stylesheet("font-size: 20pt; font-weight: bold; color: #0b6337; margin-bottom: 15px;"))
        content_layout.addWidget(title_label)

        guide_button = QPushButton("Open Hyperparameter Guide")
        guide_button.setFixedHeight(scale_widget_size(40))
        guide_button.setStyleSheet(get_adaptive_stylesheet("""
            QPushButton {
                font-size: 10pt !important;
                background-color: #f0f0f0 !important;
                border: 2px solid #cccccc !important;
                border-radius: 6px !important;
                padding: 5px !important;
                color: #333333 !important;
            }
            QPushButton:hover {
                background-color: #e0e0e0 !important;
                border: 2px solid #999999 !important;
            }
            QPushButton:pressed {
                background-color: #d0d0d0 !important;
                border: 2px solid #777777 !important;
            }
        """))
        guide_button.setAttribute(Qt.WA_Hover, True)  # Explicitly enable hover events
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
        instructions_label.setStyleSheet(get_adaptive_stylesheet("font-size: 10pt; color: gray; margin-bottom: 10px;"))
        instructions_label.setWordWrap(True)  # Allow text wrapping
        content_layout.addWidget(instructions_label)

        # Hyperparameter selection section
        hyperparam_section = QGridLayout()
        hyperparam_section.setSpacing(scale_widget_size(10))  # Add spacing between grid items
        group_box_style = get_adaptive_stylesheet("QGroupBox { font-size: 10pt; font-weight: bold; }")

        # Row 0
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

        # Row 1 (Left) and Row 2 (Left)
        model_training_group = QGroupBox("Model and Training Method")
        model_training_group.setStyleSheet(group_box_style)
        model_training_layout = QVBoxLayout()  # Vertical layout for training controls

        self.add_model_selection(model_training_layout)
        self.add_training_method_selection(model_training_layout)
        
        model_training_group.setLayout(model_training_layout)

        # Row 1 (Right)
        validation_group = QGroupBox("Validation Training")
        validation_group.setStyleSheet(group_box_style)
        validation_criteria_layout = QVBoxLayout()
        self.add_validation_criteria(validation_criteria_layout)
        validation_group.setLayout(validation_criteria_layout)

        # Row 2 (Right)
        lr_group = QGroupBox("Learning Rate Scheduler")
        lr_group.setStyleSheet(group_box_style)
        lr_layout = QVBoxLayout()  # Changed from QHBoxLayout to QVBoxLayout for vertical stacking

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

        # Add widgets to the 3-column grid layout
        # Column 1: Data Selection (row 0), Model and Training Method (rows 1-2)
        hyperparam_section.addWidget(data_selection_group, 0, 0)
        hyperparam_section.addWidget(model_training_group, 1, 0, 2, 1)
        
        # Column 2: Device and Optimizer (row 0), Validation Training (row 1)
        hyperparam_section.addWidget(device_optimizer_group, 0, 1)
        hyperparam_section.addWidget(validation_group, 1, 1)

        # Inference Filter Group (initially hidden)
        self.inference_filter_group = QGroupBox("Inference Filter")
        self.inference_filter_group.setStyleSheet(group_box_style)
        inference_filter_layout = QVBoxLayout()
        self.add_inference_filter_selection(inference_filter_layout)
        self.inference_filter_group.setLayout(inference_filter_layout)
        hyperparam_section.addWidget(self.inference_filter_group, 2, 1) # Add to grid
        
        # Column 3: Learning Rate Scheduler (spans rows 0-2)
        hyperparam_section.addWidget(lr_group, 0, 2, 3, 1)
        
        # Set equal column stretch for balanced layout
        hyperparam_section.setColumnStretch(0, 1)
        hyperparam_section.setColumnStretch(1, 1)
        hyperparam_section.setColumnStretch(2, 1)

        content_layout.addLayout(hyperparam_section)

        # Bottom buttons
        button_layout = QVBoxLayout()
        load_button = QPushButton("Load Params from File")
        load_button.setFixedHeight(scale_widget_size(40))
        load_button.setStyleSheet(get_adaptive_stylesheet("""
            QPushButton {
                font-size: 10pt !important;
                background-color: #f0f0f0 !important;
                border: 2px solid #cccccc !important;
                border-radius: 6px !important;
                padding: 5px !important;
                color: #333333 !important;
            }
            QPushButton:hover {
                background-color: #e0e0e0 !important;
                border: 2px solid #999999 !important;
            }
            QPushButton:pressed {
                background-color: #d0d0d0 !important;
                border: 2px solid #777777 !important;
            }
        """))
        load_button.setAttribute(Qt.WA_Hover, True)  # Explicitly enable hover events
        load_button.clicked.connect(self.load_params_from_json)
        button_layout.addWidget(load_button, alignment=Qt.AlignCenter)

        search_method_layout = QHBoxLayout()
        search_method_layout.setAlignment(Qt.AlignCenter)
        
        auto_search_button = QPushButton("Auto Search (Optuna)")
        auto_search_button.setFixedHeight(scale_widget_size(45))
        auto_search_button.setStyleSheet(get_adaptive_stylesheet("""
            QPushButton {
                background-color: #2E86AB !important;
                color: white !important;
                font-size: 10pt !important;
                border: none !important;
                border-radius: 6px !important;
                font-weight: bold !important;
                padding: 8px !important;
            }
            QPushButton:hover {
                background-color: #246B8A !important;
            }
            QPushButton:pressed {
                background-color: #1E5670 !important;
            }
        """))
        auto_search_button.setAttribute(Qt.WA_Hover, True)  # Explicitly enable hover events
        auto_search_button.setToolTip("Use Optuna for automatic hyperparameter optimization.\nRequires boundary format [min,max] for core hyperparameters (layers, hidden units, learning rate, epochs).\nTime and validation parameters can use single values.\nExample: [1,5] for layers, [0.001,0.1] for learning rate")
        auto_search_button.clicked.connect(self.proceed_to_auto_search)
        self.auto_search_button = auto_search_button
        
        grid_search_button = QPushButton("Exhaustive Grid Search")
        grid_search_button.setFixedHeight(scale_widget_size(45))
        grid_search_button.setStyleSheet(get_adaptive_stylesheet("""
            QPushButton {
                background-color: #0b6337 !important;
                color: white !important;
                font-size: 10pt !important;
                border: none !important;
                border-radius: 6px !important;
                font-weight: bold !important;
                padding: 8px !important;
            }
            QPushButton:hover {
                background-color: #094D2A !important;
            }
            QPushButton:pressed {
                background-color: #073A20 !important;
            }
        """))
        grid_search_button.setAttribute(Qt.WA_Hover, True)  # Explicitly enable hover events
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

        # Feature selection
        feature_label = QLabel("Feature Columns (Input):")
        feature_label.setStyleSheet("font-size: 9pt;")
        feature_label.setToolTip("Select one or more columns as input features for training.")

        self.feature_list = QListWidget()
        self.feature_list.addItems(column_names)  # Features can include all columns
        self.feature_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.feature_list.setFixedHeight(100)
        self.feature_list.setToolTip("Select multiple features.")

        # Target selection
        target_label = QLabel("Target Column (Output):")
        target_label.setStyleSheet("font-size: 9pt;")
        target_label.setToolTip("<html><body><span style='font-weight: normal;'>Select the output column for the model to predict.<br><b>Note:</b> Timestamp/time columns are filtered out.</span></body></html>")

        self.target_combo = QComboBox()
        self.target_combo.addItems(valid_target_columns)  # Only valid targets
        self.target_combo.setToolTip("Select a single target column (timestamp columns excluded).")

        # Store references in self.param_entries for parameter collection
        self.param_entries["FEATURE_COLUMNS"] = self.feature_list
        self.param_entries["TARGET_COLUMN"] = self.target_combo

        # Form layout for alignment
        form_layout = QFormLayout()
        form_layout.addRow(feature_label, self.feature_list)
        form_layout.addRow(target_label, self.target_combo)

        # Apply to parent layout
        layout.addLayout(form_layout)

    def add_training_method_selection(self, layout):
        """Adds training method selection with batch size, train-validation split, tooltips, and ensures UI alignment."""

        # Main layout with top alignment
        training_layout = QVBoxLayout()
        training_layout.setAlignment(Qt.AlignTop)  # Keep content at the top

        # Training method selection dropdown
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

        # Lookback parameter (only for sequence-to-sequence)
        self.lookback_label = QLabel("Lookback Window:")
        self.lookback_label.setStyleSheet("font-size: 9pt;")
        self.lookback_label.setToolTip("Defines how many previous time steps are used for each prediction.")
        self.lookback_entry = QLineEdit(self.params.get("LOOKBACK", "400"))
        self.lookback_entry.textChanged.connect(self.on_param_text_changed)

        # Batch training option (checkbox)
        self.batch_training_checkbox = QCheckBox("Enable Batch Training")
        self.batch_training_checkbox.setChecked(True)  # Default is checked
        self.batch_training_checkbox.setToolTip("Enable mini-batch training. This is required for FNN and recommended for sequence-based methods.")
        self.batch_training_checkbox.stateChanged.connect(self.update_batch_size_visibility)

        # Batch size entry (initially enabled as checkbox is checked by default)
        self.batch_size_label = QLabel("Batch Size:")  # Instance variable to hide/show
        self.batch_size_label.setStyleSheet("font-size: 9pt;")
        self.batch_size_label.setToolTip("Number of samples per batch.")
        self.batch_size_entry = QLineEdit(self.params.get("BATCH_SIZE", "100"))  # Default value 100
        self.batch_size_entry.textChanged.connect(self.on_param_text_changed)
        self.batch_size_entry.setEnabled(True)  # Initially enabled

        # Store references in self.param_entries for parameter collection
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

        is_rnn_model = current_model_type in ["LSTM", "GRU", "LSTM_EMA", "LSTM_LPF"]
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
        self.model_combo.setToolTip("LSTM for time-series, FNN for non-sequential data, GRU for memory-efficient training, LSTM_EMA and LSTM_LPF for filtered outputs.")

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
        # Updated to use RNN_LAYER_SIZES instead of separate LAYERS/HIDDEN_UNITS fields
        rnn_specific_keys = ["RNN_LAYER_SIZES", "LAYERS", "HIDDEN_UNITS", "GRU_LAYERS", "GRU_HIDDEN_UNITS"] # FIXED:Include both new and legacy keys
        fnn_specific_keys = ["FNN_HIDDEN_LAYERS", "FNN_DROPOUT_PROB", "FNN_ACTIVATION"] # FIXED:Add FNN specific QLineEdit keys
        
        all_model_specific_keys = rnn_specific_keys + fnn_specific_keys
        
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
        if selected_model in ["LSTM", "LSTM_EMA", "LSTM_LPF"] or selected_model == "":
            lstm_layer_sizes_label = QLabel("LSTM Layer Sizes:")
            lstm_layer_sizes_label.setStyleSheet("font-size: 9pt;")
            lstm_layer_sizes_label.setToolTip("Define LSTM layer sizes. Single config: 64,32. Multiple configs: use semicolons (64,32;128,64,32) or brackets ([64,32], [128,64,32]).")

            # Get default value from RNN_LAYER_SIZES, LSTM_UNITS, or fallback to legacy params
            default_value = self.params.get("RNN_LAYER_SIZES") or self.params.get("LSTM_UNITS")
            if not default_value:
                # Fallback to legacy LAYERS + HIDDEN_UNITS
                layers = self.params.get("LAYERS", "1")
                hidden = self.params.get("HIDDEN_UNITS", "10")
                # Convert to comma-separated format: LAYERS=2, HIDDEN_UNITS=10 -> "10,10"
                try:
                    default_value = ",".join([str(hidden)] * int(layers))
                except:
                    default_value = "64,32"

            self.lstm_layer_sizes_entry = QLineEdit(default_value)
            self.lstm_layer_sizes_entry.setToolTip("Single: '64,32,16' | Multiple with semicolons: '64,32;128,64' | Multiple with brackets: '[64,32], [128,64]'")
            self.lstm_layer_sizes_entry.textChanged.connect(self.on_param_text_changed)

            self.model_param_container.addWidget(lstm_layer_sizes_label)
            self.model_param_container.addWidget(self.lstm_layer_sizes_entry)

            # Store in param_entries using new parameter name
            model_params["RNN_LAYER_SIZES"] = self.lstm_layer_sizes_entry

        # FIXED:**GRU Parameters**
        elif selected_model == "GRU":
            gru_layer_sizes_label = QLabel("GRU Layer Sizes:")
            gru_layer_sizes_label.setStyleSheet("font-size: 9pt;")
            gru_layer_sizes_label.setToolTip("Define GRU layer sizes. Single config: 64,32. Multiple configs: use semicolons (64,32;128,64,32) or brackets ([64,32], [128,64,32]).")

            # Get default value from RNN_LAYER_SIZES, GRU_UNITS, or fallback to legacy params
            default_value = self.params.get("RNN_LAYER_SIZES") or self.params.get("GRU_UNITS")
            if not default_value:
                # Fallback to legacy GRU_LAYERS + GRU_HIDDEN_UNITS
                layers = self.params.get("GRU_LAYERS", "1")
                hidden = self.params.get("GRU_HIDDEN_UNITS", "10")
                # Convert to comma-separated format: GRU_LAYERS=2, GRU_HIDDEN_UNITS=10 -> "10,10"
                try:
                    default_value = ",".join([str(hidden)] * int(layers))
                except:
                    default_value = "64,32"

            self.gru_layer_sizes_entry = QLineEdit(default_value)
            self.gru_layer_sizes_entry.setToolTip("Single: '64,32,16' | Multiple with semicolons: '64,32;128,64' | Multiple with brackets: '[64,32], [128,64]'")
            self.gru_layer_sizes_entry.textChanged.connect(self.on_param_text_changed)

            self.model_param_container.addWidget(gru_layer_sizes_label)
            self.model_param_container.addWidget(self.gru_layer_sizes_entry)

            # Store in param_entries using new parameter name
            model_params["RNN_LAYER_SIZES"] = self.gru_layer_sizes_entry

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

            fnn_activation_label = QLabel("Activation Function:")
            fnn_activation_label.setStyleSheet("font-size: 9pt;")
            fnn_activation_label.setToolTip("Select the activation function for hidden layers.")
            self.fnn_activation_combo = QComboBox()
            self.fnn_activation_combo.addItems(["ReLU", "GELU"])
            self.param_entries["FNN_ACTIVATION"] = self.fnn_activation_combo
            self.model_param_container.addWidget(fnn_activation_label)
            self.model_param_container.addWidget(self.fnn_activation_combo)

            # Store in model_params for later update to self.param_entries
            model_params["FNN_HIDDEN_LAYERS"] = self.fnn_hidden_layers_entry
            model_params["FNN_DROPOUT_PROB"] = self.fnn_dropout_entry
            model_params["FNN_ACTIVATION"] = self.fnn_activation_combo

        # Register current model-specific QLineEdit parameters in self.param_entries
        # FIXED:This ensures self.param_entries only contains widgets relevant to the *current* model type
        self.param_entries.update(model_params)

        # Show/hide the inference filter group based on model selection
        if hasattr(self, 'inference_filter_group'):
            self.inference_filter_group.setVisible(True)



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
        scheduler_options = ["StepLR", "ReduceLROnPlateau", "CosineAnnealingWarmRestarts"]
        self.scheduler_combo.addItems(scheduler_options)
        self.scheduler_combo.setToolTip(
            "StepLR: Reduces LR at fixed intervals.\n"
            "ReduceLROnPlateau: Reduces LR when training stagnates.\n"
            "CosineAnnealingWarmRestarts: Cosine annealing with warm restarts for better convergence."
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

        # **CosineAnnealingWarmRestarts Parameters**
        self.cosine_t0_label = QLabel("T_0 (Initial restart):")
        self.cosine_t0_label.setStyleSheet("font-size: 9pt;")
        self.cosine_t0_label.setToolTip("Number of epochs for the first restart cycle.")
        self.cosine_t0_entry = QLineEdit(self.params.get("COSINE_T0", "10"))
        self.cosine_t0_entry.setToolTip("Smaller values = more frequent restarts. Good starting point: 10-20 epochs.")
        self.cosine_t0_entry.textChanged.connect(self.on_param_text_changed)
        self.param_entries["COSINE_T0"] = self.cosine_t0_entry

        self.cosine_tmult_label = QLabel("T_mult (Restart multiplier):")
        self.cosine_tmult_label.setStyleSheet("font-size: 9pt;")
        self.cosine_tmult_label.setToolTip("Factor by which T_0 is multiplied after each restart.")
        self.cosine_tmult_entry = QLineEdit(self.params.get("COSINE_T_MULT", "2"))
        self.cosine_tmult_entry.setToolTip("T_mult=1: fixed cycle length, T_mult=2: doubling cycles (recommended)")
        self.cosine_tmult_entry.textChanged.connect(self.on_param_text_changed)
        self.param_entries["COSINE_T_MULT"] = self.cosine_tmult_entry

        self.cosine_eta_min_label = QLabel("Eta Min (Min LR):")
        self.cosine_eta_min_label.setStyleSheet("font-size: 9pt;")
        self.cosine_eta_min_label.setToolTip("Minimum learning rate for cosine annealing.")
        self.cosine_eta_min_entry = QLineEdit(self.params.get("COSINE_ETA_MIN", "1e-6"))
        self.cosine_eta_min_entry.setToolTip("Very small LR at cycle bottom, prevents LR from reaching zero.")
        self.cosine_eta_min_entry.textChanged.connect(self.on_param_text_changed)
        self.param_entries["COSINE_ETA_MIN"] = self.cosine_eta_min_entry

        # FIXED:Initially hide all scheduler-specific parameters
        self.plateau_patience_label.setVisible(False)
        self.plateau_patience_entry.setVisible(False)
        self.plateau_factor_label.setVisible(False)
        self.plateau_factor_entry.setVisible(False)
        
        self.cosine_t0_label.setVisible(False)
        self.cosine_t0_entry.setVisible(False)
        self.cosine_tmult_label.setVisible(False)
        self.cosine_tmult_entry.setVisible(False)
        self.cosine_eta_min_label.setVisible(False)
        self.cosine_eta_min_entry.setVisible(False)

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
        form_layout.addRow(self.cosine_t0_label, self.cosine_t0_entry)
        form_layout.addRow(self.cosine_tmult_label, self.cosine_tmult_entry)
        form_layout.addRow(self.cosine_eta_min_label, self.cosine_eta_min_entry)
        
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
            
            self.cosine_t0_label.setVisible(False)
            self.cosine_t0_entry.setVisible(False)
            self.cosine_tmult_label.setVisible(False)
            self.cosine_tmult_entry.setVisible(False)
            self.cosine_eta_min_label.setVisible(False)
            self.cosine_eta_min_entry.setVisible(False)

        elif selected_scheduler == "ReduceLROnPlateau":
            # Hide StepLR params - ReduceLROnPlateau has its own dedicated fields
            self.lr_param_label.setVisible(False)
            self.lr_param_entry.setVisible(False)
            self.lr_period_label.setVisible(False)
            self.lr_period_entry.setVisible(False)

            # Show ReduceLROnPlateau params
            self.plateau_patience_label.setVisible(True)
            self.plateau_patience_entry.setVisible(True)
            self.plateau_factor_label.setVisible(True)
            self.plateau_factor_entry.setVisible(True)
            
            self.cosine_t0_label.setVisible(False)
            self.cosine_t0_entry.setVisible(False)
            self.cosine_tmult_label.setVisible(False)
            self.cosine_tmult_entry.setVisible(False)
            self.cosine_eta_min_label.setVisible(False)
            self.cosine_eta_min_entry.setVisible(False)

        elif selected_scheduler == "CosineAnnealingWarmRestarts":
            self.lr_param_label.setVisible(False)
            self.lr_param_entry.setVisible(False)
            self.lr_period_label.setVisible(False)
            self.lr_period_entry.setVisible(False)

            self.plateau_patience_label.setVisible(False)
            self.plateau_patience_entry.setVisible(False)
            self.plateau_factor_label.setVisible(False)
            self.plateau_factor_entry.setVisible(False)
            
            self.cosine_t0_label.setVisible(True)
            self.cosine_t0_entry.setVisible(True)
            self.cosine_tmult_label.setVisible(True)
            self.cosine_tmult_entry.setVisible(True)
            self.cosine_eta_min_label.setVisible(True)
            self.cosine_eta_min_entry.setVisible(True)


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

        # Store references in self.param_entries for parameter collection
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
        torch = _safe_import_torch()
        try:
            if torch and torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    device_options.append(f"cuda:{i}")
        except Exception as _gpu_e:
            # If querying CUDA fails (e.g., missing DLLs), silently fall back to CPU
            pass
        self.device_combo.addItems(device_options)
        self.device_combo.setToolTip("Select CPU for compatibility, GPU for faster training.")
        try:
            default_device = "cuda:0" if (torch and torch.cuda.is_available()) else "CPU"
        except Exception:
            default_device = "CPU"
        if default_device in device_options:
            self.device_combo.setCurrentText(default_device)
        elif "CPU" in device_options:
            self.device_combo.setCurrentText("CPU")
        self.param_entries["DEVICE_SELECTION"] = self.device_combo
        form_layout.addRow(device_label, self.device_combo)

        try:
            if torch and torch.cuda.is_available():
                self.mixed_precision_checkbox = QCheckBox("Use Mixed Precision Training")
                self.mixed_precision_checkbox.setChecked(True)
                self.mixed_precision_checkbox.setToolTip("Enable automatic mixed precision (AMP) to accelerate GPU training.")
                self.param_entries["USE_MIXED_PRECISION"] = self.mixed_precision_checkbox
                form_layout.addRow(self.mixed_precision_checkbox)
        except Exception:
            pass

        optimizer_label = QLabel("Optimizer:")
        optimizer_label.setStyleSheet("font-size: 9pt;")
        optimizer_label.setToolTip("Select one or more optimization algorithms for grid search.")
        self.optimizer_list = QListWidget()
        # Include AdamW as a selectable optimizer
        self.optimizer_list.addItems(["Adam", "AdamW", "SGD", "RMSprop"])
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

        # CPU Threads and optimization settings (integrated into Device and Optimizer section)
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

    def add_inference_filter_selection(self, layout):
        """Adds UI components for selecting a post-inference filter."""
        form_layout = QFormLayout()

        filter_type_label = QLabel("Filter Type:")
        filter_type_label.setStyleSheet("font-size: 9pt;")
        self.filter_type_combo = QComboBox()
        self.filter_type_combo.addItems(["None", "Moving Average", "Exponential Moving Average", "Savitzky-Golay"])
        self.param_entries["INFERENCE_FILTER_TYPE"] = self.filter_type_combo
        form_layout.addRow(filter_type_label, self.filter_type_combo)

        self.filter_window_label = QLabel("Window Size:")
        self.filter_window_label.setStyleSheet("font-size: 9pt;")
        self.filter_window_entry = QLineEdit("101")
        self.param_entries["INFERENCE_FILTER_WINDOW_SIZE"] = self.filter_window_entry
        form_layout.addRow(self.filter_window_label, self.filter_window_entry)

        self.filter_alpha_label = QLabel("Alpha (EMA):")
        self.filter_alpha_label.setStyleSheet("font-size: 9pt;")
        self.filter_alpha_entry = QLineEdit("0.1")
        self.param_entries["INFERENCE_FILTER_ALPHA"] = self.filter_alpha_entry
        form_layout.addRow(self.filter_alpha_label, self.filter_alpha_entry)

        self.filter_polyorder_label = QLabel("Polynomial Order (SavGol):")
        self.filter_polyorder_label.setStyleSheet("font-size: 9pt;")
        self.filter_polyorder_entry = QLineEdit("2")
        self.param_entries["INFERENCE_FILTER_POLYORDER"] = self.filter_polyorder_entry
        form_layout.addRow(self.filter_polyorder_label, self.filter_polyorder_entry)

        self.filter_type_combo.currentIndexChanged.connect(self.update_inference_filter_params)
        
        layout.addLayout(form_layout)
        self.update_inference_filter_params()

    def update_inference_filter_params(self):
        """Shows/hides filter parameter fields based on the selected filter type."""
        if not hasattr(self, 'filter_type_combo'): return # Widgets not created yet
        filter_type = self.filter_type_combo.currentText()
        
        is_ma = (filter_type == "Moving Average")
        is_ema = (filter_type == "Exponential Moving Average")
        is_savgol = (filter_type == "Savitzky-Golay")

        self.filter_window_label.setVisible(is_ma or is_savgol)
        self.filter_window_entry.setVisible(is_ma or is_savgol)
        
        self.filter_alpha_label.setVisible(is_ema)
        self.filter_alpha_entry.setVisible(is_ema)

        self.filter_polyorder_label.setVisible(is_savgol)
        self.filter_polyorder_entry.setVisible(is_savgol)

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
        # Open defaults templates directory
        from vestim.config_manager import get_defaults_directory
        default_dir = get_defaults_directory()
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Params", default_dir, "JSON Files (*.json);;All Files (*)")
        if filepath:
            try:
                # FIXED:Load parameters from file
                with open(filepath, 'r') as f:
                    loaded_params = json.load(f)
                
                # FIXED:Validate feature/target columns against current dataset
                validated_params = self.validate_columns_against_dataset(loaded_params)
                
                # FIXED:Load and validate parameters using the manager
                self.params = self.hyper_param_manager.validate_and_normalize_params(validated_params)
                
                # CRITICAL: Replace manager's current_params completely instead of merging
                # This prevents old parameters from previous runs from persisting
                self.hyper_param_manager.current_params = self.params.copy()
                
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

        # CRITICAL: Clear all text fields that are NOT in loaded params to prevent stale values
        for param_name, entry in list(self.param_entries.items()):
            if param_name not in self.params:
                if isinstance(entry, QLineEdit):
                    entry.clear()  # Clear text fields not in loaded params
                    
        # Update standard hyperparameters (text fields, dropdowns, checkboxes)
        for param_name, entry in list(self.param_entries.items()):
            if param_name in self.params:
                value = self.params[param_name]

                # Handle different widget types correctly
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

        # Update model parameters (ensure proper model-specific parameter refresh)
        if "MODEL_TYPE" in self.params:
            model_index = self.model_combo.findText(self.params["MODEL_TYPE"])
            if model_index != -1:
                self.model_combo.setCurrentIndex(model_index)

        # Ensure scheduler parameters update correctly
        if "SCHEDULER_TYPE" in self.params:
            scheduler_index = self.scheduler_combo.findText(self.params["SCHEDULER_TYPE"])
            if scheduler_index != -1:
                self.scheduler_combo.setCurrentIndex(scheduler_index)

        # Ensure training method and dependent fields update correctly
        if "TRAINING_METHOD" in self.params:
            method_index = self.training_method_combo.findText(self.params["TRAINING_METHOD"])
            if method_index != -1:
                self.training_method_combo.setCurrentIndex(method_index)
        
        # FIXED:Explicitly call update methods after setting combo boxes to ensure UI consistency
        # FIXED:especially if loaded params match current combo box text (so currentIndexChanged doesn't fire)
        self.update_model_params()
        self.update_scheduler_settings()
        self.update_training_method() # FIXED:This will also handle batch size visibility

        # CRITICAL FIX: Re-apply loaded parameters AFTER update methods that recreate widgets
        # update_model_params() and update_scheduler_settings() recreate widgets, losing loaded values
        # This second pass ensures RNN_LAYER_SIZES, LR params, etc. reflect loaded JSON values
        for param_name, entry in list(self.param_entries.items()):
            if param_name in self.params:
                value = self.params[param_name]
                if isinstance(entry, QLineEdit):
                    entry.setText(str(value))
                elif isinstance(entry, QComboBox):
                    index = entry.findText(str(value))
                    if index != -1:
                        entry.setCurrentIndex(index)

        # FIXED:Populate Max Training Time H, M, S fields from MAX_TRAINING_TIME_SECONDS
        if "EXPLOIT_REPETITIONS" in self.params:
            self.exploit_repetitions_entry.setText(str(self.params["EXPLOIT_REPETITIONS"]))
        if "EXPLOIT_LR" in self.params:
            self.exploit_lr_entry.setText(str(self.params["EXPLOIT_LR"]))
        
        # CRITICAL FIX: Explicitly update REPETITIONS (may be cleared by update methods)
        if "REPETITIONS" in self.params:
            if hasattr(self, 'repetitions_entry'):
                self.repetitions_entry.setText(str(self.params["REPETITIONS"]))
        
        # CRITICAL FIX: Explicitly update INITIAL_LR (may be cleared by update methods)
        if "INITIAL_LR" in self.params:
            if hasattr(self, 'initial_lr_entry'):
                self.initial_lr_entry.setText(str(self.params["INITIAL_LR"]))
            
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

            # SAFEGUARD: Validate that feature and target columns are not the same
            feature_target_valid, feature_target_error = self._validate_feature_target_columns(new_params)
            if not feature_target_valid:
                QMessageBox.critical(self, "Configuration Error", feature_target_error)
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

            # SAFEGUARD: Validate that feature and target columns are not the same
            feature_target_valid, feature_target_error = self._validate_feature_target_columns(new_params)
            if not feature_target_valid:
                QMessageBox.critical(self, "Configuration Error", feature_target_error)
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

        # Collect values from stored param entries - ONLY if visible and enabled
        for param, entry in self.param_entries.items():
            # Skip hidden or disabled fields to prevent collecting stale values
            if not entry.isVisible():
                continue
                
            if isinstance(entry, QLineEdit):
                if entry.isEnabled():  # Only collect from enabled text fields
                    value = entry.text().strip()
                    if value:  # Only add non-empty values
                        new_params[param] = value
            elif isinstance(entry, QComboBox):
                if entry.isEnabled():
                    new_params[param] = entry.currentText()
            elif isinstance(entry, QListWidget) and param == "OPTIMIZER_TYPE":
                selected_optimizers = [item.text() for item in entry.selectedItems()]
                new_params[param] = ",".join(selected_optimizers)
            elif isinstance(entry, QCheckBox):
                if entry.isVisible():  # Checkboxes can be visible but unchecked
                    new_params[param] = entry.isChecked()
            elif isinstance(entry, QListWidget):
                selected_items = [item.text() for item in entry.selectedItems()]
                if selected_items:  # Only add if items are selected
                    new_params[param] = selected_items

        # Conditionally remove inference filter params if 'None' is selected
        if new_params.get("INFERENCE_FILTER_TYPE") == "None":
            params_to_remove = [
                "INFERENCE_FILTER_WINDOW_SIZE",
                "INFERENCE_FILTER_ALPHA",
                "INFERENCE_FILTER_POLYORDER"
            ]
            for p in params_to_remove:
                if p in new_params:
                    del new_params[p]

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
            
            elif selected_scheduler == "CosineAnnealingWarmRestarts":
                if hasattr(self, 'cosine_t0_entry'):
                    t0_gui_value = self.cosine_t0_entry.text().strip()
                    self.logger.info(f"COSINE_T0 GUI field direct read: '{t0_gui_value}'")
                    self.logger.info(f"COSINE_T0 field visible: {self.cosine_t0_entry.isVisible()}")
                    self.logger.info(f"COSINE_T0 field enabled: {self.cosine_t0_entry.isEnabled()}")
                
                if hasattr(self, 'cosine_t_mult_entry'):
                    t_mult_gui_value = self.cosine_t_mult_entry.text().strip()
                    self.logger.info(f"COSINE_T_MULT GUI field direct read: '{t_mult_gui_value}'")
                    self.logger.info(f"COSINE_T_MULT field visible: {self.cosine_t_mult_entry.isVisible()}")
                    self.logger.info(f"COSINE_T_MULT field enabled: {self.cosine_t_mult_entry.isEnabled()}")
                
                if hasattr(self, 'cosine_eta_min_entry'):
                    eta_min_gui_value = self.cosine_eta_min_entry.text().strip()
                    self.logger.info(f"COSINE_ETA_MIN GUI field direct read: '{eta_min_gui_value}'")
                    self.logger.info(f"COSINE_ETA_MIN field visible: {self.cosine_eta_min_entry.isVisible()}")
                    self.logger.info(f"COSINE_ETA_MIN field enabled: {self.cosine_eta_min_entry.isEnabled()}")

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
            # FIXED: Remove LOOKBACK for FNN models - not applicable
            if "LOOKBACK" in new_params:
                del new_params["LOOKBACK"]
                self.logger.info("Removed LOOKBACK parameter for FNN model (not applicable)")
        
        # FIXED:Special handling for LSTM/GRU models - ensure compatible training method
        elif new_params.get("MODEL_TYPE") in ["LSTM", "GRU", "LSTM_EMA", "LSTM_LPF"]:
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

            # FIXED: Remove the individual time components as they're now consolidated
            keys_to_delete = [key for key in ["MAX_TRAIN_HOURS", "MAX_TRAIN_MINUTES", "MAX_TRAIN_SECONDS"] if key in params]
            for key in keys_to_delete:
                del params[key]

            self.logger.info(f"Converted time fields to total seconds: {total_seconds}")

        except ValueError as e:
            self.logger.warning(f"Error converting time fields: {e}. Setting total time to 0.")
            params["MAX_TRAINING_TIME_SECONDS"] = 0


    def on_param_text_changed(self, text=None):
        """Resets the style of a QLineEdit when its text is changed and validates critical fields."""
        sender = self.sender()
        if sender in self.error_fields:
            sender.setStyleSheet("")
            self.error_fields.remove(sender)
        
        # Map sender back to param key if possible
        changed_key = None
        for key, widget in self.param_entries.items():
            if widget is sender:
                changed_key = key
                break

        # Real-time validation: learning rates and general numeric fields
        if sender == getattr(self, 'initial_lr_entry', None):
            self._validate_lr_field(sender, "INITIAL_LR")
            return
        if sender == getattr(self, 'exploit_lr_entry', None):
            self._validate_lr_field(sender, "EXPLOIT_LR")
            return
        if sender == getattr(self, 'final_lr_entry', None):
            self._validate_lr_field(sender, "FINAL_LR", allow_zero=True)
            return

        # Apply generic numeric validation for other fields (e.g., MAX_EPOCHS, BATCH_SIZE)
        if isinstance(sender, QLineEdit) and changed_key:
            self._validate_numeric_field(sender, changed_key)
    
    def _validate_lr_field(self, field, field_name, allow_zero=False):
        """Validate a learning rate field and highlight if invalid."""
        text = field.text().strip()
        if not text:
            return  # Empty is okay, will be caught later if required
        
        # Skip validation for boundary format (Optuna)
        if text.startswith('[') and text.endswith(']'):
            field.setStyleSheet("")  # Clear any error styling
            return
        
        try:
            value = float(text)
            if value < 0 or (not allow_zero and value == 0):
                # Invalid: negative or zero (when not allowed)
                field.setStyleSheet("border: 2px solid #FF4444; background-color: #FFE6E6;")
                if value == 0:
                    field.setToolTip(f"⚠️ {field_name} cannot be zero! Training will fail. Enter a positive value like 0.001 or 1e-4.")
                else:
                    field.setToolTip(f"⚠️ {field_name} cannot be negative!")
                self.error_fields.add(field)
            else:
                # Valid
                field.setStyleSheet("")  # Clear error styling
                # Restore original tooltip
                if field_name == "INITIAL_LR":
                    field.setToolTip("The starting learning rate for the optimizer.")
                elif field_name == "EXPLOIT_LR":
                    field.setToolTip("Learning rate for exploitation phase (finetuning near convergence).")
                elif field_name == "FINAL_LR":
                    field.setToolTip("Target learning rate at end of scheduler (optional).")
        except ValueError:
            # Invalid format
            field.setStyleSheet("border: 2px solid #FF4444; background-color: #FFE6E6;")
            field.setToolTip(f"⚠️ {field_name} must be a valid number!")
            self.error_fields.add(field)

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

    def _validate_numeric_field(self, field: QLineEdit, field_name: str):
        """Generic numeric field validator with red highlight for accidental zeros/negatives.

        Rules:
        - Supports Optuna boundary format [min,max] (no highlight in that case)
        - For grid/single values, enforces per-field constraints (min/max, integer-only when required)
        - Highlights invalid input with red border and pale red background, sets helpful tooltip
        """
        text = field.text().strip()
        if text == "":
            # Empty handled elsewhere (required checks on proceed), do not flag yet
            field.setStyleSheet("")
            return

        # Skip validation for Optuna boundary format
        if text.startswith("[") and text.endswith("]"):
            field.setStyleSheet("")
            return

        # Define validation rules
        rules = {
            # Core training
            "MAX_EPOCHS": {"type": "int", "min": 1},
            "BATCH_SIZE": {"type": "int", "min": 1},
            "VALID_PATIENCE": {"type": "int", "min": 1},
            "VALID_FREQUENCY": {"type": "int", "min": 1},
            "REPETITIONS": {"type": "int", "min": 1},
            # Sequence & batching
            "LOOKBACK": {"type": "int", "min": 1},
            # Scheduler specifics
            "LR_PERIOD": {"type": "int", "min": 1},
            "LR_PARAM": {"type": "float", "min": 1e-12, "max": 1 - 1e-12},  # drop factor in (0,1)
            "PLATEAU_PATIENCE": {"type": "int", "min": 1},
            "PLATEAU_FACTOR": {"type": "float", "min": 1e-12, "max": 1 - 1e-12},
            "COSINE_T0": {"type": "int", "min": 1},
            "COSINE_T_MULT": {"type": "int", "min": 1},
            "COSINE_ETA_MIN": {"type": "float", "min": 0.0},
            # Device and data loading
            "NUM_WORKERS": {"type": "int", "min": 0},
            "PREFETCH_FACTOR": {"type": "int", "min": 1},
            # Regularization / dropout
            "WEIGHT_DECAY": {"type": "float", "min": 0.0},
            "FNN_DROPOUT_PROB": {"type": "float", "min": 0.0, "max": 1.0},
            # Exploit
            "EXPLOIT_EPOCHS": {"type": "int", "min": 0},
            "EXPLOIT_REPETITIONS": {"type": "int", "min": 0},
            # Inference filter params
            "INFERENCE_FILTER_WINDOW_SIZE": {"type": "int", "min": 1},
            "INFERENCE_FILTER_ALPHA": {"type": "float", "min": 0.0, "max": 1.0},
            "INFERENCE_FILTER_POLYORDER": {"type": "int", "min": 1},
            # Time fields (HH:MM:SS) can be zero but not negative and must be integers
            "MAX_TRAIN_HOURS": {"type": "int", "min": 0},
            "MAX_TRAIN_MINUTES": {"type": "int", "min": 0, "max": 59},
            "MAX_TRAIN_SECONDS": {"type": "int", "min": 0, "max": 59},
        }

        if field_name not in rules:
            # Not a numeric field we validate generically
            return

        rule = rules[field_name]
        want_int = rule.get("type") == "int"
        min_v = rule.get("min")
        max_v = rule.get("max")

        # Fields may contain comma-separated lists for grid; validate each token
        tokens = [t.strip() for t in text.replace(";", ",").split(",") if t.strip()]
        try:
            parsed = []
            for tok in tokens:
                val = int(tok) if want_int else float(tok)
                parsed.append(val)
                if min_v is not None and val < min_v:
                    raise ValueError(f"{field_name} value {val} is below minimum {min_v}")
                if max_v is not None and val > max_v:
                    raise ValueError(f"{field_name} value {val} exceeds maximum {max_v}")
        except ValueError:
            # Invalid number or out-of-range
            field.setStyleSheet("border: 2px solid #FF4444; background-color: #FFE6E6;")
            # Construct helpful tooltip
            limit_txt = []
            if min_v is not None:
                limit_txt.append(f">= {min_v}")
            if max_v is not None:
                limit_txt.append(f"<= {max_v}")
            lim = " and ".join(limit_txt)
            type_txt = "integer" if want_int else "number"
            field.setToolTip(f"⚠️ {field_name} must be a valid {type_txt}{' ' + lim if lim else ''}. Comma-separated lists allowed for grid search.")
            self.error_fields.add(field)
            return

        # If all tokens valid
        field.setStyleSheet("")
        field.setToolTip("")

    def _validate_feature_target_columns(self, params):
        """
        Validate that feature and target columns are properly configured and not overlapping.
        
        Returns:
            tuple: (is_valid, error_message)
        """
        try:
            feature_columns = params.get('FEATURE_COLUMNS', [])
            target_column = params.get('TARGET_COLUMN', '')
            
            # Check if target column is selected
            if not target_column or target_column.strip() == '':
                return False, "No target column selected.\n\nPlease select a target column before starting training."
            
            # Check if feature columns are selected
            if not feature_columns or len(feature_columns) == 0:
                return False, "No feature columns selected.\n\nPlease select at least one feature column before starting training."
            
            # SAFEGUARD: Check if target column is in feature columns
            if target_column in feature_columns:
                return False, (
                    f"Configuration error: Target column '{target_column}' cannot be used as a feature column.\n\n"
                    f"The target column is what the model tries to predict, so it cannot also be used as input.\n\n"
                    f"Please:\n"
                    f"• Remove '{target_column}' from the feature columns, OR\n"
                    f"• Select a different target column\n\n"
                    f"Current feature columns: {', '.join(feature_columns)}"
                )
            
            # Check for empty feature columns
            empty_features = [col for col in feature_columns if not col or col.strip() == '']
            if empty_features:
                return False, "Some feature columns are empty.\n\nPlease remove empty entries from the feature columns list."
            
            return True, ""
            
        except Exception as e:
            return False, f"Error validating columns: {str(e)}"
