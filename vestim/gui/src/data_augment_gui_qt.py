# ---------------------------------------------------------------------------------
# Author: Biswanath Dehury
# Date: `2025-04-14`
# Version: 1.0.0
# Description: 
# GUI for data augmentation - allows users to:
# 1. Resample data to desired frequency
# 2. Create new columns from existing columns using custom formulas
# 3. Pad data by prepending rows with specific values.
# ---------------------------------------------------------------------------------

from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QListWidget, QCheckBox, QLineEdit,
                            QComboBox, QTableWidget, QTableWidgetItem, QSizePolicy,
                            QFileDialog, QProgressBar, QWidget, QMessageBox, QDialog,
                            QFormLayout, QGroupBox, QSpinBox, QDoubleSpinBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject, QTimer

import os
import sys
import pandas as pd
import numpy as np

from vestim.gui.src.hyper_param_gui_qt import VEstimHyperParamGUI
from vestim.gateway.src.job_manager_qt import JobManager
from vestim.gateway.src.data_augment_manager_qt import DataAugmentManager

import logging
from vestim.logger_config import setup_logger

logger = setup_logger(log_file='data_augment.log')

class FormulaInputDialog(QDialog):
    """Dialog for entering custom formulas to create new columns"""
    def __init__(self, available_columns, session_created_column_names, parent=None):
        super().__init__(parent)
        self.available_columns = available_columns
        self.session_created_column_names = session_created_column_names
        self.formula = ""
        self.new_column_name = ""
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle("Create New Column")
        self.setMinimumWidth(500)
        layout = QVBoxLayout()
        name_layout = QHBoxLayout()
        name_label = QLabel("New Column Name:")
        name_layout.addWidget(name_label)
        self.column_name_edit = QLineEdit()
        name_layout.addWidget(self.column_name_edit)
        layout.addLayout(name_layout)
        columns_group = QGroupBox("Available Columns")
        columns_layout = QVBoxLayout()
        self.columns_list = QListWidget()
        self.columns_list.addItems(self.available_columns)
        self.columns_list.setSelectionMode(QListWidget.SingleSelection)
        self.columns_list.itemDoubleClicked.connect(self.add_column_to_formula)
        columns_layout.addWidget(self.columns_list)
        columns_group.setLayout(columns_layout)
        layout.addWidget(columns_group)
        formula_label = QLabel("Formula (Use column names, operators +,-,*,/, and functions like np.sin, np.cos, etc.):")
        layout.addWidget(formula_label)
        self.formula_edit = QLineEdit()
        layout.addWidget(self.formula_edit)
        examples_label = QLabel("Examples:\n"
                               "1. `column1 * 2 + column2`\n"
                               "2. `np.sin(column1) + np.log(column2)`\n"
                               "3. Lagged feature: `shift(column1, -1)`\n"
                               "4. Additive noise: `column1 + noise(0.0, 0.02)`\n"
                               "5. Multiplicative noise: `column1 * (1 + noise(0.0, 0.02))`\n"
                               "6. Moving average: `moving_average(column1, 10)`")
        examples_label.setStyleSheet("font-style: italic; color: gray;")
        layout.addWidget(examples_label)
        buttons_layout = QHBoxLayout()
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        buttons_layout.addWidget(self.cancel_button)
        self.submit_button = QPushButton("Create Column")
        self.submit_button.clicked.connect(self.accept_formula)
        self.submit_button.setStyleSheet("background-color: #0b6337; color: white;")
        buttons_layout.addWidget(self.submit_button)
        layout.addLayout(buttons_layout)
        self.setLayout(layout)
    
    def add_column_to_formula(self, item):
        current_text = self.formula_edit.text()
        if current_text and not current_text.endswith((' ', '+', '-', '*', '/', '(', ')')):
            current_text += ' + '
        self.formula_edit.setText(current_text + item.text())
    
    def accept_formula(self):
        new_name = self.column_name_edit.text().strip()
        if not new_name:
            QMessageBox.warning(self, "Input Error", "Please enter a name for the new column.")
            return
        if not self.formula_edit.text().strip():
            QMessageBox.warning(self, "Input Error", "Please enter a formula.")
            return
        if new_name in self.available_columns:
            QMessageBox.warning(self, "Name Conflict", f"Column name '{new_name}' already exists in the original data. Please choose a different name.")
            return
        if new_name in self.session_created_column_names:
            QMessageBox.warning(self, "Name Conflict", f"Column name '{new_name}' has already been defined in this session. Please choose a different name.")
            return
        self.new_column_name = new_name
        self.formula = self.formula_edit.text()
        logger.info(f"FormulaInputDialog: Accepting new column '{self.new_column_name}' with formula '{self.formula}'")
        self.accept()

class FilterInputDialog(QDialog):
    """Dialog for entering filter specifications."""
    def __init__(self, available_columns, parent=None):
        super().__init__(parent)
        self.available_columns = available_columns
        self.column_name = ""
        self.output_column_name = ""
        self.corner_frequency = 0.0
        self.sampling_rate = 1.0
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle("Filter Column")
        self.setMinimumWidth(400)
        layout = QVBoxLayout()
        form_layout = QFormLayout()
        self.column_combo = QComboBox()
        self.column_combo.addItems(self.available_columns)
        form_layout.addRow("Select Column:", self.column_combo)
        self.output_column_name_edit = QLineEdit()
        form_layout.addRow("New Column Name:", self.output_column_name_edit)
        self.filter_type_combo = QComboBox()
        self.filter_type_combo.addItems(["Butterworth", "Savitzky-Golay", "Exponential Moving Average"])
        form_layout.addRow("Filter Type:", self.filter_type_combo)
        self.sampling_rate_spinbox = QDoubleSpinBox()
        self.sampling_rate_spinbox.setRange(0.01, 10000.0)
        self.sampling_rate_spinbox.setValue(1.0)
        self.sampling_rate_spinbox.setSingleStep(1.0)
        form_layout.addRow("Sampling Rate (Hz):", self.sampling_rate_spinbox)
        self.filter_order_spinbox = QSpinBox()
        self.filter_order_spinbox.setRange(1, 10)
        self.filter_order_spinbox.setValue(4)
        self.filter_order_spinbox.setSingleStep(1)
        form_layout.addRow("Filter Order:", self.filter_order_spinbox)
        self.corner_frequency_spinbox = QDoubleSpinBox()
        self.corner_frequency_spinbox.setRange(0.0000001, 0.5)
        self.corner_frequency_spinbox.setValue(0.01)
        self.corner_frequency_spinbox.setSingleStep(0.0000001)
        self.corner_frequency_spinbox.setDecimals(7)
        form_layout.addRow("Corner Frequency (Hz):", self.corner_frequency_spinbox)
        layout.addLayout(form_layout)
        buttons_layout = QHBoxLayout()
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        buttons_layout.addWidget(self.cancel_button)
        self.submit_button = QPushButton("Apply Filter")
        self.submit_button.clicked.connect(self.accept_filter)
        self.submit_button.setStyleSheet("background-color: #0b6337; color: white;")
        buttons_layout.addWidget(self.submit_button)
        layout.addLayout(buttons_layout)
        self.setLayout(layout)
        
    def accept_filter(self):
        self.column_name = self.column_combo.currentText()
        self.output_column_name = self.output_column_name_edit.text().strip()
        self.corner_frequency = self.corner_frequency_spinbox.value()
        self.sampling_rate = self.sampling_rate_spinbox.value()
        self.filter_order = self.filter_order_spinbox.value()
        if not (1e-7 <= self.corner_frequency <= 0.5):
            QMessageBox.warning(self, "Input Error", "Corner frequency must be between 1e-7 Hz and 0.5 Hz.")
            return
        if not self.output_column_name:
            QMessageBox.warning(self, "Input Error", "Please enter a name for the new column.")
            return
        self.accept()

class AugmentationWorker(QObject):
    """Worker class for running data augmentation in a separate thread."""
    criticalError = pyqtSignal(str) 

    def __init__(self, data_augment_manager, job_folder, padding_length, resampling_frequency, column_formulas, normalize_data=False, filter_configs=None):
        super().__init__()
        self.data_augment_manager = data_augment_manager
        self.job_folder = job_folder
        self.padding_length = padding_length
        self.resampling_frequency = resampling_frequency
        self.column_formulas = column_formulas
        self.normalize_data = normalize_data
        self.filter_configs = filter_configs
        self.logger = logging.getLogger(__name__ + ".AugmentationWorker")

    def run(self):
        self.logger.info(f"AugmentationWorker started for job: {self.job_folder}")
        try:
            self.data_augment_manager.apply_augmentations(
                job_folder=self.job_folder,
                padding_length=self.padding_length,
                resampling_frequency=self.resampling_frequency,
                column_formulas=self.column_formulas,
                normalize_data=self.normalize_data,
                filter_configs=self.filter_configs
            )
        except Exception as e:
            self.logger.error(f"Critical error during augmentation task execution: {e}", exc_info=True)
            self.criticalError.emit(f"Critical augmentation failure: {e}")

class DataAugmentGUI(QMainWindow):
    augmentation_complete = pyqtSignal(pd.DataFrame)

    def __init__(self, job_manager=None, testing_mode=False, test_df=None, filter_configs=None):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.job_manager = job_manager
        self.testing_mode = testing_mode
        self.test_df_for_augmentation = test_df
        self.preloaded_filter_configs = filter_configs

        self.data_augment_manager = DataAugmentManager(job_manager=self.job_manager)
        self.augmentation_thread = None
        self.augmentation_worker = None
        self.hyper_param_gui = None

        if self.testing_mode:
            self.job_folder = None
            self.train_df = self.test_df_for_augmentation
        else:
            self.job_folder = self.job_manager.get_job_folder() if self.job_manager else None
            if self.job_folder:
                self.train_df = self.data_augment_manager.get_sample_train_dataframe(self.job_folder)
            else:
                self.train_df = None
        
        self.created_columns = []
        self.filter_configs = []
        self.settings_file = os.path.join(self.job_folder, "filter_settings.json") if self.job_folder else None
        self.last_used_settings_file = "defaults_templates/filter_settings_last_used.json"
        
        self.initUI()

        if self.testing_mode:
            self.prepopulate_for_testing()
        elif self.job_folder:
            self.load_filter_settings_last_used()

    def save_filter_settings(self):
        if self.settings_file and self.filter_configs:
            try:
                with open(self.settings_file, "w") as f:
                    json.dump(self.filter_configs, f)
            except Exception as e:
                self.logger.error(f"Error saving filter settings: {e}", exc_info=True)

    def save_filter_settings_last_used(self):
        if self.last_used_settings_file and self.filter_configs:
            try:
                os.makedirs(os.path.dirname(self.last_used_settings_file), exist_ok=True)
                with open(self.last_used_settings_file, "w") as f:
                    json.dump(self.filter_configs, f)
                self.logger.info(f"Saved last used filter settings to {self.last_used_settings_file}")
            except Exception as e:
                self.logger.error(f"Error saving last used filter settings: {e}", exc_info=True)

    def load_filter_settings(self):
        if self.settings_file and os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, "r") as f:
                    self.filter_configs = json.load(f)
                for config in self.filter_configs:
                    self.filter_list.addItem(f"Filter '{config['column']}' at {config['corner_frequency']}Hz (Fs={config['sampling_rate']}Hz)")
                    self.remove_filter_button.setEnabled(True)
            except Exception as e:
                self.logger.error(f"Error loading filter settings: {e}", exc_info=True)

    def initUI(self):
        self.setWindowTitle("VEstim Data Augmentation")
        self.setGeometry(100, 100, 1200, 800)
        self.setMouseTracking(True)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.setStyleSheet("QPushButton:disabled { background-color: #d3d3d3 !important; color: #a9a9a9 !important; }")

        self.header_label = QLabel("Data Augmentation, Padding, and Resampling", self)
        self.header_label.setAlignment(Qt.AlignCenter)
        self.header_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #0b6337; margin-bottom: 15px;")
        self.main_layout.addWidget(self.header_label)

        if not self.job_folder and not self.testing_mode:
            job_folder_layout = QHBoxLayout()
            self.job_folder_label = QLabel("Select Job Folder:", self)
            job_folder_layout.addWidget(self.job_folder_label)
            self.job_folder_button = QPushButton("Browse...", self)
            self.job_folder_button.clicked.connect(self.select_job_folder)
            job_folder_layout.addWidget(self.job_folder_button)
            self.job_folder_path_label = QLabel("No folder selected", self)
            job_folder_layout.addWidget(self.job_folder_path_label)
            self.main_layout.addLayout(job_folder_layout)

        top_row_layout = QHBoxLayout()
        group_box_style = "QGroupBox { font-size: 10pt; font-weight: bold; }"

        filtering_group = QGroupBox("Data Filtering")
        filtering_group.setStyleSheet(group_box_style)
        filtering_layout = QVBoxLayout()
        self.filtering_checkbox = QCheckBox("Enable data filtering")
        self.filtering_checkbox.setChecked(True)
        self.filtering_checkbox.setToolTip("Apply a Butterworth filter to a selected column.")
        self.filtering_checkbox.stateChanged.connect(self.toggle_filtering_options)
        filtering_layout.addWidget(self.filtering_checkbox)
        self.add_filter_button = QPushButton("Add Filter")
        self.add_filter_button.clicked.connect(self.show_filter_dialog)
        self.add_filter_button.setEnabled(False)
        filtering_layout.addWidget(self.add_filter_button)
        self.filter_list_label = QLabel("Applied Filters:")
        filtering_layout.addWidget(self.filter_list_label)
        self.filter_list = QListWidget()
        self.filter_list.setMinimumHeight(80)
        self.filter_list.itemDoubleClicked.connect(self.edit_filter)
        filtering_layout.addWidget(self.filter_list)
        self.remove_filter_button = QPushButton("Remove Selected Filter")
        self.remove_filter_button.clicked.connect(self.remove_filter)
        self.remove_filter_button.setEnabled(False)
        filtering_layout.addWidget(self.remove_filter_button)
        filtering_group.setLayout(filtering_layout)
        top_row_layout.addWidget(filtering_group)

        augmentation_group = QGroupBox("Column Creation")
        augmentation_group.setStyleSheet(group_box_style)
        augmentation_layout = QVBoxLayout()
        self.column_creation_checkbox = QCheckBox("Create new columns from existing data")
        self.column_creation_checkbox.setToolTip("Create derived features using mathematical formulas.")
        self.column_creation_checkbox.stateChanged.connect(self.toggle_column_creation)
        augmentation_layout.addWidget(self.column_creation_checkbox)
        self.add_formula_button = QPushButton("Add Column Formula")
        self.add_formula_button.clicked.connect(self.show_formula_dialog)
        self.add_formula_button.setEnabled(False)
        augmentation_layout.addWidget(self.add_formula_button)
        self.formula_list_label = QLabel("Created Columns:")
        augmentation_layout.addWidget(self.formula_list_label)
        self.formula_list = QListWidget()
        self.formula_list.setMinimumHeight(80)
        augmentation_layout.addWidget(self.formula_list)
        self.remove_formula_button = QPushButton("Remove Selected Column")
        self.remove_formula_button.clicked.connect(self.remove_formula)
        self.remove_formula_button.setEnabled(False)
        augmentation_layout.addWidget(self.remove_formula_button)
        augmentation_group.setLayout(augmentation_layout)
        top_row_layout.addWidget(augmentation_group)
        
        self.main_layout.addLayout(top_row_layout)

        bottom_row_layout = QHBoxLayout()

        resampling_group = QGroupBox("Data Resampling")
        resampling_group.setStyleSheet(group_box_style)
        resampling_group.setMinimumHeight(120)
        resampling_layout = QVBoxLayout()
        self.resampling_checkbox = QCheckBox("Enable data resampling")
        self.resampling_checkbox.setToolTip("Resamples time series data to a different frequency.")
        self.resampling_checkbox.stateChanged.connect(self.toggle_resampling_options)
        resampling_layout.addWidget(self.resampling_checkbox)
        frequency_layout = QHBoxLayout()
        frequency_label = QLabel("Resampling Frequency:")
        frequency_layout.addWidget(frequency_label)
        self.frequency_combo = QComboBox()
        self.frequency_combo.addItems(["0.1Hz", "0.5Hz", "1Hz", "5Hz", "10Hz"])
        self.frequency_combo.setEnabled(False)
        frequency_layout.addWidget(self.frequency_combo)
        resampling_layout.addLayout(frequency_layout)
        resampling_group.setLayout(resampling_layout)
        bottom_row_layout.addWidget(resampling_group)

        padding_group = QGroupBox("Data Padding (Prepend)")
        padding_group.setStyleSheet(group_box_style)
        padding_group.setMinimumHeight(120)
        padding_layout = QVBoxLayout()
        self.padding_checkbox = QCheckBox("Enable data padding")
        self.padding_checkbox.setToolTip("Prepends rows with zeros to the beginning of the dataset.")
        self.padding_checkbox.stateChanged.connect(self.toggle_padding_options)
        padding_layout.addWidget(self.padding_checkbox)
        padding_length_layout = QHBoxLayout()
        padding_length_label = QLabel("Padding Length (rows):")
        padding_length_layout.addWidget(padding_length_label)
        self.padding_length_spinbox = QSpinBox()
        self.padding_length_spinbox.setRange(0, 10000)
        self.padding_length_spinbox.setValue(10)
        self.padding_length_spinbox.setEnabled(False)
        padding_length_layout.addWidget(self.padding_length_spinbox)
        padding_layout.addLayout(padding_length_layout)
        padding_group.setLayout(padding_layout)
        bottom_row_layout.addWidget(padding_group)

        self.main_layout.addLayout(bottom_row_layout)

        normalization_group = QGroupBox("Data Normalization")
        normalization_group.setStyleSheet(group_box_style)
        normalization_group.setMinimumHeight(120)
        normalization_layout = QVBoxLayout()
        self.normalization_checkbox = QCheckBox("Enable data normalization (Min-Max scaling)")
        self.normalization_checkbox.setChecked(True)
        self.normalization_checkbox.setToolTip("Applies Min-Max scaling (0-1 range) to numeric columns.")
        normalization_layout.addWidget(self.normalization_checkbox)
        
        normalization_note_label = QLabel("Note: Numeric columns will be scaled. Time-related and ID-like columns are typically excluded automatically.")
        normalization_note_label.setStyleSheet("font-style: italic; color: gray;")
        normalization_note_label.setWordWrap(True)
        normalization_layout.addWidget(normalization_note_label)

        normalization_group.setLayout(normalization_layout)
        self.main_layout.addWidget(normalization_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.main_layout.addWidget(self.progress_bar)
        
        buttons_layout = QHBoxLayout()
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setMinimumHeight(35)
        self.cancel_button.setStyleSheet("""
            QPushButton { font-size: 10pt !important; background-color: #f0f0f0 !important; border: 2px solid #cccccc !important; border-radius: 6px !important; padding: 8px 16px !important; color: #333333 !important; }
            QPushButton:hover { background-color: #e0e0e0 !important; border: 2px solid #999999 !important; }
            QPushButton:pressed { background-color: #d0d0d0 !important; border: 2px solid #777777 !important; }
        """)
        self.cancel_button.setAttribute(Qt.WA_Hover, True)
        self.cancel_button.clicked.connect(self.close_gui)
        buttons_layout.addWidget(self.cancel_button)
        self.apply_button = QPushButton("Apply Changes and Continue")
        self.apply_button.setMinimumHeight(35)
        self.apply_button.clicked.connect(self.apply_changes)
        self.apply_button.setStyleSheet("""
            QPushButton { background-color: #0b6337 !important; color: white !important; font-size: 10pt !important; border: 2px solid #0b6337 !important; border-radius: 6px !important; font-weight: bold !important; padding: 8px 16px !important; }
            QPushButton:hover { background-color: #094D2A !important; border: 2px solid #094D2A !important; }
            QPushButton:pressed { background-color: #073A20 !important; border: 2px solid #073A20 !important; }
        """)
        self.apply_button.setAttribute(Qt.WA_Hover, True)
        self.apply_button.setEnabled(self.train_df is not None)
        buttons_layout.addWidget(self.apply_button)
        self.main_layout.addLayout(buttons_layout)

        if self.train_df is not None:
            self.padding_checkbox.setEnabled(True)
            self.resampling_checkbox.setEnabled(True)
            self.column_creation_checkbox.setEnabled(True)
            self.filtering_checkbox.setEnabled(True)
            self.normalization_checkbox.setEnabled(True)
        else:
            self.padding_checkbox.setEnabled(False)
            self.resampling_checkbox.setEnabled(False)
            self.column_creation_checkbox.setEnabled(False)
            self.filtering_checkbox.setEnabled(False)
            self.normalization_checkbox.setEnabled(False)
        
        if self.job_folder:
            self.load_filter_settings()
    
    def close_gui(self):
        if not self.testing_mode:
            self.save_filter_settings()
        self.close()
    
    def select_job_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Job Folder")
        if folder:
            self.job_folder = folder
            self.job_folder_path_label.setText(folder)
            try:
                self.logger.info(f"Loading sample train data for GUI from job folder: {self.job_folder}")
                sample_train_df = self.data_augment_manager.get_sample_train_dataframe(self.job_folder)
                if sample_train_df is not None and not sample_train_df.empty:
                    self.train_df = sample_train_df
                    self.apply_button.setEnabled(True)
                    QMessageBox.information(self, "Success", "Job folder selected and sample data schema loaded successfully!")
                    self.padding_checkbox.setEnabled(True)
                    self.resampling_checkbox.setEnabled(True)
                    self.column_creation_checkbox.setEnabled(True)
                    self.filtering_checkbox.setEnabled(True)
                    self.normalization_checkbox.setEnabled(True)
                else:
                    self.train_df = None
                    self.apply_button.setEnabled(False)
                    self.padding_checkbox.setEnabled(False)
                    self.resampling_checkbox.setEnabled(False)
                    self.column_creation_checkbox.setEnabled(False)
                    self.filtering_checkbox.setEnabled(False)
                    self.normalization_checkbox.setEnabled(False)
                    QMessageBox.warning(self, "Warning", "Could not load a sample data file from the train/processed_data directory.")
            except Exception as e:
                self.logger.error(f"Error loading sample data for GUI: {e}", exc_info=True)
                self.train_df = None
                self.apply_button.setEnabled(False)
                QMessageBox.critical(self, "Error", f"Could not load sample data schema: {e}")

    def toggle_padding_options(self, state):
        self.padding_length_spinbox.setEnabled(state == Qt.Checked)

    def toggle_resampling_options(self, state):
        self.frequency_combo.setEnabled(state == Qt.Checked)
    
    def toggle_column_creation(self, state):
        self.add_formula_button.setEnabled(state == Qt.Checked)
        self.remove_formula_button.setEnabled(state == Qt.Checked and self.formula_list.count() > 0)
    
    def toggle_filtering_options(self, state):
        self.add_filter_button.setEnabled(state == Qt.Checked)
        self.remove_filter_button.setEnabled(state == Qt.Checked and self.filter_list.count() > 0)

    def show_formula_dialog(self):
        if self.train_df is None:
            if self.job_folder:
                self.train_df = self.data_augment_manager.get_sample_train_dataframe(self.job_folder)
            if self.train_df is None:
                QMessageBox.warning(self, "Warning", "Please select a valid job folder with data to see available columns.")
                return

        available_columns = list(self.train_df.columns)
        session_created_names = [name for name, formula in self.created_columns]
        
        dialog = FormulaInputDialog(available_columns, session_created_names, self)
        if dialog.exec_() == QDialog.Accepted:
            new_column, formula = dialog.new_column_name, dialog.formula
            self.formula_list.addItem(f"{new_column} = {formula}")
            self.created_columns.append((new_column, formula))
            self.remove_formula_button.setEnabled(True)

    def remove_formula(self):
        selected_items = self.formula_list.selectedItems()
        if not selected_items: return
        row = self.formula_list.row(selected_items[0])
        self.formula_list.takeItem(row)
        self.created_columns.pop(row)
        if self.formula_list.count() == 0: self.remove_formula_button.setEnabled(False)
    
    def show_filter_dialog(self):
        if self.train_df is None:
            QMessageBox.warning(self, "Warning", "Please load data first.")
            return
        numeric_columns = list(self.train_df.select_dtypes(include=np.number).columns)
        exclude_names = ['time', 'timestamp', 'status']
        columns_for_filter = [col for col in numeric_columns if col.lower() not in exclude_names]

        if not columns_for_filter:
            QMessageBox.warning(self, "No Filterable Columns", "No suitable numeric columns are available for filtering.")
            return
            
        dialog = FilterInputDialog(columns_for_filter, self)
        if dialog.exec_() == QDialog.Accepted:
            column_name, output_column_name, corner_frequency, sampling_rate, filter_order = dialog.column_name, dialog.output_column_name, dialog.corner_frequency, dialog.sampling_rate, dialog.filter_order

            try:
                self.train_df = self.data_augment_manager.service.apply_butterworth_filter(
                    self.train_df, column_name, corner_frequency, sampling_rate, filter_order, output_column_name
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not apply filter: {e}")
            else:
                self.filter_list.addItem(f"Order {filter_order} Filter '{column_name}' at {corner_frequency}Hz (Fs={sampling_rate}Hz) -> {output_column_name}")
                self.filter_configs.append({"column": column_name, "output_column_name": output_column_name, "corner_frequency": corner_frequency, "sampling_rate": sampling_rate, "filter_order": filter_order})
                self.remove_filter_button.setEnabled(True)

    def remove_filter(self):
        selected_items = self.filter_list.selectedItems()
        if not selected_items: return
        row = self.filter_list.row(selected_items[0])
        self.filter_list.takeItem(row)
        self.filter_configs.pop(row)
        if self.filter_list.count() == 0: self.remove_filter_button.setEnabled(False)

    def edit_filter(self, item):
        row = self.filter_list.row(item)
        if row < 0 or row >= len(self.filter_configs): return

        config = self.filter_configs[row]
        numeric_columns = list(self.train_df.select_dtypes(include=np.number).columns)
        exclude_names = ['time', 'timestamp', 'status']
        columns_for_filter = [col for col in numeric_columns if col.lower() not in exclude_names]

        dialog = FilterInputDialog(columns_for_filter, self)
        dialog.column_combo.setCurrentText(config["column"])
        dialog.output_column_name_edit.setText(config.get("output_column_name", ""))
        dialog.corner_frequency_spinbox.setValue(config["corner_frequency"])
        dialog.sampling_rate_spinbox.setValue(config["sampling_rate"])
        dialog.filter_order_spinbox.setValue(config.get("filter_order", 4))

        if dialog.exec_() == QDialog.Accepted:
            self.filter_configs[row] = {
                "column": dialog.column_name,
                "output_column_name": dialog.output_column_name,
                "corner_frequency": dialog.corner_frequency,
                "sampling_rate": dialog.sampling_rate,
                "filter_order": dialog.filter_order
            }
            self.filter_list.item(row).setText(f"Order {dialog.filter_order} Filter '{dialog.column_name}' at {dialog.corner_frequency}Hz (Fs={dialog.sampling_rate}Hz) -> {dialog.output_column_name}")
            try:
                self.train_df = self.data_augment_manager.service.apply_butterworth_filter(
                    self.train_df, dialog.column_name, dialog.corner_frequency, dialog.sampling_rate, dialog.filter_order, dialog.output_column_name
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not apply filter: {e}")

    def apply_changes(self):
        if self.testing_mode:
            self.apply_changes_for_testing()
        else:
            self.apply_changes_for_training()

    def apply_changes_for_testing(self):
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        df = self.test_df_for_augmentation.copy()
        if self.filter_configs:
            for config in self.filter_configs:
                df = self.data_augment_manager.service.apply_butterworth_filter(
                    df,
                    column_name=config['column'],
                    corner_frequency=config['corner_frequency'],
                    sampling_rate=config['sampling_rate'],
                    filter_order=config.get('filter_order', 4),
                    output_column_name=config.get('output_column_name')
                )
        self.augmentation_complete.emit(df)
        self.close()

    def apply_changes_for_training(self):
        if self.job_folder is None:
            QMessageBox.warning(self, "Warning", "Please select a job folder first.")
            return
        self.save_filter_settings_last_used()
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.apply_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        
        if hasattr(self, 'status_label') and self.status_label is not None:
            self.status_label.deleteLater()
        self.status_label = QLabel("Processing data... Please wait.")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: blue; font-weight: bold; font-size: 14px;")
        self.main_layout.insertWidget(self.main_layout.indexOf(self.progress_bar), self.status_label)

        padding_length = self.padding_length_spinbox.value() if self.padding_checkbox.isChecked() else 0
        resampling_frequency = self.frequency_combo.currentText() if self.resampling_checkbox.isChecked() else None
        column_formulas = self.created_columns if self.column_creation_checkbox.isChecked() else None
        filter_configs = self.filter_configs if self.filter_configs else None
        normalize_data_flag = self.normalization_checkbox.isChecked()

        self.augmentation_thread = QThread()
        self.augmentation_worker = AugmentationWorker(
            self.data_augment_manager, self.job_folder, padding_length, resampling_frequency,
            column_formulas, normalize_data_flag, filter_configs
        )
        self.augmentation_worker.moveToThread(self.augmentation_thread)
        self.data_augment_manager.augmentationProgress.connect(self.handle_augmentation_progress)
        self.data_augment_manager.augmentationFinished.connect(self.handle_augmentation_finished)
        self.data_augment_manager.formulaErrorOccurred.connect(self.handle_formula_error)
        self.augmentation_worker.criticalError.connect(self.handle_critical_error)
        self.augmentation_thread.started.connect(self.augmentation_worker.run)
        self.augmentation_thread.finished.connect(self.augmentation_thread.deleteLater)
        self.data_augment_manager.augmentationFinished.connect(self.augmentation_thread.quit)
        self.data_augment_manager.formulaErrorOccurred.connect(self.augmentation_thread.quit)
        self.augmentation_worker.criticalError.connect(self.augmentation_thread.quit)
        self.augmentation_thread.start()

    def handle_augmentation_progress(self, value):
        self.progress_bar.setValue(value)

    def handle_augmentation_finished(self, job_folder, processed_files_metadata):
        failures = [f for f in processed_files_metadata if f.get('status') == 'Failed']
        if not failures:
            self.progress_bar.setValue(100)
            if hasattr(self, 'status_label'): self.status_label.setText("Processing complete!")
            QMessageBox.information(self, "Success", "Data augmentation completed successfully!")
            self.apply_button.setText("Continue to Hyperparameter Selection")
            try: self.apply_button.clicked.disconnect(self.apply_changes)
            except TypeError: pass
            try: self.apply_button.clicked.disconnect(self.go_to_hyperparameter_gui) 
            except TypeError: pass 
            self.apply_button.clicked.connect(self.go_to_hyperparameter_gui)
        else:
            error_summary = f"Augmentation finished, but {len(failures)} file(s) failed:\n"
            for fail in failures[:5]:
                 error_summary += f"- {os.path.basename(fail.get('filepath','Unknown'))}: {fail.get('error', 'Unknown error')}\n"
            if len(failures) > 5: error_summary += "- ... (see logs for more details)"
            self.progress_bar.setValue(100) 
            if hasattr(self, 'status_label'): self.status_label.setText("Processing finished with some errors.")
            QMessageBox.warning(self, "Processing Finished with Errors", error_summary)
            self.apply_button.setText("Apply Changes and Continue")
        self.apply_button.setEnabled(True)
        self.cancel_button.setEnabled(True)

    def handle_critical_error(self, error_msg):
        self.progress_bar.setVisible(False) 
        if hasattr(self, 'status_label'): self.status_label.setText(f"Critical Error: {error_msg}")
        QMessageBox.critical(self, "Critical Error", f"A critical error occurred: {error_msg}")
        self.apply_button.setEnabled(True)
        self.cancel_button.setEnabled(True)

    def handle_formula_error(self, error_msg):
        self.progress_bar.setVisible(False) 
        if hasattr(self, 'status_label'): self.status_label.setText("Formula Error: Processing stopped.")
        QMessageBox.critical(self, "Formula Error", error_msg) 
        self.apply_button.setEnabled(True)
        self.cancel_button.setEnabled(True)
    
    def go_to_hyperparameter_gui(self):
        try:
            self._next_hyper_param_gui = VEstimHyperParamGUI(job_manager=self.job_manager)
            QTimer.singleShot(0, self._execute_gui_transition)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not prepare hyperparameter selection: {e}")
            self.apply_button.setEnabled(True)
            self.cancel_button.setEnabled(True)

    def _execute_gui_transition(self):
        try:
            if hasattr(self, '_next_hyper_param_gui') and self._next_hyper_param_gui:
                self._next_hyper_param_gui.show()
                self.hyper_param_gui = self._next_hyper_param_gui
            self.close()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not complete transition: {e}")
            self.apply_button.setEnabled(True)
            self.cancel_button.setEnabled(True)

    def load_filter_settings_last_used(self):
        if not (self.last_used_settings_file and os.path.exists(self.last_used_settings_file)):
            return
        try:
            with open(self.last_used_settings_file, "r") as f:
                loaded_settings = json.load(f)
        except Exception as e:
            self.logger.error(f"Error reading last used filter settings file: {e}")
            return

        if not self.train_df is None:
            available_columns = self.train_df.columns.tolist()
            self.filter_list.clear()
            self.filter_configs.clear()
            for setting in loaded_settings:
                if setting.get("column") in available_columns:
                    try:
                        self.train_df = self.data_augment_manager.service.apply_butterworth_filter(
                            self.train_df, **setting
                        )
                        self.filter_configs.append(setting)
                        self.filter_list.addItem(f"Order {setting['filter_order']} Filter '{setting['column']}' at {setting['corner_frequency']}Hz (Fs={setting['sampling_rate']}Hz) -> {setting['output_column_name']}")
                    except Exception as e:
                        self.logger.error(f"Error applying loaded filter for column '{setting['column']}': {e}")
            if self.filter_list.count() > 0:
                self.remove_filter_button.setEnabled(True)

    def prepopulate_for_testing(self):
        self.setWindowTitle("VEstim Data Augmentation (Testing Mode)")
        self.apply_button.setText("Apply Required Augmentations and Continue Test")

        # Disable all augmentation options except the mandatory filtering
        self.resampling_checkbox.setChecked(False)
        self.resampling_checkbox.setEnabled(False)
        self.padding_checkbox.setChecked(False)
        self.padding_checkbox.setEnabled(False)
        self.column_creation_checkbox.setChecked(False)
        self.column_creation_checkbox.setEnabled(False)
        self.normalization_checkbox.setChecked(False)
        self.normalization_checkbox.setEnabled(False)
        self.add_formula_button.setEnabled(False)
        self.remove_formula_button.setEnabled(False)

        # The filtering is required, so check the box but disable it so the user can't change it.
        self.filtering_checkbox.setChecked(True)
        self.filtering_checkbox.setEnabled(False)

        # The user cannot add or remove the required filters.
        self.add_filter_button.setEnabled(False)
        self.remove_filter_button.setEnabled(False)

        # Populate the list with the required filters from the original job.
        self.filter_configs = self.preloaded_filter_configs if self.preloaded_filter_configs is not None else []
        self.filter_list.clear()
        if not self.filter_configs:
            self.logger.warning("DataAugmentGUI launched in testing mode but no filter configs were provided.")
        
        for config in self.filter_configs:
            self.filter_list.addItem(f"Order {config['filter_order']} Filter '{config['column']}' at {config['corner_frequency']}Hz (Fs={config['sampling_rate']}Hz) -> {config['output_column_name']}")

def main():
    app = QApplication(sys.argv)
    ex = DataAugmentGUI()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
