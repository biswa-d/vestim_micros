# ---------------------------------------------------------------------------------
# Author: Biswanath Dehury
# Date: `2025-04-14`
# Version: 1.0.0
# Description: 
# GUI for data augmentation - allows users to:
# 1. Resample data to desired frequency (functionality moved from data_import_gui_qt_test.py)
# 2. Create new columns from existing columns using custom formulas
# 3. Pad data by prepending rows with specific values.
# 
# Next Steps:
# 1. Create data visualization to preview augmented data
# 2. Add more predefined formula templates
# ---------------------------------------------------------------------------------

from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QListWidget, QCheckBox, QLineEdit,
                            QComboBox, QTableWidget, QTableWidgetItem, QSizePolicy,
                            QFileDialog, QProgressBar, QWidget, QMessageBox, QDialog,
                            QFormLayout, QGroupBox, QSpinBox, QDoubleSpinBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject

import os
import sys
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt # PreviewDialog removed, so this might not be needed
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas # PreviewDialog removed

from vestim.gui.src.hyper_param_gui_qt import VEstimHyperParamGUI
from vestim.gateway.src.data_augment_manager_qt import DataAugmentManager

import logging
from vestim.logger_config import setup_logger

# Set up initial logging to a default log file
logger = setup_logger(log_file='data_augment.log')

class FormulaInputDialog(QDialog):
    """Dialog for entering custom formulas to create new columns"""
    def __init__(self, available_columns, session_created_column_names, parent=None): # Added session_created_column_names
        super().__init__(parent)
        self.available_columns = available_columns
        self.session_created_column_names = session_created_column_names # Store it
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
                               "1. column1 * 2 + column2\n"
                               "2. np.sin(column1) + np.log(column2)\n"
                               "3. (column1 - column2) / column3")
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

        # Check for duplicate column names
        if new_name in self.available_columns:
            QMessageBox.warning(self, "Name Conflict", f"Column name '{new_name}' already exists in the original data. Please choose a different name.")
            return
        if new_name in self.session_created_column_names:
            QMessageBox.warning(self, "Name Conflict", f"Column name '{new_name}' has already been defined in this session. Please choose a different name.")
            return
        
        self.new_column_name = new_name
        self.formula = self.formula_edit.text()
        # Log before accepting
        logger.info(f"FormulaInputDialog: Accepting new column '{self.new_column_name}' with formula '{self.formula}'")
        self.accept()

class AugmentationWorker(QObject):
    """Worker class for running data augmentation in a separate thread."""
    # Removed progress and finished signals, manager handles these.
    # Keep an error signal for critical failures within the worker's run method itself.
    criticalError = pyqtSignal(str) 

    def __init__(self, data_augment_manager, job_folder, padding_length, resampling_frequency, column_formulas): # Added padding_length
        super().__init__()
        self.data_augment_manager = data_augment_manager
        self.job_folder = job_folder
        self.padding_length = padding_length # Store padding_length
        self.resampling_frequency = resampling_frequency
        self.column_formulas = column_formulas
        self.logger = logging.getLogger(__name__ + ".AugmentationWorker")

    def run(self):
        self.logger.info(f"AugmentationWorker started for job: {self.job_folder}")
        try:
            # Call the manager's method. It will emit its own progress/finished/formulaError signals.
            # We don't need the progress_callback here anymore.
            # The return value (folder_path, metadata) is handled by the manager's finished signal.
            self.data_augment_manager.apply_augmentations(
                job_folder=self.job_folder,
                padding_length=self.padding_length, # Pass padding_length
                resampling_frequency=self.resampling_frequency,
                column_formulas=self.column_formulas
                # No progress_callback needed here
            )
            # If apply_augmentations completes without raising an exception, the worker's job is done.
            # The manager's 'augmentationFinished' signal will notify the GUI.
        except Exception as e:
            # This catches critical errors if apply_augmentations itself fails badly.
            self.logger.error(f"Critical error during augmentation task execution: {e}", exc_info=True)
            self.criticalError.emit(f"Critical augmentation failure: {e}")

class DataAugmentGUI(QMainWindow):
    def __init__(self, job_folder=None):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.job_folder = job_folder
        self.data_augment_manager = DataAugmentManager()
        self.augmentation_thread = None
        self.augmentation_worker = None
        self.hyper_param_gui = None 
        
        if self.job_folder:
            self.logger.info(f"DataAugmentGUI initialized with job_folder: {self.job_folder}. Loading sample data.")
            try:
                sample_train_df = self.data_augment_manager.get_sample_train_dataframe(self.job_folder)
                if sample_train_df is not None and not sample_train_df.empty:
                    self.train_df = sample_train_df
                    self.test_df = None 
                    self.logger.info("Sample train data loaded successfully in __init__.")
                else:
                    self.train_df = None
                    self.test_df = None
                    self.logger.warning(f"Could not load sample train data in __init__ for {self.job_folder}.")
            except Exception as e:
                self.logger.error(f"Error loading sample data in __init__ for {self.job_folder}: {e}", exc_info=True)
                self.train_df = None
                self.test_df = None
        else:
            self.train_df = None
            self.test_df = None
        
        self.created_columns = []
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle("VEstim Data Augmentation")
        self.setGeometry(100, 100, 900, 650) # Increased height for padding section
        
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        self.header_label = QLabel("Data Augmentation, Padding, and Resampling", self)
        self.header_label.setAlignment(Qt.AlignCenter)
        self.header_label.setStyleSheet("font-size: 18px; font-weight: bold; color: green;")
        self.main_layout.addWidget(self.header_label)
        
        if not self.job_folder:
            job_folder_layout = QHBoxLayout()
            self.job_folder_label = QLabel("Select Job Folder:", self)
            job_folder_layout.addWidget(self.job_folder_label)
            self.job_folder_button = QPushButton("Browse...", self)
            self.job_folder_button.clicked.connect(self.select_job_folder)
            job_folder_layout.addWidget(self.job_folder_button)
            self.job_folder_path_label = QLabel("No folder selected", self)
            job_folder_layout.addWidget(self.job_folder_path_label)
            self.main_layout.addLayout(job_folder_layout)

        # Padding Group
        padding_group = QGroupBox("Data Padding (Prepend)")
        padding_layout = QVBoxLayout()
        self.padding_checkbox = QCheckBox("Enable data padding")
        self.padding_checkbox.stateChanged.connect(self.toggle_padding_options)
        padding_layout.addWidget(self.padding_checkbox)
        padding_length_layout = QHBoxLayout()
        padding_length_label = QLabel("Padding Length (rows):")
        padding_length_layout.addWidget(padding_length_label)
        self.padding_length_spinbox = QSpinBox()
        self.padding_length_spinbox.setRange(0, 1000) # Sensible range
        self.padding_length_spinbox.setValue(10)      # Default value
        self.padding_length_spinbox.setEnabled(False)
        padding_length_layout.addWidget(self.padding_length_spinbox)
        padding_layout.addLayout(padding_length_layout)
        padding_group.setLayout(padding_layout)
        self.main_layout.addWidget(padding_group)
        
        resampling_group = QGroupBox("Data Resampling")
        resampling_layout = QVBoxLayout()
        self.resampling_checkbox = QCheckBox("Enable data resampling")
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
        self.main_layout.addWidget(resampling_group)
        
        augmentation_group = QGroupBox("Column Creation")
        augmentation_layout = QVBoxLayout()
        self.column_creation_checkbox = QCheckBox("Create new columns from existing data")
        self.column_creation_checkbox.stateChanged.connect(self.toggle_column_creation)
        augmentation_layout.addWidget(self.column_creation_checkbox)
        self.add_formula_button = QPushButton("Add Column Formula")
        self.add_formula_button.clicked.connect(self.show_formula_dialog)
        self.add_formula_button.setEnabled(False)
        augmentation_layout.addWidget(self.add_formula_button)
        self.formula_list_label = QLabel("Created Columns:")
        augmentation_layout.addWidget(self.formula_list_label)
        self.formula_list = QListWidget()
        self.formula_list.setMinimumHeight(100) # Adjusted height
        augmentation_layout.addWidget(self.formula_list)
        self.remove_formula_button = QPushButton("Remove Selected Column")
        self.remove_formula_button.clicked.connect(self.remove_formula)
        self.remove_formula_button.setEnabled(False)
        augmentation_layout.addWidget(self.remove_formula_button)
        augmentation_group.setLayout(augmentation_layout)
        self.main_layout.addWidget(augmentation_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.main_layout.addWidget(self.progress_bar)
        
        buttons_layout = QHBoxLayout()
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.close)
        buttons_layout.addWidget(self.cancel_button)
        self.apply_button = QPushButton("Apply Changes and Continue")
        self.apply_button.clicked.connect(self.apply_changes)
        self.apply_button.setStyleSheet("background-color: #0b6337; color: white;")
        self.apply_button.setEnabled(self.train_df is not None)
        buttons_layout.addWidget(self.apply_button)
        self.main_layout.addLayout(buttons_layout)

        if self.train_df is not None:
            self.padding_checkbox.setEnabled(True) # Enable padding checkbox
            self.resampling_checkbox.setEnabled(True)
            self.column_creation_checkbox.setEnabled(True)
        else:
            self.padding_checkbox.setEnabled(False) # Disable padding checkbox
            self.resampling_checkbox.setEnabled(False)
            self.column_creation_checkbox.setEnabled(False)
    
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
                    self.test_df = None
                    self.apply_button.setEnabled(True)
                    QMessageBox.information(self, "Success", "Job folder selected and sample data schema loaded successfully!")
                    self.padding_checkbox.setEnabled(True) # Enable padding
                    self.resampling_checkbox.setEnabled(True)
                    self.column_creation_checkbox.setEnabled(True)
                else:
                    self.train_df = None
                    self.test_df = None
                    self.apply_button.setEnabled(False)
                    self.padding_checkbox.setEnabled(False) # Disable padding
                    self.resampling_checkbox.setEnabled(False)
                    self.column_creation_checkbox.setEnabled(False)
                    QMessageBox.warning(self, "Warning", "Could not load a sample data file from the train/processed_data directory. Ensure CSV files exist there.")
                    self.logger.warning(f"No sample train data loaded from {self.job_folder}")
            except Exception as e:
                self.logger.error(f"Error loading sample data for GUI: {e}", exc_info=True)
                self.train_df = None; self.test_df = None
                self.apply_button.setEnabled(False); self.padding_checkbox.setEnabled(False); self.resampling_checkbox.setEnabled(False); self.column_creation_checkbox.setEnabled(False)
                QMessageBox.critical(self, "Error", f"Could not load sample data schema: {e}")

    def toggle_padding_options(self, state):
        self.padding_length_spinbox.setEnabled(state == Qt.Checked)

    def toggle_resampling_options(self, state):
        self.frequency_combo.setEnabled(state == Qt.Checked)
    
    def toggle_column_creation(self, state):
        self.add_formula_button.setEnabled(state == Qt.Checked)
        self.remove_formula_button.setEnabled(state == Qt.Checked and self.formula_list.count() > 0)
    
    def show_formula_dialog(self):
        if self.train_df is None:
            QMessageBox.warning(self, "Warning", "Please load data first (select a valid job folder).")
            return
        available_columns = list(self.train_df.columns)
        # Get names of columns already created in this session
        session_created_names = [name for name, formula in self.created_columns]
        
        dialog = FormulaInputDialog(available_columns, session_created_names, self) # Pass session names
        if dialog.exec_() == QDialog.Accepted:
            new_column, formula = dialog.new_column_name, dialog.formula
            logger.info(f"DataAugmentGUI: Formula dialog accepted. New column: '{new_column}', Formula: '{formula}'")
            self.formula_list.addItem(f"{new_column} = {formula}")
            self.created_columns.append((new_column, formula))
            logger.info(f"DataAugmentGUI: self.created_columns is now: {self.created_columns}")
            self.remove_formula_button.setEnabled(True)
    
    def remove_formula(self):
        selected_items = self.formula_list.selectedItems()
        if not selected_items: return
        row = self.formula_list.row(selected_items[0])
        self.formula_list.takeItem(row)
        self.created_columns.pop(row)
        if self.formula_list.count() == 0: self.remove_formula_button.setEnabled(False)
    
    def apply_changes(self):
        self.logger.info("apply_changes method entered.")
        if self.job_folder is None:
            QMessageBox.warning(self, "Warning", "Please select a job folder first.")
            return
        self.logger.info(f"Job folder in apply_changes: {self.job_folder}")

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("") # Ensure progress bar is not red
        self.apply_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        
        if hasattr(self, 'status_label') and self.status_label is not None:
            self.status_label.deleteLater()
            self.status_label = None
        
        self.status_label = QLabel("Processing data... Please wait.")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: blue; font-weight: bold; font-size: 14px;")
        idx = self.main_layout.indexOf(self.progress_bar)
        if idx != -1:
            self.main_layout.insertWidget(idx, self.status_label)
        else: 
             self.main_layout.insertWidget(self.main_layout.count() -1 if self.main_layout.count() > 1 else 0, self.status_label)

        self.logger.info("Starting data augmentation process via worker thread...")
        
        padding_enabled = self.padding_checkbox.isChecked()
        padding_length = self.padding_length_spinbox.value() if padding_enabled else 0
        
        resampling_enabled = self.resampling_checkbox.isChecked()
        resampling_frequency = self.frequency_combo.currentText() if resampling_enabled else None
        
        column_formulas = self.created_columns if self.column_creation_checkbox.isChecked() else None
        
        self.logger.info(f"Worker - Padding enabled: {padding_enabled}, length: {padding_length}")
        self.logger.info(f"Worker - Resampling enabled: {resampling_enabled}, frequency: {resampling_frequency}")
        self.logger.info(f"Worker - Column formulas: {column_formulas}")

        self.augmentation_thread = QThread()
        self.augmentation_worker = AugmentationWorker(
            data_augment_manager=self.data_augment_manager,
            job_folder=self.job_folder,
            padding_length=padding_length, # Pass padding length
            resampling_frequency=resampling_frequency,
            column_formulas=column_formulas
        )
        self.augmentation_worker.moveToThread(self.augmentation_thread)

        # Connect MANAGER signals to GUI slots
        self.data_augment_manager.augmentationProgress.connect(self.handle_augmentation_progress)
        self.data_augment_manager.augmentationFinished.connect(self.handle_augmentation_finished)
        self.data_augment_manager.formulaErrorOccurred.connect(self.handle_formula_error) # New connection

        # Connect WORKER's critical error signal
        self.augmentation_worker.criticalError.connect(self.handle_critical_error) # Renamed handler

        # Standard thread management
        self.augmentation_thread.started.connect(self.augmentation_worker.run)
        self.augmentation_thread.finished.connect(self.augmentation_thread.deleteLater) 
        
        # Ensure thread quits on manager signals or worker critical error
        self.data_augment_manager.augmentationFinished.connect(self.augmentation_thread.quit)
        self.data_augment_manager.formulaErrorOccurred.connect(self.augmentation_thread.quit)
        self.augmentation_worker.criticalError.connect(self.augmentation_thread.quit)
        
        self.augmentation_thread.start()

    def handle_augmentation_progress(self, value):
        self.progress_bar.setValue(value)

    # Modified to accept metadata list from manager signal
    def handle_augmentation_finished(self, job_folder, processed_files_metadata):
        self.logger.info(f"Augmentation finished (manager signal) for job: {job_folder}")
        # Check metadata for overall success/failure
        failures = [f for f in processed_files_metadata if f.get('status') == 'Failed']
        if not failures:
            self.logger.info("Augmentation completed successfully for all files.")
            self.progress_bar.setValue(100)
            if hasattr(self, 'status_label') and self.status_label is not None:
                self.status_label.setText("Processing complete!")
            QMessageBox.information(self, "Success", "Data augmentation completed successfully!")
            
            # --- Transition logic ---
            self.apply_button.setText("Continue to Hyperparameter Selection")
            try:
                self.apply_button.clicked.disconnect(self.apply_changes)
            except TypeError:
                self.logger.warning("Could not disconnect apply_changes.")
            try:
                self.apply_button.clicked.disconnect(self.go_to_hyperparameter_gui) 
            except TypeError: pass 
            self.apply_button.clicked.connect(self.go_to_hyperparameter_gui)
            # --- End Transition logic ---

        else:
            # Handle cases where processing finished but some files failed (e.g., resampling errors)
            # Note: Formula errors should have been caught by handle_formula_error and stopped the process earlier.
            error_summary = f"Augmentation finished, but {len(failures)} file(s) failed (non-formula errors):\n"
            for fail in failures[:5]: # Show details for first few failures
                 error_summary += f"- {os.path.basename(fail.get('filepath','Unknown'))}: {fail.get('error', 'Unknown error')}\n"
            if len(failures) > 5: error_summary += "- ... (see logs for more details)"
            
            self.logger.warning(f"Augmentation finished with non-formula errors: {error_summary}")
            self.progress_bar.setValue(100) 
            self.progress_bar.setStyleSheet("") # Reset progress bar style (remove red)
            if hasattr(self, 'status_label') and self.status_label is not None:
                self.status_label.setText("Processing finished with some errors (see logs).")
            QMessageBox.warning(self, "Processing Finished with Errors", error_summary)
            # Do not transition if there were errors
            self.apply_button.setText("Apply Changes and Continue") # Reset button text

        # Re-enable buttons regardless of success/failure
        self.apply_button.setEnabled(True)
        self.cancel_button.setEnabled(True)

    # Renamed from handle_augmentation_error and now handles critical worker errors
    def handle_critical_error(self, error_msg):
        self.logger.error(f"Critical augmentation worker error: {error_msg}")
        self.progress_bar.setValue(0) # Reset progress
        self.progress_bar.setStyleSheet("") # Reset style
        self.progress_bar.setVisible(False) 
        
        if hasattr(self, 'status_label') and self.status_label is not None:
            self.status_label.setText(f"Critical Error: {error_msg}")

        QMessageBox.critical(self, "Critical Error", f"A critical error occurred during processing: {error_msg}")
        
        # Re-enable buttons
        self.apply_button.setEnabled(True)
        self.cancel_button.setEnabled(True)

    # New handler for formula errors emitted by the manager
    def handle_formula_error(self, error_msg):
        """Handles formula errors emitted by the manager."""
        self.logger.error(f"Formula error reported by manager: {error_msg}")
        # Reset progress bar appearance and hide it
        self.progress_bar.setValue(0) 
        self.progress_bar.setStyleSheet("") 
        self.progress_bar.setVisible(False) 

        if hasattr(self, 'status_label') and self.status_label is not None:
            self.status_label.setText(f"Formula Error: Processing stopped.")

        # Display the concise formula error message from the manager/service
        QMessageBox.critical(self, "Formula Error", error_msg) 
        
        # Re-enable buttons
        self.apply_button.setEnabled(True)
        self.cancel_button.setEnabled(True)
    
    def go_to_hyperparameter_gui(self):
        self.logger.info("Transitioning to hyperparameter GUI...")
        try:
            self.hyper_param_gui = VEstimHyperParamGUI()
            self.hyper_param_gui.show()
            self.close() # Close current window
        except Exception as e:
            self.logger.error(f"Error transitioning to hyperparameter GUI: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Could not open hyperparameter selection: {e}")

def main():
    app = QApplication(sys.argv)
    # Example: Launch with a specific job folder if available (e.g., from command line or previous step)
    # job_folder_to_pass = "path/to/your/job_folder" # Replace with actual logic if needed
    # ex = DataAugmentGUI(job_folder=job_folder_to_pass)
    ex = DataAugmentGUI() 
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()