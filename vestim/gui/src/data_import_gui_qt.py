# ---------------------------------------------------------------------------------
# Author: Biswanath Dehury
# Date: `2025-04-14`
# Version: 1.1.0
# Description: 
# Entry file for the program and gives the user an UI and to choose folders to select train and test data from.
# Now it has Arbin, STLA and Digatron data sources to choose from from the dropdown menu
# Shows the progress bar for file conversion and the STLA and Arbin data processors are used to convert the files from mat to csv and organize them
# The job folder is created and the files are copied and converted to the respective folders as train raw and processed and similar for test files
# 
# Next Steps:
# 1. Resampling moved to data_augment_gui_qt_test.py (Done)
# 2. Letting the user to create new columns from the existing columns (Done in data_augment_gui_qt_test.py)
# 3. Letting the user to select features and targets from the data (To be implemented in the hyperparameter GUI)
# ---------------------------------------------------------------------------------


from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QListWidget, QFileDialog, QProgressBar, QWidget, QMessageBox, QComboBox, QSizePolicy
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject

import os, sys
import requests
import json
from datetime import datetime

from vestim.gui.src.data_augment_gui_qt import DataAugmentGUI  # Import the new data augmentation GUI
from vestim.backend.src.services.data_processor.src.data_processor_qt_arbin import DataProcessorArbin
from vestim.backend.src.services.data_processor.src.data_processor_qt_stla import DataProcessorSTLA
from vestim.backend.src.services.data_processor.src.data_processor_qt_digatron import DataProcessorDigatron
from vestim.gui.src.api_gateway import APIGateway

import logging

from vestim.logger_config import setup_logger  # Assuming you have logger_config.py as shared earlier
# Set up initial logging to a default log file
logger = setup_logger(log_file='default.log')  # Log everything to 'default.log' initially

DEFAULT_DATA_EXTENSIONS = [".csv", ".txt", ".mat", ".xls", ".xlsx", ".RES"] # Added .RES for Biologic, expand as needed

class DataImportGUI(QMainWindow):
    jobCreated = pyqtSignal(str)  # Emit job_id when a job is created

    def __init__(self, api_gateway: APIGateway, job_id: str = None):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.api_gateway = api_gateway
        self.job_id = job_id
        self.train_folder_path = ""
        self.test_folder_path = ""
        
        self.organizer_thread = None
        self.organizer = None
        self.data_augment_gui = None
        
        # If job_id is provided, this is for resuming an existing job
        if self.job_id:
            self.logger.info(f"DataImportGUI initialized for existing job: {self.job_id}")
            # Get job status for state restoration
            try:
                self.job_status = self.api_gateway.get_job_detailed_status(self.job_id)
                if self.job_status:
                    self.logger.info(f"Retrieved job status: {self.job_status.get('status')} - {self.job_status.get('progress_message')}")
                else:
                    self.logger.warning("Could not retrieve detailed job status")
                    self.job_status = {"status": "created", "phase_progress": {}}
            except Exception as e:
                self.logger.error(f"Error retrieving job status: {e}")
                self.job_status = {"status": "created", "phase_progress": {}}
        else:
            self.job_status = None

        self.initUI()
        
        # Restore GUI state if resuming existing job
        if self.job_id and self.job_status:
            self.restore_gui_state()

    def initUI(self):
        self.setWindowTitle("VEstim Modelling Tool")
        self.setGeometry(100, 100, 900, 600)

        # Main layout
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Header
        self.header_label = QLabel("Select data folders to train your LSTM Model", self)
        self.header_label.setAlignment(Qt.AlignCenter)
        self.header_label.setStyleSheet("font-size: 18px; font-weight: bold; color: green;")
        self.main_layout.addWidget(self.header_label)

        # Training folder section
        train_layout = QVBoxLayout()
        self.train_select_button = QPushButton("Select Train Data Folder", self)
        self.train_select_button.setStyleSheet("""
            background-color: #add8e6;  /* Light blue background */
            font-weight: bold;
            padding: 8px 15px;  /* Adds padding inside the button */
            color: black;  /* Set the text color to black for contrast */
        """)
        self.train_select_button.setFixedHeight(30)  # Set a consistent height for the button
        self.train_select_button.setMinimumWidth(150)  # Set a reasonable minimum width
        self.train_select_button.setMaximumWidth(300)  # Set a reasonable maximum width
        self.train_select_button.clicked.connect(self.select_train_folder)

        # Center the train button
        train_button_layout = QHBoxLayout()
        train_button_layout.addStretch(1)  # Add stretch before
        train_button_layout.addWidget(self.train_select_button, alignment=Qt.AlignCenter)
        train_button_layout.addStretch(1)  # Add stretch after
        train_layout.addLayout(train_button_layout)

        # Train folder list widget (covers full width)
        self.train_list_widget = QListWidget(self)
        self.train_list_widget.setSelectionMode(QListWidget.MultiSelection)
        self.train_list_widget.setMinimumHeight(100)  # Set minimum height for scrollability
        train_layout.addWidget(self.train_list_widget)

        # Add the training section to the main layout
        self.main_layout.addLayout(train_layout)

        # Testing folder section
        test_layout = QVBoxLayout()
        self.test_select_button = QPushButton("Select Test Data Folder", self)
        self.test_select_button.setStyleSheet("""
            background-color: #add8e6;  /* Light blue background */
            font-weight: bold;
            padding: 8px 15px;  /* Adds padding inside the button */
            color: black;  /* Set the text color to black for contrast */
        """)
        self.test_select_button.setFixedHeight(30)  # Set a consistent height for the button
        self.test_select_button.setMinimumWidth(150)  # Set a reasonable minimum width
        self.test_select_button.setMaximumWidth(300)  # Set a reasonable maximum width
        self.test_select_button.clicked.connect(self.select_test_folder)

        # Center the test button
        test_button_layout = QHBoxLayout()
        test_button_layout.addStretch(1)
        test_button_layout.addWidget(self.test_select_button, alignment=Qt.AlignCenter)
        test_button_layout.addStretch(1)
        test_layout.addLayout(test_button_layout)

        # Test folder list widget (covers full width)
        self.test_list_widget = QListWidget(self)
        self.test_list_widget.setSelectionMode(QListWidget.MultiSelection)
        self.test_list_widget.setMinimumHeight(100)
        test_layout.addWidget(self.test_list_widget)

        # Add the testing section to the main layout
        self.main_layout.addLayout(test_layout)

        # Main layout for data source selection and organize button
        combined_layout = QHBoxLayout()

        # Data source label with color change, bold text, and padding
        data_source_label = QLabel("Data Source:")
        data_source_label.setStyleSheet("color: purple; font-weight: bold; font-size: 14px; padding-right: 10px;")
        combined_layout.addWidget(data_source_label)

        # Data source selection with consistent height and styling
        self.data_source_combo = QComboBox(self)
        self.data_source_combo.addItems(["Arbin", "STLA", "Digatron", "Biologic"])  # Added Biologic
        self.data_source_combo.setFixedHeight(35)  # Set a specific height for the ComboBox
        self.data_source_combo.setFixedWidth(120)  # Set a specific width for the ComboBox
        self.data_source_combo.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        combined_layout.addWidget(self.data_source_combo)
        self.data_source_combo.currentIndexChanged.connect(self.on_data_source_selection_changed)

        # Add stretchable space between the dropdown and the button
        combined_layout.addStretch(1)  # Push the button to the right

        # Organize button with consistent height and padding
        self.organize_button = QPushButton("Load and Prepare Files", self)
        self.organize_button.setStyleSheet("""
            background-color: #0b6337; 
            font-weight: bold; 
            padding: 10px 20px;  /* Adjust padding for visual appeal */
            color: white;  
            font-size: 14px;  /* Increase font size */
        """)
        self.organize_button.setFixedHeight(35)  # Ensure consistent height
        self.organize_button.setMinimumWidth(150)  # Set minimum width
        self.organize_button.setMaximumWidth(300)  # Set maximum width
        self.organize_button.clicked.connect(self.organize_files)

        # Add stretchable space to center the button and provide border spacing
        combined_layout.addStretch(2)  # Adds extra space for centering

        # Add the organize button to the combined layout
        combined_layout.addWidget(self.organize_button)

        # Add the combined layout to the main layout
        self.main_layout.addLayout(combined_layout)

        # Add margins to the main layout for border spacing
        self.main_layout.setContentsMargins(30, 10, 30, 10)  # Increase left, right margins for more centering

        # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)  # Initially hidden
        self.main_layout.addWidget(self.progress_bar)

    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)

    def on_data_source_selection_changed(self, index):
        """
        Called when the data source selection changes.
        Refreshes the file lists based on the new data source.
        """
        selected_source = self.data_source_combo.currentText()
        logger.info(f"Data source changed to: {selected_source}. Refreshing file lists.")
        if self.train_folder_path:
            self.populate_file_list(self.train_folder_path, self.train_list_widget, selected_source)
        if self.test_folder_path:
            self.populate_file_list(self.test_folder_path, self.test_list_widget, selected_source)

    def select_train_folder(self):
        self.train_folder_path = QFileDialog.getExistingDirectory(self, "Select Training Folder")
        if self.train_folder_path:
            selected_source = self.data_source_combo.currentText()
            # self.data_source_combo.blockSignals(True) # Not needed if populate_file_list handles current source
            # try:
            self.populate_file_list(self.train_folder_path, self.train_list_widget, selected_source)
            logger.info(f"Selected training folder: {self.train_folder_path}. Populated for source: {selected_source}.")
            # finally:
            #     self.data_source_combo.blockSignals(False)
        self.check_folders_selected()

    def select_test_folder(self):
        self.test_folder_path = QFileDialog.getExistingDirectory(self, "Select Testing Folder")
        if self.test_folder_path:
            selected_source = self.data_source_combo.currentText()
            # self.data_source_combo.blockSignals(True)
            # try:
            self.populate_file_list(self.test_folder_path, self.test_list_widget, selected_source)
            logger.info(f"Selected testing folder: {self.test_folder_path}. Populated for source: {selected_source}.")
            # finally:
            #     self.data_source_combo.blockSignals(False)
        self.check_folders_selected()

    def populate_file_list(self, folder_path, list_widget, data_source):
        """
        Populate the list widget with files matching extensions for the given data_source.
        """
        list_widget.clear()
        if not folder_path or not os.path.isdir(folder_path):
            return

        if data_source == "Arbin":
            extensions_to_check = [".mat"]
        elif data_source == "Digatron":
            extensions_to_check = [".csv"]
        elif data_source == "STLA":
            extensions_to_check = [".xlsx", ".xls"] # Only Excel files for STLA
        else: # Default for Biologic, and any others not explicitly handled
            extensions_to_check = [ext.lower() for ext in DEFAULT_DATA_EXTENSIONS]
        
        logger.info(f"Populating list for '{folder_path}' (Source: {data_source}). Scanning for extensions: {extensions_to_check}")
        
        items_added_count = 0
        try:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in extensions_to_check):
                        list_widget.addItem(os.path.join(root, file))
                        items_added_count +=1
        except Exception as e:
            logger.error(f"Error during file iteration or adding item for {folder_path}: {e}", exc_info=True)
        
        logger.info(f"Finished populating list for '{folder_path}'. Total items added: {items_added_count}. List widget current count: {list_widget.count()}")
        list_widget.update() # Explicitly request a widget update

    def check_folders_selected(self):
        if self.train_folder_path and self.test_folder_path:
            self.organize_button.setEnabled(True)
        else:
            self.organize_button.setEnabled(False)

    def organize_files(self):
        
        logger.info("Starting file organization process...")
        # Use selectedItems() to get the selected files
        train_files = [item.text() for item in self.train_list_widget.selectedItems()]
        test_files = [item.text() for item in self.test_list_widget.selectedItems()]
        print(f"Train files: {train_files}")
        print(f"Test files: {test_files}")


        if not train_files or not test_files:
            self.show_error("Please select files for both training and testing.")
            self.organize_button.setEnabled(True)
            self.organize_button.setText("Load and Prepare Files")
            self.organize_button.setStyleSheet("""
            background-color: #0b6337;
            font-weight: bold;
            padding: 10px 20px;  /* Adjust padding for visual appeal */
            color: white;
            font-size: 14px;  /* Increase font size */
        """)
            return
        # Update the button label and color when the process starts
        self.organize_button.setText("Importing and Preprocessing Files")
        self.organize_button.setStyleSheet("""
            background-color: #3ecf86;  /* Light green color */
            font-weight: bold; 
            padding: 10px 20px;  /* Same padding to maintain size */
            color: white;
        """)
        self.organize_button.setEnabled(False)  # Disable the button during processing

        # Show progress label and start the background thread
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Determine which data processor to use based on the selected data source
        selected_source = self.data_source_combo.currentText()

        # Create and start the file organizer thread
        self.organizer = FileOrganizer(
            train_files,
            test_files,
            self.api_gateway,
            data_source=selected_source
        )
        self.organizer_thread = QThread()
        self.organizer.moveToThread(self.organizer_thread)

        # Connect signals
        self.organizer.progress.connect(self.handle_progress_update)
        self.organizer.finished.connect(self.on_processing_finished)
        self.organizer.error.connect(self.on_processing_error)

        self.organizer_thread.started.connect(self.organizer.run)
        self.organizer_thread.start()

    def on_processing_finished(self, result):
        """Handle successful completion of the file processing thread."""
        self.progress_bar.setValue(100)
        job_id = result.get('job_id', '')
        QMessageBox.information(self, "Success", f"Job {job_id} created successfully. You can now open it from the dashboard.")
        
        # Emit the job_id that was created
        self.jobCreated.emit(job_id)

        # Don't automatically close/hide - let user decide when to close
        # The dashboard will show the new job and user can open the next phase when ready

    def on_processing_error(self, error_message):
        """Handle errors from the file processing thread."""
        self.progress_bar.setVisible(False)
        self.organize_button.setEnabled(True)
        self.organize_button.setText("Load and Prepare Files")
        QMessageBox.critical(self, "Error", f"An error occurred: {error_message}")
    def handle_progress_update(self, progress_value):
        # Update the progress bar with the percentage
        self.progress_bar.setValue(progress_value)
        
class FileOrganizer(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, train_files, test_files, api_gateway, data_source=None):
        super().__init__()
        self.train_files = train_files
        self.test_files = test_files
        self.api_gateway = api_gateway
        self.data_source = data_source

    def run(self):
        try:
            selections = {
                "train_files": self.train_files,
                "test_files": self.test_files,
                "data_source": self.data_source,
            }
            job = self.api_gateway.post("jobs/process-and-create", json={"selections": selections})
            self.finished.emit(job)
        except Exception as e:
            self.error.emit(str(e))

def main():
    app = QApplication(sys.argv)
    gui = DataImportGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
