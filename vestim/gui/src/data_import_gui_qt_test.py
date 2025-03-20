# ---------------------------------------------------------------------------------
# Author: Biswanath Dehury
# Date: `{{date:2025-03-01}}`
# Version: 1.0.0
# Description: 
# Entry file for the program and gives the user an UI and to choose folders to select train and test data from.
# Now it has Digatron, Tesla and Pouch data sources to choose from from the dropdown menu
#Shows the progress bar for file conversion and the Tesla and Digatron data processors are used to convert the files from mat to csv amd organize them
#The job folder is created and the files are copied and converted to the respective folders s train raw and processed and similar for test files
# 
# Next Steps:
# 1. Resamling of data to 1hz (Done)
# 2. Letting the user to select features and targets from the data [ TO DO ] -> To be implemented in the hyperparameter GUI
# ---------------------------------------------------------------------------------


from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QListWidget, QFileDialog, QProgressBar, QWidget, QMessageBox, QComboBox, QSizePolicy
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject

import os, sys
from vestim.gui.src.hyper_param_gui_qt_test import VEstimHyperParamGUI  # Adjust this import based on your actual path
from vestim.services.data_processor.src.data_processor_qt_digatron import DataProcessorDigatron
from vestim.services.data_processor.src.data_processor_qt_tesla import DataProcessorTesla
from vestim.services.data_processor.src.data_processor_qt_pouch import DataProcessorPouch

import logging

from vestim.logger_config import setup_logger  # Assuming you have logger_config.py as shared earlier
# Set up initial logging to a default log file
logger = setup_logger(log_file='default.log')  # Log everything to 'default.log' initially

class DataImportGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.train_folder_path = ""
        self.test_folder_path = ""
        self.selected_train_files = []
        self.selected_test_files = []
        self.data_processor_digatron = DataProcessorDigatron()  # Initialize DataProcessor
        self.data_processor_tesla = DataProcessorTesla()  # Initialize DataProcessor
        self.data_processor_pouch = DataProcessorPouch()

        self.sampling_frequency = None  # Default sampling frequency

        self.organizer_thread = None
        self.organizer = None

        self.initUI()

    def initUI(self):
        self.setWindowTitle("VEstim Modelling Tool")
        self.setGeometry(100, 100, 900, 700)  # Set window size

        # Main vertical layout
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # ---- HEADER ----
        self.header_label = QLabel("Select Data Folders for Training, Validation, and Testing", self)
        self.header_label.setAlignment(Qt.AlignCenter)
        self.header_label.setStyleSheet("font-size: 20px; font-weight: bold; color: darkgreen; padding: 10px;")
        self.main_layout.addWidget(self.header_label)

        # ---- SPACING BELOW HEADER ----
        self.main_layout.addSpacing(10)

        # ---- Training & Validation Section (Side-by-Side) ----
        train_val_layout = QHBoxLayout()  # Horizontal layout for Train & Validation

        # --- Training Data Section ---
        train_layout = QVBoxLayout()
        self.train_select_button = QPushButton("Select Train Data Folder", self)
        self.train_select_button.setFixedHeight(30)
        self.train_select_button.clicked.connect(self.select_train_folder)
        train_layout.addWidget(self.train_select_button)

        self.train_list_widget = QListWidget(self)
        self.train_list_widget.setSelectionMode(QListWidget.MultiSelection)
        self.train_list_widget.setMinimumHeight(120)  
        train_layout.addWidget(self.train_list_widget)
        train_val_layout.addLayout(train_layout)

        # --- Validation Data Section ---
        val_layout = QVBoxLayout()
        self.val_select_button = QPushButton("Select Validation Data Folder", self)
        self.val_select_button.setFixedHeight(30)
        self.val_select_button.clicked.connect(self.select_valid_folder)
        val_layout.addWidget(self.val_select_button)

        self.valid_list_widget = QListWidget(self)
        self.valid_list_widget.setSelectionMode(QListWidget.MultiSelection)
        self.valid_list_widget.setMinimumHeight(120)
        val_layout.addWidget(self.valid_list_widget)
        train_val_layout.addLayout(val_layout)

        # Add Train & Validation layout to main layout
        self.main_layout.addLayout(train_val_layout)

        # ---- SPACING BELOW TRAIN/VALID SECTION ----
        self.main_layout.addSpacing(10)

        # ---- Testing Section (Below Train/Valid) ----
        test_layout = QVBoxLayout()

        # Test folder selection button
        self.test_select_button = QPushButton("Select Test Data Folder", self)
        self.test_select_button.setFixedHeight(30)
        self.test_select_button.clicked.connect(self.select_test_folder)

        # Center the test button
        test_button_layout = QHBoxLayout()
        test_button_layout.addStretch(1)
        test_button_layout.addWidget(self.test_select_button, alignment=Qt.AlignCenter)
        test_button_layout.addStretch(1)
        test_layout.addLayout(test_button_layout)

        # Test files list
        self.test_list_widget = QListWidget(self)
        self.test_list_widget.setSelectionMode(QListWidget.MultiSelection)
        self.test_list_widget.setMinimumHeight(100)
        test_layout.addWidget(self.test_list_widget)

        # Reduce test section height by adding stretch
        test_layout.addStretch(1)

        # Add testing section to main layout
        self.main_layout.addLayout(test_layout)

        # ---- SPACING BELOW TEST SECTION ----
        self.main_layout.addSpacing(10)

        # ---- Data Source and Sampling Frequency (Compact Alignment) ----
        data_layout = QHBoxLayout()
        
        # Add left padding to shift everything to the right
        data_layout.setContentsMargins(60, 0, 0, 0)  # (Left, Top, Right, Bottom)

        # Data Source (Label + Dropdown without unnecessary spacing)
        data_source_container = QHBoxLayout()
        data_source_label = QLabel("Data Source:")
        data_source_label.setStyleSheet("color: purple; font-weight: bold; font-size: 14px;")

        self.data_source_combo = QComboBox(self)
        self.data_source_combo.addItems(["Digatron", "Tesla", "Pouch"])
        self.data_source_combo.setFixedHeight(35)
        self.data_source_combo.setFixedWidth(120)
        self.data_source_combo.currentIndexChanged.connect(self.update_file_display)

        # **Ensure they are tightly packed together**
        data_source_container.addWidget(data_source_label)
        data_source_container.addWidget(self.data_source_combo)
        data_source_container.addStretch(0)  # No unnecessary stretch
        data_source_container.setSpacing(2)  # **Minimize spacing between label & dropdown**
        data_layout.addLayout(data_source_container)

        data_layout.addSpacing(80)  # Add spacing between the dropdowns
        # Sampling Frequency (Label + Dropdown without unnecessary spacing)
        sampling_freq_container = QHBoxLayout()
        sampling_frequency_label = QLabel("Resampling Freq:")
        sampling_frequency_label.setStyleSheet("color: purple; font-weight: bold; font-size: 14px;")

        self.sampling_frequency_combo = QComboBox(self)
        self.sampling_frequency_combo.addItems(["None", "0.1Hz", "0.5Hz", "1Hz", "5Hz", "10Hz"])
        self.sampling_frequency_combo.setFixedHeight(35)
        self.sampling_frequency_combo.setFixedWidth(110)
        self.sampling_frequency_combo.currentIndexChanged.connect(self.update_sampling_frequency)

        # **Ensure they are tightly packed together**
        sampling_freq_container.addWidget(sampling_frequency_label)
        sampling_freq_container.addWidget(self.sampling_frequency_combo)
        sampling_freq_container.addStretch(0)  # No unnecessary stretch
        sampling_freq_container.setSpacing(2)  # **Minimize spacing between label & dropdown**
        data_layout.addLayout(sampling_freq_container)

        # Add to main layout
        self.main_layout.addLayout(data_layout)



        # ---- SPACING BELOW SOURCE & FREQUENCY SELECTION ----
        self.main_layout.addSpacing(15)

        # ---- Organize Button (Centered Below) ----
        button_layout = QHBoxLayout()
        button_layout.addStretch(1)  # Add stretch to center the button

        self.organize_button = QPushButton("Load and Prepare Files", self)
        self.organize_button.setFixedHeight(40)
        self.organize_button.setMinimumWidth(180)
        self.organize_button.setMaximumWidth(300)
        self.organize_button.setStyleSheet("background-color: #0b6337; font-weight: bold; color: white;")
        self.organize_button.clicked.connect(self.organize_files)

        button_layout.addWidget(self.organize_button, alignment=Qt.AlignCenter)
        button_layout.addStretch(1)  # Add stretch to center the button

        self.main_layout.addLayout(button_layout)

        # ---- SPACING BELOW ORGANIZE BUTTON ----
        self.main_layout.addSpacing(10)

        # Progress bar (now spans full width)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setAlignment(Qt.AlignCenter)  # Center the percentage text
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                text-align: center;  /* Ensures text is centered */
                font-weight: bold;   /* Makes text bold */
                font-size: 14px;     /* Adjusts text size */
                color: black;        /* Sets text color to black */
            }
        """)

        self.progress_bar.setMinimumWidth(700)  # Set wide width
        self.progress_bar.setMaximumWidth(900)  # Ensure it scales well
        self.progress_bar.setStyleSheet("padding: 5px;")  # Add slight padding for aesthetics

        # Add to the layout with no centering constraint (full width effect)
        self.main_layout.addWidget(self.progress_bar)


        # Add final spacing at the bottom for a clean UI
        self.main_layout.addStretch(1)




    def update_file_display(self):
        selected_source = self.data_source_combo.currentText()
        if selected_source == "Digatron" or selected_source == "Tesla":
            # Show only .mat files
            self.populate_file_list(self.train_folder_path, self.train_list_widget, file_extension=".mat")
            self.populate_file_list(self.test_folder_path, self.test_list_widget, file_extension=".mat")
            self.populate_file_list(self.valid_folder_path, self.valid_list_widget, file_extension=".mat")
        elif selected_source == "Pouch":
            # Show only .csv files
            self.populate_file_list(self.train_folder_path, self.train_list_widget, file_extension=".csv")
            self.populate_file_list(self.test_folder_path, self.test_list_widget, file_extension=".csv")
            self.populate_file_list(self.valid_folder_path, self.valid_list_widget, file_extension=".csv")
    
    def update_sampling_frequency(self):
        selected_value = self.sampling_frequency_combo.currentText().strip()

        if selected_value.lower() == "none":  # Check if "None" is selected
            self.sampling_frequency = None
        else:
            try:
                frequency_hz = float(selected_value.replace("Hz", "").strip())  # Convert "1Hz" → 1.0
                
                if frequency_hz <= 0:
                    self.sampling_frequency = None  # Invalid values default to None
                else:
                    interval_ms = int(1000 / frequency_hz)  # Convert Hz to milliseconds
                    self.sampling_frequency = f"{interval_ms}L" if interval_ms < 1000 else f"{interval_ms // 1000}S"

            except ValueError:
                self.sampling_frequency = None  # Fallback in case of errors

        self.logger.info(f"Updated resampling frequency: {self.sampling_frequency}")  # Logging for debugging


    def select_train_folder(self):
        self.train_folder_path = QFileDialog.getExistingDirectory(self, "Select Training Folder")
        if self.train_folder_path:
            self.populate_file_list(self.train_folder_path, self.train_list_widget)
            logger.info(f"Selected training folder: {self.train_folder_path}")
        self.check_folders_selected()

    def select_valid_folder(self):
        self.valid_folder_path = QFileDialog.getExistingDirectory(self, "Select Validation Folder")
        if self.valid_folder_path:
            self.populate_file_list(self.valid_folder_path, self.valid_list_widget)
            logger.info(f"Selected validation folder: {self.valid_folder_path}")
        self.check_folders_selected()

    def select_test_folder(self):
        self.test_folder_path = QFileDialog.getExistingDirectory(self, "Select Testing Folder")
        if self.test_folder_path:
            self.populate_file_list(self.test_folder_path, self.test_list_widget)
            logger.info(f"Selected testing folder: {self.test_folder_path}")
        self.check_folders_selected()

    def populate_file_list(self, folder_path, list_widget, file_extension=".mat"):
        """ Populate the list widget with specified file extension. """
        list_widget.clear()
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(file_extension):  # Filter files by the selected extension
                    list_widget.addItem(os.path.join(root, file))

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
        valid_files = [item.text() for item in self.valid_list_widget.selectedItems()]
        print(f"Train files: {train_files}")
        print(f"Test files: {test_files}")


        if not train_files or not test_files:
            self.show_error("No files selected for either training or testing.")
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
        if selected_source == "Digatron":
            data_processor = self.data_processor_digatron
        elif selected_source == "Tesla":
            data_processor = self.data_processor_tesla
        elif selected_source == "Pouch":
            data_processor = self.data_processor_pouch
        else:
            self.show_error("Invalid data source selected.")
            return

        # Create and start the file organizer thread with the selected data processor
        self.organizer = FileOrganizer(train_files, valid_files, test_files, data_processor, sampling_frequency=self.sampling_frequency)
        self.organizer_thread = QThread()

        # Connect signals and slots
        self.organizer.progress.connect(self.handle_progress_update)
        self.organizer.job_folder_signal.connect(self.on_files_processed)

        self.organizer.moveToThread(self.organizer_thread)
        self.organizer_thread.started.connect(self.organizer.run)
        self.organizer_thread.start()


    def handle_progress_update(self, progress_value):
        # Update the progress bar with the percentage
        self.progress_bar.setValue(progress_value)

    def on_files_processed(self, job_folder):
        # Handle when files are processed and job folder is created
        logger.info(f"Files processed successfully. Job folder: {job_folder}")
        self.progress_bar.setVisible(False)
        
        # Change the button label to indicate next step and enable it
        self.organize_button.setText("Select Hyperparameters")
        self.organize_button.setStyleSheet("""
            background-color: #1f8b4c; 
            font-weight: bold;
            padding: 10px 20px;
            color: white;
            font-size: 14px;
                                           
        """)
        self.organize_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # Allow the button to expand horizontally
        self.organize_button.setEnabled(True)
        
        # Update the button action to move to the next screen
        self.organize_button.clicked.disconnect()  # Disconnect the previous function
        self.organize_button.clicked.connect(lambda: self.move_to_next_screen(job_folder))  # Connect new function

    def move_to_next_screen(self, job_folder):
        self.close()
        self.hyper_param_gui = VEstimHyperParamGUI()
        self.hyper_param_gui.show()

    def show_error(self, message):
        # Display error message
        QMessageBox.critical(self, "Error", message)

class FileOrganizer(QObject):
    progress = pyqtSignal(int)  # Emit progress percentage
    job_folder_signal = pyqtSignal(str)  # To communicate when the job folder is created

    def __init__(self, train_files, valid_files, test_files, data_processor, sampling_frequency=None):
        super().__init__()
        self.train_files = train_files
        self.test_files = test_files
        self.valid_files = valid_files
        self.data_processor = data_processor
        self.sampling_frequency = sampling_frequency

    def run(self):
        if not self.train_files or not self.test_files:
            self.progress.emit(0)  # Emit 0% if no files selected
            return

        try:
            # Call the backend method from DataProcessor to organize and convert files
            job_folder = self.data_processor.organize_and_convert_files(self.train_files, self.valid_files, self.test_files, progress_callback=self.update_progress, sampling_frequency=self.sampling_frequency)
            logger.info(f"Job folder created: {job_folder}")
            # Emit success message with job folder details
            self.job_folder_signal.emit(job_folder)
        except Exception as e:
            self.progress.emit(0)  # Emit 0% if there is an error
            logger.error(f"Error occurred during file organization: {e}")

    def update_progress(self, progress_value):
        """Emit progress as a percentage."""
        self.progress.emit(progress_value)

def main():
    app = QApplication(sys.argv)
    gui = DataImportGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
