from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QListWidget, QFileDialog, QProgressBar, QWidget, QMessageBox, QComboBox
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject

import os, sys
from vestim.gui.src.hyper_param_gui_qt_test import VEstimHyperParamGUI  # Adjust this import based on your actual path
from vestim.services.data_processor.src.data_processor_qt_digatron import DataProcessorDigatron
from vestim.services.data_processor.src.data_processor_qt_tesla import DataProcessorTesla
from vestim.services.data_processor.src.data_processor_qt_pouch import DataProcessorPouch

from vestim.logger_config import setup_logger  # Assuming you have logger_config.py as shared earlier

# Set up initial logging to a default log file
logger = setup_logger(log_file='default.log')  # Log everything to 'default.log' initially

class DataImportGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.train_folder_path = ""
        self.test_folder_path = ""
        self.selected_train_files = []
        self.selected_test_files = []
        self.data_processor_digatron = DataProcessorDigatron()  # Initialize DataProcessor
        self.data_processor_tesla = DataProcessorTesla()  # Initialize DataProcessor
        self.data_processor_pouch = DataProcessorPouch()

        self.organizer_thread = None
        self.organizer = None

        self.initUI()

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

        #Adding the option to select the data processor (Version 2.0 VEstim)
        # Main layout for data source selection and organize button
        combined_layout = QHBoxLayout()

        # Data source label with color change, bold text, and padding
        data_source_label = QLabel("Select Data Source:")
        data_source_label.setStyleSheet("color: purple; font-weight: bold; font-size: 14px; padding-right: 10px;")  # Set text color to purple, bold, and larger size
        combined_layout.addWidget(data_source_label)

        # Data source selection with consistent height and styling
        self.data_source_combo = QComboBox(self)
        self.data_source_combo.addItems(["Digatron", "Tesla", "Pouch"])  # Add the data sources
        self.data_source_combo.setFixedHeight(35)  # Set a specific height for the ComboBox
        self.data_source_combo.setFixedWidth(150)  # Set a specific width for the ComboBox
        self.data_source_combo.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")  # Bold text and larger font size
        combined_layout.addWidget(self.data_source_combo)
        self.data_source_combo.currentIndexChanged.connect(self.update_file_display)

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

    def select_train_folder(self):
        self.train_folder_path = QFileDialog.getExistingDirectory(self, "Select Training Folder")
        if self.train_folder_path:
            self.populate_file_list(self.train_folder_path, self.train_list_widget)
            logger.info(f"Selected training folder: {self.train_folder_path}")
        self.check_folders_selected()

    def select_test_folder(self):
        self.test_folder_path = QFileDialog.getExistingDirectory(self, "Select Testing Folder")
        if self.test_folder_path:
            self.populate_file_list(self.test_folder_path, self.test_list_widget)
            logger.info(f"Selected testing folder: {self.test_folder_path}")
        self.check_folders_selected()
    
    def update_file_display(self):
        selected_source = self.data_source_combo.currentText()
        if selected_source == "Digatron" or selected_source == "Tesla":
            # Show only .mat files for Digatron and Tesla
            self.populate_file_list(self.train_folder_path, self.train_list_widget, file_extension=".mat")
            self.populate_file_list(self.test_folder_path, self.test_list_widget, file_extension=".mat")
        elif selected_source == "Pouch":
            # Show only .csv files for Pouch
            self.populate_file_list(self.train_folder_path, self.train_list_widget, file_extension=".csv")
            self.populate_file_list(self.test_folder_path, self.test_list_widget, file_extension=".csv")

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

        logger.info("Starting file organization process...")
        # Use selectedItems() to get the selected files
        train_files = [item.text() for item in self.train_list_widget.selectedItems()]
        test_files = [item.text() for item in self.test_list_widget.selectedItems()]

        if not train_files or not test_files:
            self.show_error("No files selected for either training or testing.")
            return

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
        self.organizer = FileOrganizer(train_files, test_files, data_processor)
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
        self.organize_button.setText("Proceed to Hyperparameter Selection")
        self.organize_button.setStyleSheet("""
            background-color: #1f8b4c; 
            font-weight: bold;
            padding: 10px 20px;
            color: white;
            font-size: 14px;
                                           
        """)
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

    def __init__(self, train_files, test_files, data_processor):
        super().__init__()
        self.train_files = train_files
        self.test_files = test_files
        self.data_processor = data_processor

    def run(self):
        if not self.train_files or not self.test_files:
            self.progress.emit(0)  # Emit 0% if no files selected
            return

        try:
            # Call the backend method from DataProcessor to organize and convert files
            job_folder = self.data_processor.organize_and_convert_files(self.train_files, self.test_files, progress_callback=self.update_progress)
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
