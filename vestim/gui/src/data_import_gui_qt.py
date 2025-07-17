# ---------------------------------------------------------------------------------
# Author: Biswanath Dehury
# Date: `2025-04-14`
# Version: 1.1.0
# Description: 
# Entry file for the program and gives the user an UI and to choose folders to select train and test data from.
# Now it has CSV, MAT, and XLSX file format options to choose from in the dropdown menu (CSV as default)
# Shows the progress bar for file conversion and the MAT and XLSX data processors are used to convert the files and organize them
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
from vestim.gateway.src.job_manager_qt import JobManager
from vestim.gui.src.data_augment_gui_qt import DataAugmentGUI  # Import the new data augmentation GUI
from vestim.services.data_processor.src.data_processor_qt_csv import DataProcessorCSV
from vestim.services.data_processor.src.data_processor_qt_mat import DataProcessorMAT
from vestim.services.data_processor.src.data_processor_qt_xlsx import DataProcessorXLSX
from vestim.config_manager import get_data_directory, get_default_folders, update_last_used_folders, get_default_file_format

import logging

from vestim.logger_config import setup_logger  # Assuming you have logger_config.py as shared earlier
# Set up initial logging to a default log file
logger = setup_logger(log_file='default.log')  # Log everything to 'default.log' initially

DEFAULT_DATA_EXTENSIONS = [".csv", ".txt", ".mat", ".xls", ".xlsx", ".RES"] # Common data file extensions

def shorten_path_for_display(full_path, max_levels=3):
    """
    Shorten a file path for display, showing data/.../ format.
    Keeps the last max_levels directories plus filename.
    """
    try:
        # Normalize the path
        normalized_path = os.path.normpath(full_path)
        path_parts = normalized_path.split(os.sep)
        
        # Find if 'data' folder exists in path
        data_index = -1
        for i, part in enumerate(path_parts):
            if part.lower() == 'data':
                data_index = i
                break
        
        if data_index >= 0:
            # Start from data folder
            relevant_parts = path_parts[data_index:]
        else:
            # No data folder found, use last max_levels + 1 (for filename)
            relevant_parts = path_parts[-(max_levels + 1):]
        
        # If path is too long, show data/.../last_few_parts
        if len(relevant_parts) > max_levels + 1:
            display_parts = [relevant_parts[0], '...'] + relevant_parts[-(max_levels-1):]
        else:
            display_parts = relevant_parts
            
        return '/'.join(display_parts)
    except:
        # Fallback to just filename if anything goes wrong
        return os.path.basename(full_path)

class DataImportGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.job_manager = JobManager()
        self.train_folder_path = ""
        self.val_folder_path = ""  # NEW: Added validation folder
        self.test_folder_path = ""
        self.selected_train_files = []
        self.selected_val_files = []  # NEW: Added validation files
        self.selected_test_files = []
        self.data_processor_csv = DataProcessorCSV()  # Initialize CSV processor
        self.data_processor_mat = DataProcessorMAT()  # Initialize MAT processor
        self.data_processor_xlsx = DataProcessorXLSX()  # Initialize XLSX processor

        # Resampling moved to data augmentation GUI
        self.organizer_thread = None
        self.organizer = None
        # self.is_selecting_folder_flag = False # Removed flag, will use blockSignals

        self.initUI()
        
        # Load default settings after UI is initialized
        self.load_default_settings()

    def initUI(self):
        self.setWindowTitle("VEstim Modelling Tool")
        self.setGeometry(100, 100, 900, 600)

        # Main layout
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Header
        self.header_label = QLabel("Select training, validation, and test data folders for your LSTM Model", self)
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

        # Validation folder section (moved before test)
        val_layout = QVBoxLayout()
        self.val_select_button = QPushButton("Select Validation Data Folder", self)
        self.val_select_button.setStyleSheet("""
            background-color: #ffcccb;  /* Light red background */
            font-weight: bold;
            padding: 8px 15px;  /* Adds padding inside the button */
            color: black;  /* Set the text color to black for contrast */
        """)
        self.val_select_button.setFixedHeight(30)  # Set a consistent height for the button
        self.val_select_button.setMinimumWidth(150)  # Set a reasonable minimum width
        self.val_select_button.setMaximumWidth(300)  # Set a reasonable maximum width
        self.val_select_button.clicked.connect(self.select_val_folder)

        # Center the validation button
        val_button_layout = QHBoxLayout()
        val_button_layout.addStretch(1)
        val_button_layout.addWidget(self.val_select_button, alignment=Qt.AlignCenter)
        val_button_layout.addStretch(1)
        val_layout.addLayout(val_button_layout)

        # Validation folder list widget (covers full width)
        self.val_list_widget = QListWidget(self)
        self.val_list_widget.setSelectionMode(QListWidget.MultiSelection)
        self.val_list_widget.setMinimumHeight(100)
        val_layout.addWidget(self.val_list_widget)

        # Add the validation section to the main layout
        self.main_layout.addLayout(val_layout)

        # Testing folder section (moved after validation)
        test_layout = QVBoxLayout()
        self.test_select_button = QPushButton("Select Test Data Folder", self)
        self.test_select_button.setStyleSheet("""
            background-color: #98fb98;  /* Light green background */
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
        data_source_label = QLabel("File Format:")
        data_source_label.setStyleSheet("color: purple; font-weight: bold; font-size: 14px; padding-right: 10px;")
        combined_layout.addWidget(data_source_label)

        # Data source selection with consistent height and styling
        self.data_source_combo = QComboBox(self)
        self.data_source_combo.addItems(["csv", "mat", "xlsx"])  # CSV as default/top option
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

    def on_data_source_selection_changed(self, index):
        """
        Called when the file format selection changes.
        Refreshes the file lists based on the new file format.
        """
        selected_format = self.data_source_combo.currentText()
        logger.info(f"File format changed to: {selected_format}. Refreshing file lists.")
        if self.train_folder_path:
            self.populate_file_list(self.train_folder_path, self.train_list_widget, selected_format)
            self.auto_select_all_files(self.train_list_widget)
        if self.val_folder_path:
            self.populate_file_list(self.val_folder_path, self.val_list_widget, selected_format)
            self.auto_select_all_files(self.val_list_widget)
        if self.test_folder_path:
            self.populate_file_list(self.test_folder_path, self.test_list_widget, selected_format)
            self.auto_select_all_files(self.test_list_widget)

    def select_train_folder(self):
        # Get default data directory from config, fallback to current directory
        default_dir = get_data_directory() or os.getcwd()
        new_folder = QFileDialog.getExistingDirectory(self, "Select Training Folder", default_dir)
        if new_folder:
            self.train_folder_path = new_folder
            selected_format = self.data_source_combo.currentText()
            self.populate_file_list(new_folder, self.train_list_widget, selected_format)
            self.auto_select_all_files(self.train_list_widget)
            self.update_button_text(self.train_select_button, new_folder, "Train")
            logger.info(f"Selected training folder: {new_folder}. Populated for format: {selected_format}.")
        self.check_folders_selected()

    def select_test_folder(self):
        # Get default data directory from config, fallback to current directory
        default_dir = get_data_directory() or os.getcwd()
        new_folder = QFileDialog.getExistingDirectory(self, "Select Testing Folder", default_dir)
        if new_folder:
            self.test_folder_path = new_folder
            selected_format = self.data_source_combo.currentText()
            self.populate_file_list(new_folder, self.test_list_widget, selected_format)
            self.auto_select_all_files(self.test_list_widget)
            self.update_button_text(self.test_select_button, new_folder, "Test")
            logger.info(f"Selected testing folder: {new_folder}. Populated for format: {selected_format}.")
        self.check_folders_selected()

    def select_val_folder(self):
        # Get default data directory from config, fallback to current directory
        default_dir = get_data_directory() or os.getcwd()
        new_folder = QFileDialog.getExistingDirectory(self, "Select Validation Folder", default_dir)
        if new_folder:
            self.val_folder_path = new_folder
            selected_format = self.data_source_combo.currentText()
            self.populate_file_list(new_folder, self.val_list_widget, selected_format)
            self.auto_select_all_files(self.val_list_widget)
            self.update_button_text(self.val_select_button, new_folder, "Validation")
            logger.info(f"Selected validation folder: {new_folder}. Populated for format: {selected_format}.")
        self.check_folders_selected()

    def populate_file_list(self, folder_path, list_widget, file_format):
        """
        Populate the list widget with files matching extensions for the given file_format.
        """
        list_widget.clear()
        if not folder_path or not os.path.isdir(folder_path):
            return

        if file_format == "mat":
            extensions_to_check = [".mat"]
        elif file_format == "csv":
            extensions_to_check = [".csv"]
        elif file_format == "xlsx":
            extensions_to_check = [".xlsx", ".xls"] # Excel files
        else: # Default fallback
            extensions_to_check = [ext.lower() for ext in DEFAULT_DATA_EXTENSIONS]
        
        logger.info(f"Populating list for '{folder_path}' (Format: {file_format}). Scanning for extensions: {extensions_to_check}")
        
        items_added_count = 0
        try:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in extensions_to_check):
                        full_path = os.path.join(root, file)
                        shortened_path = shorten_path_for_display(full_path)
                        
                        # Create a list widget item with shortened display text
                        from PyQt5.QtWidgets import QListWidgetItem
                        item = QListWidgetItem(shortened_path)
                        # Store the full path as item data for backend use
                        item.setData(Qt.UserRole, full_path)
                        list_widget.addItem(item)
                        items_added_count +=1
        except Exception as e:
            logger.error(f"Error during file iteration or adding item for {folder_path}: {e}", exc_info=True)
        
        logger.info(f"Finished populating list for '{folder_path}'. Total items added: {items_added_count}. List widget current count: {list_widget.count()}")
        list_widget.update() # Explicitly request a widget update

    def check_folders_selected(self):
        if self.train_folder_path and self.val_folder_path and self.test_folder_path:
            self.organize_button.setEnabled(True)
        else:
            self.organize_button.setEnabled(False)

    def organize_files(self):
        
        logger.info("Starting file organization process...")
        # Use selectedItems() to get the selected files - get full paths from stored data
        train_files = [item.data(Qt.UserRole) for item in self.train_list_widget.selectedItems()]
        val_files = [item.data(Qt.UserRole) for item in self.val_list_widget.selectedItems()]
        test_files = [item.data(Qt.UserRole) for item in self.test_list_widget.selectedItems()]
        print(f"Train files: {train_files}")
        print(f"Validation files: {val_files}")
        print(f"Test files: {test_files}")


        if not train_files or not val_files or not test_files:
            self.show_error("No files selected for training, validation, or testing.")
            return
            
        # Save current settings as new defaults
        selected_format = self.data_source_combo.currentText()
        update_last_used_folders(
            train_folder=self.train_folder_path,
            val_folder=self.val_folder_path, 
            test_folder=self.test_folder_path,
            file_format=selected_format
        )
        logger.info(f"Saved current settings as defaults: Train={self.train_folder_path}, Val={self.val_folder_path}, Test={self.test_folder_path}, Format={selected_format}")
        
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

        # Determine which data processor to use based on the selected file format
        selected_format = self.data_source_combo.currentText()
        if selected_format == "mat":
            data_processor = self.data_processor_mat
        elif selected_format == "xlsx":
            data_processor = self.data_processor_xlsx
        elif selected_format == "csv":
            data_processor = self.data_processor_csv
        else:
            self.show_error("Invalid file format selected.")
            return

        # Create and start the file organizer thread with the selected data processor
        # Removed sampling_frequency parameter as this is now handled in the data augmentation GUI
        self.organizer = FileOrganizer(train_files, val_files, test_files, data_processor)
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
        self.organize_button.setText("Continue to Data Augmentation")
        self.organize_button.setStyleSheet("""
            background-color: #1f8b4c; 
            font-weight: bold;
            padding: 10px 20px;
            color: white;
            font-size: 14px;
        """)
        self.organize_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.organize_button.setEnabled(True)
        
        # Update the button action to move to the next screen
        self.organize_button.clicked.disconnect()
        self.organize_button.clicked.connect(lambda: self.move_to_next_screen(job_folder))

    def move_to_next_screen(self, job_folder):
        # Changed to open the Data Augmentation GUI instead of Hyperparameter GUI
        self.close()
        self.data_augment_gui = DataAugmentGUI(job_manager=self.job_manager)
        self.data_augment_gui.show()

    def show_error(self, message):
        # Display error message
        QMessageBox.critical(self, "Error", message)

    def load_default_settings(self):
        """Load default settings and populate the GUI with last used folders"""
        try:
            default_settings = get_default_folders()
            
            # Set default file format
            default_format = get_default_file_format()
            format_index = self.data_source_combo.findText(default_format)
            if format_index >= 0:
                self.data_source_combo.setCurrentIndex(format_index)
            
            # Load default folder paths
            train_folder = default_settings.get("train_folder", "")
            val_folder = default_settings.get("val_folder", "")
            test_folder = default_settings.get("test_folder", "")
            
            # Auto-populate folders if they exist and contain files
            if train_folder and os.path.exists(train_folder):
                self.train_folder_path = train_folder
                self.populate_file_list(train_folder, self.train_list_widget, default_format)
                self.auto_select_all_files(self.train_list_widget)
                self.update_button_text(self.train_select_button, train_folder, "Train")
                logger.info(f"Auto-loaded training folder: {train_folder}")
            
            if val_folder and os.path.exists(val_folder):
                self.val_folder_path = val_folder
                self.populate_file_list(val_folder, self.val_list_widget, default_format)
                self.auto_select_all_files(self.val_list_widget)
                self.update_button_text(self.val_select_button, val_folder, "Validation")
                logger.info(f"Auto-loaded validation folder: {val_folder}")
            
            if test_folder and os.path.exists(test_folder):
                self.test_folder_path = test_folder
                self.populate_file_list(test_folder, self.test_list_widget, default_format)
                self.auto_select_all_files(self.test_list_widget)
                self.update_button_text(self.test_select_button, test_folder, "Test")
                logger.info(f"Auto-loaded test folder: {test_folder}")
            
            # Check if all folders are loaded
            self.check_folders_selected()
            
            # Update header text if defaults are loaded
            if self.train_folder_path and self.val_folder_path and self.test_folder_path:
                self.header_label.setText("Default data folders loaded. Select different folders if needed, or click 'Load and Prepare Files' to proceed.")
                self.header_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #0b6337;")
                
        except Exception as e:
            logger.error(f"Error loading default settings: {e}")
    
    def auto_select_all_files(self, list_widget):
        """Automatically select all files in a list widget"""
        try:
            for i in range(list_widget.count()):
                item = list_widget.item(i)
                if item:
                    item.setSelected(True)
        except Exception as e:
            logger.error(f"Error auto-selecting files: {e}")
    
    def update_button_text(self, button, folder_path, folder_type):
        """Update button text to show the loaded folder"""
        try:
            folder_name = os.path.basename(folder_path)
            button.setText(f"{folder_type}: {folder_name}")
        except Exception as e:
            logger.error(f"Error updating button text: {e}")

class FileOrganizer(QObject):
    progress = pyqtSignal(int)  # Emit progress percentage
    job_folder_signal = pyqtSignal(str)  # To communicate when the job folder is created

    def __init__(self, train_files, val_files, test_files, data_processor, sampling_frequency=None):
        super().__init__()
        self.train_files = train_files
        self.val_files = val_files
        self.test_files = test_files
        self.data_processor = data_processor
        self.sampling_frequency = sampling_frequency  # Keep for backwards compatibility with existing code

    def run(self):
        if not self.train_files or not self.val_files or not self.test_files:
            self.progress.emit(0)  # Emit 0% if no files selected
            return

        try:
            # Call the backend method from DataProcessor to organize and convert files
            # Now the data processors support three sets of files: train, val, test
            job_folder = self.data_processor.organize_and_convert_files(
                self.train_files,   # Training files
                self.val_files,     # Validation files  
                self.test_files,    # Test files
                progress_callback=self.update_progress, 
                sampling_frequency=None  # Remove resampling here as it's moved to data augmentation
            )
            
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
    
    # Initialize configuration manager early
    from vestim.config_manager import get_config_manager
    config_manager = get_config_manager()
    
    # Log the projects directory being used
    projects_dir = config_manager.get_projects_directory()
    logger.info(f"Vestim starting - Projects directory: {projects_dir}")
    
    gui = DataImportGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
