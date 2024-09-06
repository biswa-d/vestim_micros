from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QListWidget, QFileDialog, QProgressBar, QWidget, QMessageBox
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject

import os, sys
from src.gui.src.hyper_param_gui_qt import VEstimHyperParamGUI  # Adjust this import based on your actual path
from src.services.data_processor.src.data_processor_qt import DataProcessor

class DataImportGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.train_folder_path = ""
        self.test_folder_path = ""
        self.selected_train_files = []
        self.selected_test_files = []
        self.data_processor = DataProcessor()  # Initialize DataProcessor

        self.organizer_thread = None
        self.organizer = None

        self.initUI()

    def initUI(self):
        self.setWindowTitle("VEstim Modelling Tool")
        self.setGeometry(100, 100, 800, 600)

        # Main layout
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Header
        self.header_label = QLabel("Select data folders to train your LSTM Model", self)
        self.header_label.setAlignment(Qt.AlignCenter)
        self.header_label.setStyleSheet("font-size: 16px; font-weight: bold; color: green;")
        self.main_layout.addWidget(self.header_label)

        # Folder selection layout (now horizontal)
        folder_layout = QHBoxLayout()

        # Training folder section
        train_layout = QVBoxLayout()
        self.train_select_button = QPushButton("Select Training Folder", self)
        self.train_select_button.clicked.connect(self.select_train_folder)
        self.train_list_widget = QListWidget(self)
        self.train_list_widget.setSelectionMode(QListWidget.MultiSelection)
        train_layout.addWidget(self.train_select_button)
        train_layout.addWidget(self.train_list_widget)

        # Testing folder section
        test_layout = QVBoxLayout()
        self.test_select_button = QPushButton("Select Testing Folder", self)
        self.test_select_button.clicked.connect(self.select_test_folder)
        self.test_list_widget = QListWidget(self)
        self.test_list_widget.setSelectionMode(QListWidget.MultiSelection)
        test_layout.addWidget(self.test_select_button)
        test_layout.addWidget(self.test_list_widget)

        # Add training and testing layouts to the folder layout
        folder_layout.addLayout(train_layout)
        folder_layout.addLayout(test_layout)
        self.main_layout.addLayout(folder_layout)

        # Organize button
        self.organize_button = QPushButton("Load and Prepare Files", self)
        self.organize_button.setStyleSheet("""
            background-color: #0b6337; 
            font-weight: bold; 
            padding: 10px 20px;  /* Adds padding inside the button */
            color: white;  /* Set the text color to white */
        """)
        # self.organize_button.setFixedSize(self.organize_button.sizeHint())  # Size to fit the text + some padding
        self.organize_button.adjustSize()
        self.organize_button.setEnabled(False)
        self.organize_button.clicked.connect(self.organize_files)

        # Center the button using a layout
        button_layout = QHBoxLayout()
        button_layout.addStretch(1)  # Add stretchable space before the button
        button_layout.addWidget(self.organize_button, alignment=Qt.AlignCenter)
        button_layout.addStretch(1)  # Add stretchable space after the button

        # Add padding around the button by setting the margins
        button_layout.setContentsMargins(50, 20, 50, 20)  # Add margins (left, top, right, bottom)

        # Add the button layout to the main layout
        self.main_layout.addLayout(button_layout)


        # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)  # Initially hidden
        self.main_layout.addWidget(self.progress_bar)

    def select_train_folder(self):
        self.train_folder_path = QFileDialog.getExistingDirectory(self, "Select Training Folder")
        if self.train_folder_path:
            self.populate_file_list(self.train_folder_path, self.train_list_widget)
        self.check_folders_selected()

    def select_test_folder(self):
        self.test_folder_path = QFileDialog.getExistingDirectory(self, "Select Testing Folder")
        if self.test_folder_path:
            self.populate_file_list(self.test_folder_path, self.test_list_widget)
        self.check_folders_selected()

    def populate_file_list(self, folder_path, list_widget):
        list_widget.clear()
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".mat"):
                    list_widget.addItem(os.path.join(root, file))

    def check_folders_selected(self):
        if self.train_folder_path and self.test_folder_path:
            self.organize_button.setEnabled(True)
        else:
            self.organize_button.setEnabled(False)

    def organize_files(self):
        # Update the button label to indicate the process has started
        self.organize_button.setText("Importing and Preprocessing Files")
        self.organize_button.setStyleSheet("background-color: #3ecf86; font-weight: bold;")  # Light red to show it's disabled
        self.organize_button.setEnabled(False)  # Optionally disable the button during processing
        self.organize_button.adjustSize()
        # Show progress label and start the background thread
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Use selectedItems() to get the selected files
        train_files = [item.text() for item in self.train_list_widget.selectedItems()]
        test_files = [item.text() for item in self.test_list_widget.selectedItems()]

        if not train_files or not test_files:
            self.show_error("No files selected for either training or testing.")
            return

        # Create and start the file organizer thread
        self.organizer = FileOrganizer(train_files, test_files, self.data_processor)
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
        self.progress_bar.setVisible(False)
        
        # Change the button label to indicate next step and enable it
        self.organize_button.setText("Proceed to Hyperparameter Selection")
        self.organize_button.setStyleSheet("""
            background-color: #1f8b4c; 
            font-weight: bold;
            padding: 10px 20px;
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
            
            # Emit success message with job folder details
            self.job_folder_signal.emit(job_folder)
        except Exception as e:
            self.progress.emit(0)  # Emit 0% if there is an error
            print(f"Error occurred: {e}")

    def update_progress(self, progress_value):
        """Emit progress as a percentage."""
        self.progress.emit(progress_value)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = DataImportGUI()
    gui.show()
    sys.exit(app.exec_())
