import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QFileDialog, QWidget, QListWidget, QListWidgetItem, QMessageBox, QTextEdit)
from PyQt5.QtCore import Qt
from vestim.gui.src.adaptive_gui_utils import get_adaptive_stylesheet, scale_widget_size

class TestSelectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Test a Trained Model")
        self.job_folder_path = ""
        self.test_files = []
        self.initUI()

    def initUI(self):
        self.setGeometry(100, 100, scale_widget_size(1000), scale_widget_size(750))
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setSpacing(20)
        self.main_layout.setContentsMargins(30, 30, 30, 30)

        # Header
        header_label = QLabel("Select Job and Test Data")
        header_label.setAlignment(Qt.AlignCenter)
        header_label.setStyleSheet(get_adaptive_stylesheet("font-size: 20pt; font-weight: bold; color: #0b6337;"))
        self.main_layout.addWidget(header_label)

        # Job folder selection
        job_layout = QHBoxLayout()
        self.job_folder_button = QPushButton("Select Job Folder")
        self.job_folder_button.clicked.connect(self.select_job_folder)
        job_layout.addWidget(self.job_folder_button)
        
        self.job_path_label = QLabel("Job Folder: Not Selected")
        self.job_path_label.setStyleSheet(get_adaptive_stylesheet("font-size: 11pt; color: #666;"))
        job_layout.addWidget(self.job_path_label, 1)
        self.main_layout.addLayout(job_layout)

        # Test files selection
        test_files_label = QLabel("Test Data Files:")
        test_files_label.setStyleSheet(get_adaptive_stylesheet("font-size: 12pt; font-weight: bold;"))
        self.main_layout.addWidget(test_files_label)
        
        files_layout = QHBoxLayout()
        self.add_files_button = QPushButton("Add Test Files")
        self.add_files_button.clicked.connect(self.add_test_files)
        files_layout.addWidget(self.add_files_button)
        
        self.clear_files_button = QPushButton("Clear All")
        self.clear_files_button.clicked.connect(self.clear_test_files)
        files_layout.addWidget(self.clear_files_button)
        files_layout.addStretch()
        self.main_layout.addLayout(files_layout)

        # Files list
        self.files_list = QListWidget()
        self.files_list.setMaximumHeight(200)
        self.main_layout.addWidget(self.files_list)

        # Run button
        self.run_test_button = QPushButton("Start Testing")
        self.run_test_button.setEnabled(False)
        self.run_test_button.clicked.connect(self.start_testing)
        self.main_layout.addWidget(self.run_test_button, alignment=Qt.AlignCenter)

        # Add log window for testing progress
        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setMaximumHeight(200)
        self.log_widget.setPlaceholderText("Testing progress will be shown here...")
        self.log_widget.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 8px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 10px;
            }
        """)
        self.main_layout.addWidget(self.log_widget)

        self.main_layout.addStretch()
        self.apply_styles()

    def apply_styles(self):
        button_style = get_adaptive_stylesheet("""
            QPushButton {
                font-size: 12pt;
                padding: 10px 20px;
                border-radius: 6px;
                background-color: #add8e6;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #87ceeb;
            }
        """)
        self.job_folder_button.setStyleSheet(button_style)
        self.add_files_button.setStyleSheet(button_style)
        self.clear_files_button.setStyleSheet(button_style)

        run_button_style = get_adaptive_stylesheet("""
            QPushButton {
                font-size: 14pt;
                padding: 15px 30px;
                border-radius: 8px;
                background-color: #0b6337;
                color: white;
                font-weight: bold;
                min-width: 250px;
            }
            QPushButton:hover {
                background-color: #094D2A;
            }
            QPushButton:disabled {
                background-color: #d3d3d3;
                color: #a9a9a9;
            }
        """)
        self.run_test_button.setStyleSheet(run_button_style)
        self.central_widget.setStyleSheet("background-color: #f8f9fa;")

    def select_job_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Job Folder")
        if folder:
            # Validate job folder
            if not self._validate_job_folder(folder):
                QMessageBox.warning(self, "Invalid Job Folder", 
                                  "Selected folder is not a valid VEstim job folder.\n"
                                  "Please select a folder containing hyperparams.json and job_metadata.json")
                return
                
            self.job_folder_path = folder
            self.job_path_label.setText(f"Job Folder: {os.path.basename(folder)}")
            self.check_ready()
    
    def _validate_job_folder(self, folder):
        """Validate that the selected folder is a valid VEstim job folder"""
        required_files = ['hyperparams.json', 'job_metadata.json']
        for file in required_files:
            if not os.path.exists(os.path.join(folder, file)):
                return False
        return True

    def add_test_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Test Data Files", "", "CSV Files (*.csv)")
        for file in files:
            if file not in self.test_files:
                self.test_files.append(file)
                item = QListWidgetItem(os.path.basename(file))
                item.setToolTip(file)  # Show full path on hover
                self.files_list.addItem(item)
        self.check_ready()

    def clear_test_files(self):
        self.test_files.clear()
        self.files_list.clear()
        self.check_ready()

    def check_ready(self):
        self.run_test_button.setEnabled(self.job_folder_path and len(self.test_files) > 0)

    def start_testing(self):
        """Start the testing process: augmentation → backend testing → launch GUI"""
        if not self.test_files:
            QMessageBox.warning(self, "No Files", "Please select at least one test file.")
            return
        
        # Disable the button and clear log
        self.run_test_button.setEnabled(False)
        self.run_test_button.setText("Testing in Progress...")
        self.log_widget.clear()
        
        # For now, start with the first file (we can enhance for multiple files later)
        test_file = self.test_files[0]
        self.log_widget.append(f"🚀 Starting testing with file: {os.path.basename(test_file)}")
        self.log_widget.append("📋 Checking augmentation requirements...")
        
        # Start the testing manager in background and then launch GUI
        QApplication.processEvents()  # Update UI
        
        try:
            # Import and start the standalone testing manager
            from vestim.gateway.src.standalone_testing_manager_qt import VEstimStandaloneTestingManager
            
            self.log_widget.append("🔧 Initializing testing manager...")
            QApplication.processEvents()
            
            # The manager will handle augmentation and testing, then we launch GUI
            self.testing_manager = VEstimStandaloneTestingManager(self.job_folder_path, test_file)
            
            # Connect to manager signals for progress updates
            self.testing_manager.progress.connect(self.update_log)
            self.testing_manager.augmentation_required.connect(self.handle_augmentation_required)
            self.testing_manager.finished.connect(self.testing_completed)
            self.testing_manager.error.connect(self.testing_error)
            
            # Store reference to GUI for result sharing
            self.results_gui = None
            
            self.log_widget.append("🔗 Connected to testing manager signals")
            
            # Launch the testing GUI BEFORE starting the testing manager
            # so it can receive results as they come in
            self.log_widget.append("🎯 Launching standalone testing GUI...")
            self.launch_testing_gui()
            
            QApplication.processEvents()
            
            # Start the testing process
            self.log_widget.append("▶️ Starting testing manager...")
            QApplication.processEvents()
            self.testing_manager.start()
            
        except Exception as e:
            self.log_widget.append(f"❌ Error initializing testing: {str(e)}")
            self.run_test_button.setEnabled(True)
            self.run_test_button.setText("Start Testing")
    
    def update_log(self, message):
        """Update the log with progress messages"""
        self.log_widget.append(message)
        QApplication.processEvents()
    
    def handle_augmentation_required(self, test_df, filter_configs):
        """Handle when manual augmentation is required - launch augmentation GUI"""
        self.log_widget.append("⚠️ Manual augmentation required - opening augmentation GUI...")
        QApplication.processEvents()
        
        try:
            # Import the standalone augmentation GUI
            from vestim.gui.src.standalone_augmentation_gui_qt import StandaloneAugmentationGUI
            
            self.log_widget.append("🔧 Launching standalone augmentation interface...")
            QApplication.processEvents()
            
            # Create and show the augmentation GUI with the test data and filter configs
            self.augmentation_gui = StandaloneAugmentationGUI(test_df, filter_configs)
            
            # Connect completion signal to continue testing
            self.augmentation_gui.augmentation_completed.connect(self.continue_testing_after_augmentation)
            
            self.augmentation_gui.show()
            
            self.log_widget.append("🔧 Augmentation GUI opened. Apply required filters and click 'Continue with Testing'")
            
        except ImportError as e:
            self.log_widget.append(f"❌ Could not import augmentation GUI: {e}")
            self.log_widget.append("⚠️ Continuing with original data (may cause normalization errors)")
            if hasattr(self, 'testing_manager'):
                self.testing_manager.resume_test_with_augmented_data(test_df)
        except Exception as e:
            self.log_widget.append(f"❌ Error launching augmentation GUI: {e}")
            self.run_test_button.setEnabled(True)
            self.run_test_button.setText("Start Testing")
    
    def continue_testing_after_augmentation(self, augmented_df):
        """Continue testing with the augmented data"""
        self.log_widget.append("✅ Augmentation completed! Continuing with testing...")
        QApplication.processEvents()
        
        if hasattr(self, 'testing_manager'):
            self.testing_manager.resume_test_with_augmented_data(augmented_df)
        else:
            self.log_widget.append("❌ Error: Testing manager not available")
            self.run_test_button.setEnabled(True)
            self.run_test_button.setText("Start Testing")
    
    def testing_completed(self):
        """Handle when testing is completed"""
        self.log_widget.append("✅ Backend testing completed!")
        self.log_widget.append("📊 Check the testing results window for all model results!")
        QApplication.processEvents()
        
        # Update button to show testing is complete
        self.run_test_button.setText("Testing Complete")
        self.run_test_button.setEnabled(True)
    
    def testing_error(self, error_msg):
        """Handle testing errors"""
        self.log_widget.append(f"❌ Testing error: {error_msg}")
        self.run_test_button.setEnabled(True)
        self.run_test_button.setText("Start Testing")

    def launch_testing_gui(self):
        """Launch the standalone testing GUI before testing starts"""
        try:
            from vestim.gui.src.standalone_testing_gui_qt import VEstimStandaloneTestingGUI
            
            # Launch the GUI - it will show results as they come in
            self.testing_gui = VEstimStandaloneTestingGUI(self.job_folder_path)
            self.results_gui = self.testing_gui
            
            # Connect the results_ready signal from manager to GUI
            if hasattr(self.testing_manager, 'results_ready'):
                self.testing_manager.results_ready.connect(self.testing_gui.add_result_row)
                self.log_widget.append("🔗 Connected results signal to testing GUI")
            else:
                self.log_widget.append("⚠️ Warning: testing_manager has no results_ready signal")
            
            self.testing_gui.show()
            
            self.log_widget.append("🎯 Standalone Testing GUI opened and ready to receive results!")
            
        except Exception as e:
            self.log_widget.append(f"❌ Failed to launch testing GUI: {str(e)}")
            import traceback
            traceback.print_exc()
            self.run_test_button.setEnabled(True)
            self.run_test_button.setText("Start Testing")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    test_selection_screen = TestSelectionGUI()
    test_selection_screen.show()
    sys.exit(app.exec_())