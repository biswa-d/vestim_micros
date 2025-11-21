import sys
import os
import json
import pandas as pd
import datetime
import traceback
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QFileDialog, QWidget, QListWidget, QListWidgetItem, QMessageBox, QTextEdit)
from PyQt5.QtCore import Qt
from vestim.gui.src.adaptive_gui_utils import get_adaptive_stylesheet, scale_widget_size
from vestim.gateway.src.standalone_testing_manager_qt import VEstimStandaloneTestingManager
from vestim.gui.src.standalone_augmentation_gui_qt import StandaloneAugmentationGUI
from vestim.gui.src.standalone_testing_gui_qt import VEstimStandaloneTestingGUI

class TestSelectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Test a Trained Model")
        self.job_folder_path = ""
        self.test_files = []
        self.current_file_index = 0
        self.session_timestamp = None  # Will be set when testing starts
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

        # Job folder selection with better description
        job_desc = QLabel(
            "Please select a job folder which was created from running the PyBattML tool "
            "containing the model and task directory with trained model."
        )
        job_desc.setWordWrap(True)
        job_desc.setStyleSheet(get_adaptive_stylesheet("""
            color: #495057; font-size: 10pt; margin: 10px 0px; 
            background-color: #f8f9fa; border-radius: 4px; padding: 8px;
        """))
        self.main_layout.addWidget(job_desc)
        
        job_layout = QHBoxLayout()
        self.job_folder_button = QPushButton("Select Job Folder")
        self.job_folder_button.setToolTip("Browse for a job folder containing trained models")
        self.job_folder_button.clicked.connect(self.select_job_folder)
        job_layout.addWidget(self.job_folder_button)
        
        self.job_path_label = QLabel("Job Folder: Not Selected")
        self.job_path_label.setStyleSheet(get_adaptive_stylesheet("font-size: 11pt; color: #666;"))
        job_layout.addWidget(self.job_path_label, 1)
        self.main_layout.addLayout(job_layout)

        # Test files selection with better description  
        test_files_label = QLabel("Test Data Configuration")
        test_files_label.setStyleSheet(get_adaptive_stylesheet("font-size: 12pt; font-weight: bold; color: #0b6337; margin-top: 15px;"))
        self.main_layout.addWidget(test_files_label)
        
        test_desc = QLabel(
            "Add CSV files with the same column structure as training data. "
            "Must include target columns (voltage, SOC, etc.) for error calculation."
        )
        test_desc.setWordWrap(True)
        test_desc.setStyleSheet(get_adaptive_stylesheet("color: #6c757d; font-size: 9pt; margin-bottom: 10px;"))
        self.main_layout.addWidget(test_desc)
        
        files_layout = QHBoxLayout()
        self.add_files_button = QPushButton("Add Test Files")
        self.add_files_button.setToolTip("Select CSV files containing test data with target variables")
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
        self.run_test_button.setToolTip("Begin testing all models against selected test data")
        self.run_test_button.clicked.connect(self.start_testing)
        self.run_test_button.setStyleSheet(get_adaptive_stylesheet("""
            QPushButton {
                font-size: 14pt; font-weight: bold; padding: 15px 30px;
                background-color: #28a745; color: white; border-radius: 8px;
            }
            QPushButton:hover { background-color: #218838; }
            QPushButton:disabled { background-color: #6c757d; }
        """))
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
        files, _ = QFileDialog.getOpenFileNames(self, "Select Test Data Files", "", "Data Files (*.csv *.xlsx *.mat)")
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
        
        # Generate session timestamp for consistent folder organization
        self.session_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Disable the button and clear log
        self.run_test_button.setEnabled(False)
        self.run_test_button.setText("Testing in Progress...")
        self.log_widget.clear()
        
        # Process all selected files with a single GUI
        self.log_widget.append(f"Starting testing with {len(self.test_files)} file(s)")
        
        # Launch the testing GUI ONCE to show results from all files
        self.log_widget.append("Launching standalone testing GUI...")
        self.launch_testing_gui()
        
        # Process files sequentially
        self.current_file_index = 0
        self.start_next_file_testing()
    
    def start_next_file_testing(self):
        """Start testing for the next file in the list"""
        if self.current_file_index >= len(self.test_files):
            # All files processed
            self.all_testing_completed()
            return
            
        test_file = self.test_files[self.current_file_index]
        file_name = os.path.basename(test_file)
        self.log_widget.append(f"Processing file {self.current_file_index + 1}/{len(self.test_files)}: {file_name}")
        self.log_widget.append("Checking augmentation requirements...")
        
        # Start the testing manager in background and then launch GUI
        QApplication.processEvents()  # Update UI
        
        try:
            # Use the imported standalone testing manager
            self.log_widget.append("Initializing testing manager...")
            QApplication.processEvents()
            
            # The manager will handle augmentation and testing for this file
            self.testing_manager = VEstimStandaloneTestingManager(self.job_folder_path, test_file, self.session_timestamp)
            
            # Connect to manager signals for progress updates
            self.testing_manager.progress.connect(self.update_log)
            self.testing_manager.augmentation_required.connect(self.handle_augmentation_required)
            self.testing_manager.finished.connect(self.testing_completed)
            self.testing_manager.error.connect(self.testing_error)
            
            # Connect results to the existing GUI (created once)
            if hasattr(self, 'testing_gui') and self.testing_gui:
                self.testing_manager.results_ready.connect(self.testing_gui.add_result_row)
                self.log_widget.append("Connected results to existing GUI")
            else:
                self.log_widget.append("Warning: No testing GUI available for results")
            
            # Start the testing process
            self.log_widget.append("Starting testing manager...")
            QApplication.processEvents()
            self.testing_manager.start()
            
        except Exception as e:
            self.log_widget.append(f"Error initializing testing: {str(e)}")
            self.testing_error(str(e))
    
    def update_log(self, message):
        """Update the log with progress messages - send to terminal for debugging"""
        print(f"[TESTING] {message}")
        # Also keep GUI log for critical messages only
        if ("Error" in message) or ("completed" in message):
            self.log_widget.append(message)
        QApplication.processEvents()
    
    def handle_augmentation_required(self, test_df, filter_configs):
        """Handle when manual augmentation is required - launch augmentation GUI"""
        self.log_widget.append("Manual augmentation required - opening augmentation GUI...")
        QApplication.processEvents()
        
        try:
            # Use the imported standalone augmentation GUI
            self.log_widget.append("Launching standalone augmentation interface...")
            QApplication.processEvents()
            
            # Create and show the augmentation GUI with the test data and filter configs
            self.augmentation_gui = StandaloneAugmentationGUI(test_df, filter_configs)
            
            # Connect completion signal to continue testing
            self.augmentation_gui.augmentation_completed.connect(self.continue_testing_after_augmentation)
            
            self.augmentation_gui.show()
            
            self.log_widget.append("Augmentation GUI opened. Apply required filters and click 'Continue with Testing'")
            
        except ImportError as e:
            self.log_widget.append(f"Could not import augmentation GUI: {e}")
            self.log_widget.append("Continuing with original data (may cause normalization errors)")
            if hasattr(self, 'testing_manager'):
                self.testing_manager.resume_test_with_augmented_data(test_df)
        except Exception as e:
            self.log_widget.append(f"Error launching augmentation GUI: {e}")
            self.run_test_button.setEnabled(True)
            self.run_test_button.setText("Start Testing")
    
    def continue_testing_after_augmentation(self, augmented_df):
        """Continue testing with the augmented data"""
        self.log_widget.append("Augmentation completed. Continuing with testing...")
        QApplication.processEvents()
        
        if hasattr(self, 'testing_manager'):
            self.testing_manager.resume_test_with_augmented_data(augmented_df)
        else:
            self.log_widget.append("Error: Testing manager not available")
            self.run_test_button.setEnabled(True)
            self.run_test_button.setText("Start Testing")
    
    def testing_completed(self):
        """Handle when testing is completed for current file"""
        current_file = os.path.basename(self.test_files[self.current_file_index])
        self.log_widget.append(f"Testing completed for: {current_file}")
        
        # Move to next file
        self.current_file_index += 1
        if self.current_file_index < len(self.test_files):
            self.log_widget.append(f"Moving to next file...")
            QApplication.processEvents()
            self.start_next_file_testing()
        else:
            self.all_testing_completed()
    
    def all_testing_completed(self):
        """Handle when all files have been tested"""
        self.log_widget.append("All files tested successfully.")
        self.log_widget.append("Check the testing results windows for model results.")
        
        # Create consolidated summary CSV in main job folder
        if hasattr(self, 'testing_gui') and self.testing_gui:
            print(f"[DEBUG] testing_gui type: {type(self.testing_gui)}")
            print(f"[DEBUG] Has create_consolidated_summary_csv: {hasattr(self.testing_gui, 'create_consolidated_summary_csv')}")
            
            # Notify testing GUI that all testing is complete
            if hasattr(self.testing_gui, 'show_completion_message'):
                try:
                    self.testing_gui.show_completion_message()
                    print("[DEBUG] Called show_completion_message on testing GUI")
                except Exception as e:
                    print(f"[ERROR] Failed to call show_completion_message: {e}")
            
            if hasattr(self.testing_gui, 'create_consolidated_summary_csv'):
                try:
                    self.testing_gui.create_consolidated_summary_csv()
                    self.log_widget.append(f"Standalone testing summary saved: standalone_testing_summary_{self.session_timestamp}.csv")
                except Exception as e:
                    print(f"[ERROR] Failed to call create_consolidated_summary_csv: {e}")
                    traceback.print_exc()
                    self.log_widget.append(f"Warning: Could not create consolidated summary: {e}")
            else:
                print(f"[ERROR] testing_gui object does not have create_consolidated_summary_csv method")
                print(f"[DEBUG] Available methods: {[m for m in dir(self.testing_gui) if not m.startswith('_')][:10]}")
                self.log_widget.append("Warning: Could not create consolidated summary - method not found")
        else:
            print(f"[DEBUG] No testing_gui available: hasattr={hasattr(self, 'testing_gui')}, value={getattr(self, 'testing_gui', None)}")
            self.log_widget.append("Warning: No testing GUI available for consolidated summary")
        
        QApplication.processEvents()
        
        # Keep button disabled after completion to prevent re-running
        self.run_test_button.setText("All Testing Complete")
        self.run_test_button.setEnabled(False)
        self.run_test_button.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                font-size: 14pt;
                font-weight: bold;
                padding: 15px;
                border-radius: 8px;
                border: 2px solid #1e7e34;
            }
        """)
    
    def testing_error(self, error_msg):
        """Handle testing errors"""
        current_file = os.path.basename(self.test_files[self.current_file_index])
        self.log_widget.append(f"Testing error for {current_file}: {error_msg}")
        
        # Move to next file even after error
        self.current_file_index += 1
        if self.current_file_index < len(self.test_files):
            self.log_widget.append(f"Continuing with next file...")
            QApplication.processEvents()
            self.start_next_file_testing()
        else:
            self.log_widget.append("All files processed (some with errors)")
            self.run_test_button.setEnabled(True)
            self.run_test_button.setText("Start Testing")

    def launch_testing_gui(self):
        """Launch the standalone testing GUI once to show results from all files"""
        try:
            print(f"[DEBUG] Creating VEstimStandaloneTestingGUI with job_folder_path={self.job_folder_path}, session_timestamp={self.session_timestamp}")
            
            # Create the GUI only once to show results from all files - pass session timestamp
            self.testing_gui = VEstimStandaloneTestingGUI(self.job_folder_path, self.session_timestamp)
            self.results_gui = self.testing_gui
            
            print(f"[DEBUG] VEstimStandaloneTestingGUI created successfully: {type(self.testing_gui)}")
            print(f"[DEBUG] Has create_consolidated_summary_csv: {hasattr(self.testing_gui, 'create_consolidated_summary_csv')}")
            
            # Show the GUI - it will receive results as testing progresses
            self.testing_gui.show()
            
            print("[DEBUG] Standalone Testing GUI opened and ready to receive results from all files!")
            
        except Exception as e:
            self.log_widget.append(f"Failed to launch testing GUI: {str(e)}")
            traceback.print_exc()
            print(f"[ERROR] Exception during testing GUI creation: {e}")
            self.run_test_button.setEnabled(True)
            self.run_test_button.setText("Start Testing")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    test_selection_screen = TestSelectionGUI()
    test_selection_screen.show()
    sys.exit(app.exec_())