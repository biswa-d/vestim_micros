#!/usr/bin/env python3
"""
PyBattML Complete Installer
User-friendly installer that guides users through choosing installation and project directories
"""

import sys
import os
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QLabel, QLineEdit, QFileDialog, 
                             QProgressDialog, QMessageBox, QGroupBox, QTextEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon


class SetupWorker(QThread):
    """Background worker for environment setup"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, install_dir, project_dir):
        super().__init__()
        self.install_dir = install_dir
        self.project_dir = project_dir
    
    def run(self):
        try:
            # Import and run the setup directly (avoid subprocess issues in PyInstaller)
            import sys
            from pathlib import Path
            
            # Add vestim to path if running from PyInstaller bundle
            if hasattr(sys, '_MEIPASS'):
                vestim_path = Path(sys._MEIPASS)
                if str(vestim_path) not in sys.path:
                    sys.path.insert(0, str(vestim_path))
            
            from vestim.smart_environment_setup import SmartEnvironmentSetup
            
            # Create log callback to emit progress
            def progress_callback(message):
                self.progress.emit(message)
                # Also print to console for debugging
                print(message)
                sys.stdout.flush()
            
            # Run setup directly
            print("\n" + "="*80)
            print("Starting PyBattML Environment Setup")
            print("="*80 + "\n")
            
            setup = SmartEnvironmentSetup(self.install_dir, self.project_dir, progress_callback)
            success = setup.run_full_setup()
            
            if success:
                result_msg = "Setup completed successfully!"
                print("\n" + "="*80)
                print("INSTALLATION SUCCESSFUL")
                print("="*80 + "\n")
            else:
                result_msg = "Setup failed - check console output for details"
                print("\n" + "="*80)
                print("INSTALLATION FAILED")
                print("="*80)
                print("Check the setup.log file at:", setup.log_file)
                print("="*80 + "\n")
            
            self.finished.emit(success, result_msg)
            
        except Exception as e:
            import traceback
            error_msg = f"Setup failed: {e}\n\nFull traceback:\n{traceback.format_exc()}"
            self.progress.emit(error_msg)
            
            # Print detailed error to console
            print("\n" + "="*80)
            print("CRITICAL ERROR DURING INSTALLATION")
            print("="*80)
            print(error_msg)
            print("="*80 + "\n")
            
            self.finished.emit(False, error_msg)


class PyBattMLInstaller(QMainWindow):
    """Main installer window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyBattML Complete Installer")
        self.setMinimumSize(600, 500)
        
        # Default directories
        self.install_dir = str(Path.home() / "PyBattML")
        self.project_dir = str(Path.home() / "Documents" / "PyBattML_Projects")
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Title
        title = QLabel("PyBattML - Python Battery Modeling Library")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title)
        
        subtitle = QLabel("Complete Installation Setup")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setFont(QFont("Arial", 10))
        layout.addWidget(subtitle)
        
        layout.addSpacing(20)
        
        # Installation directory group
        install_group = QGroupBox("Installation Directory")
        install_group.setFont(QFont("Arial", 10, QFont.Bold))
        install_layout = QVBoxLayout(install_group)
        
        install_desc = QLabel("Where PyBattML application and Python environment will be installed:")
        install_desc.setFont(QFont("Arial", 9))
        install_layout.addWidget(install_desc)
        
        install_path_layout = QHBoxLayout()
        self.install_path_edit = QLineEdit(self.install_dir)
        install_path_layout.addWidget(self.install_path_edit)
        
        install_browse_btn = QPushButton("Browse...")
        install_browse_btn.clicked.connect(self.browse_install_dir)
        install_path_layout.addWidget(install_browse_btn)
        
        install_layout.addLayout(install_path_layout)
        layout.addWidget(install_group)
        
        # Project directory group  
        project_group = QGroupBox("Project Directory")
        project_group.setFont(QFont("Arial", 10, QFont.Bold))
        project_layout = QVBoxLayout(project_group)
        
        project_desc = QLabel("Where your data files and templates will be stored for easy access:")
        project_desc.setFont(QFont("Arial", 9))
        project_layout.addWidget(project_desc)
        
        project_path_layout = QHBoxLayout()
        self.project_path_edit = QLineEdit(self.project_dir)
        project_path_layout.addWidget(self.project_path_edit)
        
        project_browse_btn = QPushButton("Browse...")
        project_browse_btn.clicked.connect(self.browse_project_dir)
        project_path_layout.addWidget(project_browse_btn)
        
        project_layout.addLayout(project_path_layout)
        layout.addWidget(project_group)
        
        # Information
        info_group = QGroupBox("What will be installed:")
        info_layout = QVBoxLayout(info_group)
        
        info_text = QTextEdit()
        info_text.setMaximumHeight(100)
        info_text.setPlainText(
            "Complete Python environment with all dependencies\n"
            "GPU detection and CUDA support (if available)\n"
            "PyTorch with appropriate hardware optimization\n"
            "Sample data and hyperparameter templates\n"
            "Desktop launcher for easy access"
        )
        info_text.setReadOnly(True)
        info_layout.addWidget(info_text)
        layout.addWidget(info_group)
        
        layout.addStretch()
        
        # Install button
        self.install_btn = QPushButton("Start Installation")
        self.install_btn.setFont(QFont("Arial", 12, QFont.Bold))
        self.install_btn.setMinimumHeight(50)
        self.install_btn.clicked.connect(self.start_installation)
        layout.addWidget(self.install_btn)
    
    def browse_install_dir(self):
        """Browse for installation directory"""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Installation Directory", self.install_dir)
        if dir_path:
            self.install_dir = dir_path
            self.install_path_edit.setText(dir_path)
    
    def browse_project_dir(self):
        """Browse for project directory"""  
        dir_path = QFileDialog.getExistingDirectory(self, "Select Project Directory", self.project_dir)
        if dir_path:
            self.project_dir = dir_path
            self.project_path_edit.setText(dir_path)
    
    def start_installation(self):
        """Start the installation process"""
        install_dir = self.install_path_edit.text()
        project_dir = self.project_path_edit.text()
        
        if not install_dir or not project_dir:
            QMessageBox.warning(self, "Error", "Please select both installation and project directories.")
            return
        
        # Create directories if they don't exist
        try:
            Path(install_dir).mkdir(parents=True, exist_ok=True)
            Path(project_dir).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create directories: {e}")
            return
        
        # Show progress dialog
        self.progress_dialog = QProgressDialog("Initializing installation...", "Cancel", 0, 0, self)
        self.progress_dialog.setWindowTitle("Installing Vestim")
        self.progress_dialog.setModal(True)
        self.progress_dialog.setCancelButton(None)
        self.progress_dialog.setMinimumWidth(500)
        self.progress_dialog.show()
        
        # Start setup worker
        self.worker = SetupWorker(install_dir, project_dir)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()
        
        # Disable install button
        self.install_btn.setEnabled(False)
    
    def on_progress(self, message):
        """Handle progress updates"""
        self.progress_dialog.setLabelText(f"Installing...\n{message}")
        QApplication.processEvents()
    
    def on_finished(self, success, full_output):
        """Handle installation completion"""
        self.progress_dialog.close()
        self.install_btn.setEnabled(True)
        
        if success:
            QMessageBox.information(self, "Installation Complete", 
                f"PyBattML has been successfully installed!\n\n"
                f"Installation: {self.install_path_edit.text()}\n"
                f"Project Data: {self.project_path_edit.text()}\n\n"
                f"You can now launch PyBattML using the desktop shortcut or launcher scripts.")
            self.close()
        else:
            # Show the actual error details in the dialog
            error_details = full_output if full_output else "Unknown error occurred"
            
            # Create a detailed error message box
            error_msg = QMessageBox(self)
            error_msg.setIcon(QMessageBox.Critical)
            error_msg.setWindowTitle("Installation Failed")
            error_msg.setText("Installation failed. Please review the error details below.")
            error_msg.setDetailedText(error_details)
            error_msg.setStandardButtons(QMessageBox.Ok)
            
            # Also log to console so it stays visible
            print("\n" + "="*80)
            print("INSTALLATION FAILED")
            print("="*80)
            print(error_details)
            print("="*80)
            print("\nPlease copy the error above for debugging.")
            print("Press any key in the console window or close this dialog to exit...")
            
            error_msg.exec_()


def main():
    """Main entry point"""
    print("="*80)
    print("PyBattML Complete Installer")
    print("="*80)
    print("Starting installation GUI...")
    print("Console window will remain open to capture any error messages.")
    print("="*80 + "\n")
    
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("PyBattML Installer")
    app.setApplicationVersion("2.0.0")
    
    installer = PyBattMLInstaller()
    installer.show()
    
    exit_code = app.exec_()
    
    # Keep console open if there was an error
    if exit_code != 0:
        print("\nInstallation ended with errors. Press Enter to close...")
        input()
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
