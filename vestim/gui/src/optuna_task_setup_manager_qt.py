# ---------------------------------------------------------------------------------
# Author: Biswanath Dehury
# Date: 2025-07-15
# Version: 1.0.0
# Description: Optuna Task Setup Manager for creating training tasks from best configs
# ---------------------------------------------------------------------------------

import os
import json
import logging
from typing import Dict, Any, List
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, 
    QMessageBox, QProgressBar, QTextEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QFont

from vestim.gateway.src.job_manager_qt import JobManager


class OptunaPrepareTaskManager(QWidget):
    """Manager for preparing training tasks from Optuna best configurations"""
    
    def __init__(self, base_params: Dict[str, Any], best_configs: List[Dict[str, Any]], parent=None):
        super().__init__(parent)
        self.base_params = base_params
        self.best_configs = best_configs
        self.job_manager = JobManager()
        self.logger = logging.getLogger(__name__)
        
        self.setup_window()
        self.build_gui()
        self.setup_tasks()
    
    def setup_window(self):
        """Setup window properties"""
        self.setWindowTitle("Optuna Task Preparation")
        self.setGeometry(100, 100, 800, 600)
        
        # Load icon if available
        try:
            resources_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources')
            icon_path = os.path.join(resources_path, 'icon.ico')
            if os.path.exists(icon_path):
                self.setWindowIcon(QIcon(icon_path))
        except Exception as e:
            self.logger.warning(f"Could not load icon: {e}")
    
    def build_gui(self):
        """Build the main GUI layout"""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title_label = QLabel("Optuna Task Preparation")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 18pt; font-weight: bold; margin-bottom: 10px;")
        main_layout.addWidget(title_label)
        
        # Subtitle
        subtitle_label = QLabel(f"Preparing {len(self.best_configs)} training tasks from best configurations")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("font-size: 12pt; color: gray; margin-bottom: 20px;")
        main_layout.addWidget(subtitle_label)
        
        # Configuration preview table
        self.create_config_preview_table(main_layout)
        
        # Progress section
        self.create_progress_section(main_layout)
        
        # Buttons
        self.create_buttons(main_layout)
        
        self.setLayout(main_layout)
    
    def create_config_preview_table(self, layout):
        """Create table showing the configurations that will be used"""
        # Table header
        table_label = QLabel("Best Configurations Preview:")
        table_label.setStyleSheet("font-size: 12pt; font-weight: bold; margin-top: 10px;")
        layout.addWidget(table_label)
        
        # Table
        self.config_table = QTableWidget()
        
        # Get parameter names from first config
        if self.best_configs:
            param_names = list(self.best_configs[0]['params'].keys())
            self.config_table.setColumnCount(len(param_names) + 2)  # +2 for Rank and Objective
            
            headers = ['Rank', 'Objective Value'] + param_names
            self.config_table.setHorizontalHeaderLabels(headers)
            
            # Populate table
            self.config_table.setRowCount(len(self.best_configs))
            for i, config in enumerate(self.best_configs):
                # Rank
                self.config_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
                
                # Objective value
                obj_val = config.get('objective_value', 'N/A')
                self.config_table.setItem(i, 1, QTableWidgetItem(f"{obj_val:.6f}" if isinstance(obj_val, (int, float)) else str(obj_val)))
                
                # Parameters
                for j, param_name in enumerate(param_names):
                    param_value = config['params'].get(param_name, 'N/A')
                    self.config_table.setItem(i, j + 2, QTableWidgetItem(str(param_value)))
            
            # Resize columns to content
            self.config_table.resizeColumnsToContents()
            header = self.config_table.horizontalHeader()
            header.setSectionResizeMode(QHeaderView.Stretch)
        
        self.config_table.setFixedHeight(200)
        layout.addWidget(self.config_table)
    
    def create_progress_section(self, layout):
        """Create progress display section"""
        progress_label = QLabel("Task Creation Progress:")
        progress_label.setStyleSheet("font-size: 12pt; font-weight: bold; margin-top: 20px;")
        layout.addWidget(progress_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Status text
        self.status_text = QTextEdit()
        self.status_text.setFixedHeight(150)
        self.status_text.setReadOnly(True)
        layout.addWidget(self.status_text)
    
    def create_buttons(self, layout):
        """Create control buttons"""
        button_layout = QHBoxLayout()
        
        # Back button
        back_button = QPushButton("← Back to Results")
        back_button.setFixedHeight(35)
        back_button.clicked.connect(self.go_back)
        button_layout.addWidget(back_button)
        
        button_layout.addStretch()
        
        # Create tasks button
        self.create_tasks_button = QPushButton("Create Training Tasks")
        self.create_tasks_button.setFixedHeight(35)
        self.create_tasks_button.setStyleSheet("background-color: #2E86AB; color: white; font-weight: bold;")
        self.create_tasks_button.clicked.connect(self.create_training_tasks)
        button_layout.addWidget(self.create_tasks_button)
        
        # Proceed to monitoring button (initially disabled)
        self.proceed_button = QPushButton("Proceed to Task Monitoring")
        self.proceed_button.setFixedHeight(35)
        self.proceed_button.setStyleSheet("background-color: #0b6337; color: white; font-weight: bold;")
        self.proceed_button.setEnabled(False)
        self.proceed_button.clicked.connect(self.proceed_to_monitoring)
        button_layout.addWidget(self.proceed_button)
        
        layout.addLayout(button_layout)
    
    def setup_tasks(self):
        """Initial setup and validation"""
        self.add_status_message(f"Ready to create {len(self.best_configs)} training tasks")
        self.add_status_message(f"Base model: {self.base_params.get('MODEL_TYPE', 'Unknown')}")
        self.add_status_message(f"Features: {self.base_params.get('FEATURE_COLUMNS', [])}")
        self.add_status_message(f"Target: {self.base_params.get('TARGET_COLUMN', 'Unknown')}")
    
    def add_status_message(self, message):
        """Add a status message to the text area"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_text.append(f"[{timestamp}] {message}")
    
    def create_training_tasks(self):
        """Create training tasks from the best configurations"""
        try:
            self.create_tasks_button.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setMaximum(len(self.best_configs))
            self.progress_bar.setValue(0)
            
            self.add_status_message("Starting task creation...")
            
            task_configs = []
            
            for i, best_config in enumerate(self.best_configs):
                self.add_status_message(f"Creating task {i+1}/{len(self.best_configs)}...")
                
                # Create a complete configuration by merging base params with the specific config
                task_config = self.create_task_config(best_config, i+1)
                task_configs.append(task_config)
                
                self.progress_bar.setValue(i + 1)
                QApplication.processEvents()  # Update GUI
            
            # Save all task configurations
            self.save_task_configurations(task_configs)
            
            self.add_status_message(f"✓ Successfully created {len(task_configs)} training tasks!")
            self.add_status_message("Tasks are ready for execution.")
            
            # Enable proceed button
            self.proceed_button.setEnabled(True)
            self.create_tasks_button.setText("Tasks Created ✓")
            self.create_tasks_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
            
        except Exception as e:
            self.logger.error(f"Error creating tasks: {e}")
            self.add_status_message(f"✗ Error creating tasks: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to create tasks: {str(e)}")
            self.create_tasks_button.setEnabled(True)
    
    def create_task_config(self, best_config: Dict[str, Any], task_number: int) -> Dict[str, Any]:
        """Create a complete task configuration from base params and best config"""
        
        # Start with base parameters
        task_config = self.base_params.copy()
        
        # Override with the optimized parameters from Optuna
        optimized_params = best_config['params']
        
        # Update the task config with specific optimized values
        for param_name, param_value in optimized_params.items():
            task_config[param_name] = param_value
        
        # Add metadata
        task_config['TASK_ID'] = f"optuna_task_{task_number}"
        task_config['OPTUNA_TRIAL_NUMBER'] = best_config.get('trial_number', task_number)
        task_config['OPTUNA_OBJECTIVE_VALUE'] = best_config.get('objective_value', 'Unknown')
        task_config['OPTUNA_RANK'] = task_number
        task_config['TASK_SOURCE'] = 'optuna_optimization'
        
        # Ensure required parameters are present
        if 'REPETITIONS' not in task_config:
            task_config['REPETITIONS'] = 1
        
        return task_config
    
    def save_task_configurations(self, task_configs: List[Dict[str, Any]]):
        """Save task configurations to the job folder"""
        try:
            # Get job folder
            job_folder = self.job_manager.get_job_folder()
            if not job_folder or not os.path.exists(job_folder):
                raise ValueError("Could not find job folder")
            
            # Create timestamp for unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save individual task configs
            for i, config in enumerate(task_configs):
                config_filename = os.path.join(job_folder, f"optuna_task_config_{i+1}_{timestamp}.json")
                with open(config_filename, 'w') as f:
                    json.dump(config, f, indent=2)
            
            # Save summary file
            summary_data = {
                'total_tasks': len(task_configs),
                'creation_timestamp': timestamp,
                'source': 'optuna_optimization',
                'base_params': self.base_params,
                'task_files': [f"optuna_task_config_{i+1}_{timestamp}.json" for i in range(len(task_configs))]
            }
            
            summary_filename = os.path.join(job_folder, f"optuna_tasks_summary_{timestamp}.json")
            with open(summary_filename, 'w') as f:
                json.dump(summary_data, f, indent=2)
            
            self.add_status_message(f"Task configurations saved to: {job_folder}")
            self.add_status_message(f"Summary file: optuna_tasks_summary_{timestamp}.json")
            
        except Exception as e:
            self.logger.error(f"Error saving task configurations: {e}")
            raise
    
    def proceed_to_monitoring(self):
        """Proceed to task monitoring interface"""
        try:
            # Import and show the training setup GUI for monitoring
            from vestim.gui.src.training_setup_gui_qt import VEstimTrainSetupGUI
            
            # Create a list of configurations for the training setup GUI
            config_list = []
            for i, best_config in enumerate(self.best_configs):
                task_config = self.create_task_config(best_config, i+1)
                config_list.append(task_config)
            
            self.close()
            self.training_setup_gui = VEstimTrainSetupGUI(config_list)
            self.training_setup_gui.show()
            
        except Exception as e:
            self.logger.error(f"Error proceeding to monitoring: {e}")
            QMessageBox.critical(self, "Error", f"Failed to proceed to monitoring: {str(e)}")
    
    def go_back(self):
        """Go back to Optuna results"""
        self.close()
        # The parent window should still be available


if __name__ == "__main__":
    # Test the manager
    app = QApplication([])
    
    # Sample data for testing
    base_params = {
        'MODEL_TYPE': 'LSTM',
        'FEATURE_COLUMNS': ['SOC', 'Current', 'Temp'],
        'TARGET_COLUMN': 'Voltage',
        'TRAINING_METHOD': 'Sequence-to-Sequence',
        'DEVICE_SELECTION': 'CPU'
    }
    
    best_configs = [
        {
            'params': {'LAYERS': 2, 'HIDDEN_UNITS': 64, 'INITIAL_LR': 0.001, 'MAX_EPOCHS': 100},
            'objective_value': 0.0234,
            'trial_number': 15
        },
        {
            'params': {'LAYERS': 1, 'HIDDEN_UNITS': 32, 'INITIAL_LR': 0.0005, 'MAX_EPOCHS': 150},
            'objective_value': 0.0267,
            'trial_number': 23
        }
    ]
    
    manager = OptunaPrepareTaskManager(base_params, best_configs)
    manager.show()
    
    app.exec_()
