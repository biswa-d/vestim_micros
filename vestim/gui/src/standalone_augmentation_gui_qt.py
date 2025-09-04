"""
Simplified Augmentation GUI for Standalone Testing
This GUI provides a streamlined interface for applying augmentation steps
when they cannot be automatically applied from metadata.
"""

import sys
import os
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QWidget, QListWidget, QListWidgetItem, QMessageBox, QTextEdit,
    QGroupBox, QFormLayout, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox
)
from PyQt5.QtCore import Qt, pyqtSignal

class StandaloneAugmentationGUI(QMainWindow):
    augmentation_completed = pyqtSignal(pd.DataFrame)
    
    def __init__(self, test_df, filter_configs=None):
        super().__init__()
        self.test_df = test_df.copy()
        self.original_df = test_df.copy()
        self.filter_configs = filter_configs or []
        
        self.setWindowTitle("Apply Required Augmentation - Standalone Testing")
        self.setGeometry(100, 100, 1000, 700)
        
        self.initUI()
        self.populate_required_steps()
    
    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Title
        title_label = QLabel("Apply Required Data Augmentation")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #0b6337; margin-bottom: 15px;")
        main_layout.addWidget(title_label)
        
        # Info label
        info_label = QLabel("The following augmentation steps are required based on the original training job:")
        info_label.setStyleSheet("font-size: 11px; color: #666; margin-bottom: 10px;")
        main_layout.addWidget(info_label)
        
        # Steps list
        steps_group = QGroupBox("Required Steps")
        steps_layout = QVBoxLayout(steps_group)
        
        self.steps_list = QListWidget()
        self.steps_list.setMaximumHeight(200)
        steps_layout.addWidget(self.steps_list)
        
        main_layout.addWidget(steps_group)
        
        # Filter configuration
        self.filter_config_group = QGroupBox("Filter Configuration")
        self.filter_config_layout = QFormLayout(self.filter_config_group)
        main_layout.addWidget(self.filter_config_group)
        
        # Data preview
        preview_group = QGroupBox("Data Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.data_info_label = QLabel()
        self.update_data_info()
        preview_layout.addWidget(self.data_info_label)
        
        main_layout.addWidget(preview_group)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.apply_button = QPushButton("Apply Filters")
        self.apply_button.clicked.connect(self.apply_filters)
        self.apply_button.setStyleSheet("""
            QPushButton {
                background-color: #0b6337;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #0d7940;
            }
        """)
        button_layout.addWidget(self.apply_button)
        
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_data)
        button_layout.addWidget(self.reset_button)
        
        button_layout.addStretch()
        
        self.continue_button = QPushButton("Continue with Testing")
        self.continue_button.clicked.connect(self.continue_testing)
        self.continue_button.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        button_layout.addWidget(self.continue_button)
        
        main_layout.addLayout(button_layout)
    
    def populate_required_steps(self):
        """Populate the list of required augmentation steps"""
        if not self.filter_configs:
            item = QListWidgetItem("No specific filter configurations found")
            self.steps_list.addItem(item)
            self.setup_basic_filter_interface()
            return
        
        for i, config in enumerate(self.filter_configs):
            if config.get("type") == "filter":
                step_text = f"Butterworth Filter: {config.get('column', 'Unknown')} → {config.get('output_column_name', 'Unknown')}"
                item = QListWidgetItem(step_text)
                self.steps_list.addItem(item)
        
        # Setup filter configuration interface
        self.setup_filter_interface()
    
    def setup_basic_filter_interface(self):
        """Setup a basic filter interface when no specific config is found"""
        self.column_combo = QComboBox()
        self.column_combo.addItems(list(self.test_df.columns))
        self.filter_config_layout.addRow("Source Column:", self.column_combo)
        
        self.output_name_edit = QLineEdit()
        self.filter_config_layout.addRow("Output Column Name:", self.output_name_edit)
        
        self.corner_freq_spin = QDoubleSpinBox()
        self.corner_freq_spin.setRange(0.001, 100.0)
        self.corner_freq_spin.setValue(0.003)
        self.corner_freq_spin.setDecimals(4)
        self.filter_config_layout.addRow("Corner Frequency (Hz):", self.corner_freq_spin)
        
        self.sampling_rate_spin = QDoubleSpinBox()
        self.sampling_rate_spin.setRange(0.1, 1000.0)
        self.sampling_rate_spin.setValue(10.0)
        self.filter_config_layout.addRow("Sampling Rate (Hz):", self.sampling_rate_spin)
        
        self.filter_order_spin = QSpinBox()
        self.filter_order_spin.setRange(1, 10)
        self.filter_order_spin.setValue(2)
        self.filter_config_layout.addRow("Filter Order:", self.filter_order_spin)
    
    def setup_filter_interface(self):
        """Setup filter interface based on known configurations"""
        if not self.filter_configs:
            return
        
        config = self.filter_configs[0]  # Use first filter config as template
        
        self.column_combo = QComboBox()
        self.column_combo.addItems(list(self.test_df.columns))
        if config.get('column') in self.test_df.columns:
            self.column_combo.setCurrentText(config.get('column'))
        self.filter_config_layout.addRow("Source Column:", self.column_combo)
        
        self.output_name_edit = QLineEdit()
        self.output_name_edit.setText(config.get('output_column_name', ''))
        self.filter_config_layout.addRow("Output Column Name:", self.output_name_edit)
        
        self.corner_freq_spin = QDoubleSpinBox()
        self.corner_freq_spin.setRange(0.001, 100.0)
        self.corner_freq_spin.setValue(config.get('corner_frequency', 0.003))
        self.corner_freq_spin.setDecimals(4)
        self.filter_config_layout.addRow("Corner Frequency (Hz):", self.corner_freq_spin)
        
        self.sampling_rate_spin = QDoubleSpinBox()
        self.sampling_rate_spin.setRange(0.1, 1000.0)
        self.sampling_rate_spin.setValue(config.get('sampling_rate', 10.0))
        self.filter_config_layout.addRow("Sampling Rate (Hz):", self.sampling_rate_spin)
        
        self.filter_order_spin = QSpinBox()
        self.filter_order_spin.setRange(1, 10)
        self.filter_order_spin.setValue(config.get('filter_order', 2))
        self.filter_config_layout.addRow("Filter Order:", self.filter_order_spin)
    
    def apply_filters(self):
        """Apply the Butterworth filter to the data"""
        try:
            from vestim.services.data_processor.src.data_augment_service import DataAugmentService
            
            source_column = self.column_combo.currentText()
            output_column = self.output_name_edit.text().strip()
            corner_freq = self.corner_freq_spin.value()
            sampling_rate = self.sampling_rate_spin.value()
            filter_order = self.filter_order_spin.value()
            
            if not output_column:
                QMessageBox.warning(self, "Invalid Input", "Please specify an output column name.")
                return
            
            if source_column not in self.test_df.columns:
                QMessageBox.warning(self, "Invalid Input", f"Column '{source_column}' not found in data.")
                return
            
            # Apply the filter
            data_service = DataAugmentService()
            self.test_df = data_service.apply_butterworth_filter(
                self.test_df,
                column_name=source_column,
                corner_frequency=corner_freq,
                sampling_rate=sampling_rate,
                filter_order=filter_order,
                output_column_name=output_column
            )
            
            self.update_data_info()
            QMessageBox.information(self, "Filter Applied", f"Successfully created filtered column '{output_column}'")
            
        except Exception as e:
            QMessageBox.critical(self, "Filter Error", f"Error applying filter: {str(e)}")
    
    def reset_data(self):
        """Reset data to original state"""
        self.test_df = self.original_df.copy()
        self.update_data_info()
        QMessageBox.information(self, "Data Reset", "Data has been reset to original state.")
    
    def update_data_info(self):
        """Update data information display"""
        info_text = f"Data Shape: {self.test_df.shape[0]} rows × {self.test_df.shape[1]} columns\n"
        info_text += f"Columns: {', '.join(list(self.test_df.columns))}"
        self.data_info_label.setText(info_text)
    
    def continue_testing(self):
        """Continue with testing using the current data"""
        # Emit the augmented data
        self.augmentation_completed.emit(self.test_df)
        self.close()

def main():
    """Test the standalone augmentation GUI"""
    app = QApplication(sys.argv)
    
    # Create sample data
    import numpy as np
    data = {
        'Time': np.arange(100),
        'Power': np.random.randn(100),
        'Current': np.random.randn(100),
        'Voltage': 3.7 + 0.1 * np.random.randn(100)
    }
    test_df = pd.DataFrame(data)
    
    # Create sample filter config
    filter_configs = [{
        "type": "filter",
        "column": "Power",
        "output_column_name": "Power_filtered",
        "corner_frequency": 0.003,
        "sampling_rate": 10.0,
        "filter_order": 2
    }]
    
    gui = StandaloneAugmentationGUI(test_df, filter_configs)
    gui.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()