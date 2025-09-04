# ---------------------------------------------------------------------------------
# Author: GitHub Copilot
# Date: 2024-12-19
# Version: 2.0.0
# Description: CLEAN Standalone Testing GUI for VEstim Tool
# 
# This GUI provides a simplified interface for viewing standalone testing results
# EXACTLY like the main testing GUI - just title, hyperparams, and results table.
# No input fields - all processing done in test selection GUI.
#
# Copyright (c) 2024 Biswanath Dehury, Dr. Phil Kollmeyer's Battery Lab at McMaster University
# ---------------------------------------------------------------------------------

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, 
    QWidget, QTreeWidget, QTreeWidgetItem, QProgressBar, QMessageBox, 
    QGroupBox, QTextEdit, QFrame
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import os, sys, json
import logging

class VEstimStandaloneTestingGUI(QMainWindow):
    def __init__(self, job_folder_path):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Initialize variables
        self.job_folder_path = job_folder_path
        self.results_list = []
        
        # Add param_labels needed for hyperparameter display
        self.param_labels = {
            'MODEL_TYPE': 'Model Architecture',
            'NUM_LEARNABLE_PARAMS': 'Total Parameters',
            'LAYERS': 'Hidden Layers',
            'HIDDEN_UNITS': 'Hidden Units',
            'FNN_HIDDEN_LAYERS': 'FNN Hidden Layers',
            'HIDDEN_LAYER_SIZES': 'Layer Sizes',
            'FNN_DROPOUT_PROB': 'FNN Dropout',
            'DROPOUT_PROB': 'Dropout Probability',
            'TRAINING_METHOD': 'Training Method',
            'LOOKBACK': 'Lookback Steps',
            'BATCH_TRAINING': 'Batch Training',
            'BATCH_SIZE': 'Batch Size',
            'MAX_EPOCHS': 'Max Epochs',
            'SCHEDULER_TYPE': 'LR Scheduler',
            'INITIAL_LR': 'Initial Learning Rate',
            'LR_DROP_PERIOD': 'LR Drop Period',
            'LR_PERIOD': 'LR Step Period',
            'LR_DROP_FACTOR': 'LR Drop Factor',
            'LR_PARAM': 'LR Parameter',
            'PLATEAU_PATIENCE': 'Plateau Patience',
            'PLATEAU_FACTOR': 'Plateau Factor',
            'COSINE_T0': 'Cosine T0',
            'COSINE_T_MULT': 'Cosine T Mult',
            'COSINE_ETA_MIN': 'Cosine Eta Min',
            'VALID_FREQUENCY': 'Validation Frequency',
            'VALID_PATIENCE': 'Validation Patience',
            'REPETITIONS': 'Repetitions',
            'CURRENT_REPETITION': 'Current Repetition',
            'DEVICE_SELECTION': 'Device Selection',
            'CURRENT_DEVICE': 'Current Device',
            'USE_MIXED_PRECISION': 'Mixed Precision',
            'MAX_TRAINING_TIME_SECONDS': 'Max Training Time',
            'FEATURE_COLUMNS': 'Feature Columns',
            'TARGET_COLUMN': 'Target Column',
            'INFERENCE_FILTER_TYPE': 'Inference Filter',
            'INFERENCE_FILTER_WINDOW_SIZE': 'Filter Window Size',
            'INFERENCE_FILTER_ALPHA': 'Filter Alpha',
            'INFERENCE_FILTER_POLYORDER': 'Filter Poly Order',
            'PIN_MEMORY': 'Pin Memory'
        }
        
        self.initUI()

    def initUI(self):
        # Set window title with job folder name
        job_folder_name = os.path.basename(self.job_folder_path) if self.job_folder_path else "Unknown"
        self.setWindowTitle(f"VEstim Tool - Standalone Testing Results: {job_folder_name}")
        self.setGeometry(100, 100, 1200, 800)  # Same size as main testing GUI

        # Create central widget and main layout
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Simple styling like main testing GUI
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
            }
        """)

        # Title
        title_label = QLabel("Standalone Testing Results")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #0b6337; margin-bottom: 20px;")
        self.main_layout.addWidget(title_label)

        # Create hyperparameters display section
        self.create_hyperparams_section()
        
        # Create results table section
        self.create_results_section()

    def create_hyperparams_section(self):
        """Create hyperparameters display section EXACTLY like main testing GUI"""
        # Create hyperparam frame EXACTLY like main testing GUI - no layout initially
        self.hyperparam_frame = QFrame()
        self.hyperparam_frame.setObjectName("hyperparamFrame")
        self.hyperparam_frame.setStyleSheet("""
            #hyperparamFrame {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                background-color: #ffffff;
            }
        """)
        self.main_layout.addWidget(self.hyperparam_frame)
        
        # Load and display hyperparameters
        self.load_job_hyperparameters()

    def display_hyperparameters(self, params):
        """Display hyperparameters using the EXACT same method as main testing GUI"""
        from vestim.gui.src.adaptive_gui_utils import display_hyperparameters
        display_hyperparameters(self, params)
    
    def create_results_section(self):
        """Create the results table section exactly like main testing GUI"""
        results_group = QGroupBox("Testing Results")
        results_layout = QVBoxLayout(results_group)
        
        # Create results table
        self.results_table = QTreeWidget()
        self.results_table.setHeaderLabels(["Model", "Task", "File", "MAE", "MSE", "RMSE", "MAPE", "R²", "Actions"])
        self.results_table.setRootIsDecorated(False)
        self.results_table.setAlternatingRowColors(True)
        
        # Set column widths
        header = self.results_table.header()
        header.resizeSection(0, 100)  # Model
        header.resizeSection(1, 100)  # Task  
        header.resizeSection(2, 150)  # File
        header.resizeSection(3, 80)   # MAE
        header.resizeSection(4, 80)   # MSE
        header.resizeSection(5, 80)   # RMSE
        header.resizeSection(6, 80)   # MAPE
        header.resizeSection(7, 80)   # R²
        header.resizeSection(8, 100)  # Actions
        
        results_layout.addWidget(self.results_table)
        self.main_layout.addWidget(results_group)
        
        # Progress bar at bottom
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(0)  # Indeterminate
        self.progress_bar.hide()  # Initially hidden
        self.main_layout.addWidget(self.progress_bar)
        
        # Load any existing results from the testing manager
        self.load_testing_results()
    
    def load_job_hyperparameters(self):
        """Load and display hyperparameters EXACTLY like main testing GUI"""
        try:
            hyperparams_file = os.path.join(self.job_folder_path, 'hyperparams.json')
            if os.path.exists(hyperparams_file):
                with open(hyperparams_file, 'r') as f:
                    hyperparams = json.load(f)
                
                # Use EXACT same hyperparameters display method as main testing GUI
                self.display_hyperparameters(hyperparams)
                
            else:
                # Create a simple label if no hyperparams file
                layout = QVBoxLayout()
                label = QLabel("No hyperparams.json file found in the job folder.")
                layout.addWidget(label)
                self.hyperparam_frame.setLayout(layout)
        except Exception as e:
            # Create a simple label for error
            layout = QVBoxLayout()
            label = QLabel(f"Error loading hyperparameters: {str(e)}")
            layout.addWidget(label)
            self.hyperparam_frame.setLayout(layout)
    
    def load_testing_results(self):
        """Load testing results from the backend testing manager results"""
        # Start with empty table - results will be added via signals from testing manager
        # Add a placeholder to show the GUI is ready
        placeholder_item = QTreeWidgetItem(self.results_table)
        placeholder_item.setText(0, "Loading...")
        placeholder_item.setText(1, "Waiting for test results...")
        placeholder_item.setText(2, "...")
        placeholder_item.setText(3, "...")
        placeholder_item.setText(4, "...")
        placeholder_item.setText(5, "...")
        placeholder_item.setText(6, "...")
        placeholder_item.setText(7, "...")
        
        # Create disabled plot button for placeholder
        plot_button = QPushButton("Plot")
        plot_button.setEnabled(False)
        plot_button.setStyleSheet("""
            QPushButton {
                background-color: #d3d3d3;
                color: #a9a9a9;
                font-weight: bold;
                padding: 5px 15px;
                border-radius: 3px;
                border: none;
            }
        """)
        self.results_table.setItemWidget(placeholder_item, 8, plot_button)
    
    def add_result_row(self, result):
        """Add result row from standalone testing manager results"""
        print(f"[DEBUG] Received result data: {list(result.keys()) if isinstance(result, dict) else type(result)}")
        
        # Clear placeholder items first
        if self.results_table.topLevelItemCount() > 0:
            item = self.results_table.topLevelItem(0)
            if item and item.text(0) == "Loading...":
                self.results_table.clear()
        
        try:
            # Extract data from the testing manager result structure
            model_type = result.get('model_type', 'N/A')
            architecture = result.get('architecture', 'N/A')
            task = result.get('task', 'N/A')
            target_column = result.get('target_column', 'N/A')
            
            # Get error metrics
            mae = result.get('MAE', 'N/A')
            mse = result.get('MSE', 'N/A')  
            rmse = result.get('RMSE', 'N/A')
            mape = result.get('MAPE', 'N/A')
            r2 = result.get('R²', 'N/A')
            
            # Get prediction data for plotting
            predictions = result.get('predictions', [])
            actual_values = result.get('actual_values', [])
            predictions_file = result.get('predictions_file', '')
            target_display = result.get('target_display', target_column)
            error_unit = result.get('error_unit', 'units')
            
            # Create a simple file identifier
            file_name = "Test Data"
            
            print(f"[DEBUG] Adding result row: {model_type}/{architecture}/{task}")
            print(f"[DEBUG] Predictions file: {predictions_file}")
            
            # Create tree widget item
            item = QTreeWidgetItem(self.results_table)
            item.setText(0, f"{model_type}")
            item.setText(1, f"{architecture}_{task}")
            item.setText(2, file_name)
            item.setText(3, f"{mae:.4f}" if isinstance(mae, (int, float)) else str(mae))
            item.setText(4, f"{mse:.4f}" if isinstance(mse, (int, float)) else str(mse))
            item.setText(5, f"{rmse:.4f}" if isinstance(rmse, (int, float)) else str(rmse))
            item.setText(6, f"{mape:.2f}%" if isinstance(mape, (int, float)) else str(mape))
            item.setText(7, f"{r2:.4f}" if isinstance(r2, (int, float)) else str(r2))
            
            # Create plot button with result data
            plot_button = QPushButton("Plot")
            plot_button.setStyleSheet("""
                QPushButton {
                    background-color: #0b6337;
                    color: white;
                    font-weight: bold;
                    padding: 5px 15px;
                    border-radius: 3px;
                    border: none;
                }
                QPushButton:hover {
                    background-color: #0d7940;
                }
            """)
            
            # Store result data for plotting
            plot_data = {
                'predictions_file': predictions_file,
                'model_info': f"{model_type} - {architecture}/{task}",
                'target_column': target_column,
                'target_display': target_display,
                'error_unit': error_unit,
                'metrics': {
                    'MAE': mae,
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAPE': mape,
                    'R²': r2
                }
            }
            
            plot_button.clicked.connect(lambda: self.show_model_plot(plot_data))
            self.results_table.setItemWidget(item, 8, plot_button)
            
            print(f"[DEBUG] Successfully added result row to table")
            
        except Exception as e:
            print(f"[DEBUG] Error adding result row: {e}")
            import traceback
            traceback.print_exc()
    
    def show_model_plot(self, plot_data):
        """Show plot for the model results by reading from saved predictions file"""
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            import numpy as np
            
            predictions_file = plot_data['predictions_file']
            model_info = plot_data['model_info']
            target_display = plot_data['target_display']
            error_unit = plot_data['error_unit']
            metrics = plot_data['metrics']
            
            if not predictions_file or not os.path.exists(predictions_file):
                QMessageBox.warning(self, "Plot Error", f"Predictions file not found: {predictions_file}")
                return
            
            # Read the predictions file (EXACTLY like main testing GUI)
            try:
                df = pd.read_csv(predictions_file)
                print(f"[DEBUG] Loaded predictions file: {predictions_file}")
                print(f"[DEBUG] Columns: {list(df.columns)}")
            except Exception as e:
                QMessageBox.warning(self, "Plot Error", f"Error reading predictions file: {str(e)}")
                return
            
            # Extract predictions and actual values from the CSV file
            pred_col = None
            true_col = None
            error_col = None
            
            # Find the relevant columns (flexible column name matching)
            for col in df.columns:
                if 'predicted' in col.lower() or f'predicted_{target_display.lower()}' in col.lower():
                    pred_col = col
                elif 'true' in col.lower() or f'true_{target_display.lower()}' in col.lower():
                    true_col = col
                elif 'error' in col.lower():
                    error_col = col
            
            if pred_col is None:
                QMessageBox.warning(self, "Plot Error", f"Could not find prediction column in file. Columns: {list(df.columns)}")
                return
            
            predictions = df[pred_col].values
            actual_values = df[true_col].values if true_col and true_col in df.columns else None
            
            print(f"[DEBUG] Found prediction column: {pred_col}")
            print(f"[DEBUG] Found true column: {true_col}")
            print(f"[DEBUG] Predictions shape: {predictions.shape}")
            print(f"[DEBUG] Actual values shape: {actual_values.shape if actual_values is not None else None}")
            
            # Create the plot (EXACTLY like main testing GUI)
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot 1: Predictions vs Time
            x_axis = range(len(predictions))
            axes[0].plot(x_axis, predictions, label='Predictions', color='blue', linewidth=1.5, alpha=0.8)
            
            if actual_values is not None and len(actual_values) > 0:
                axes[0].plot(x_axis, actual_values, label='Actual', color='red', linewidth=1.5, alpha=0.8)
                axes[0].legend()
            
            axes[0].set_title(f'{model_info} - {target_display} Predictions vs Time')
            axes[0].set_xlabel('Sample Index')
            axes[0].set_ylabel(f'{target_display}')
            axes[0].grid(True, alpha=0.3)
            
            # Plot 2: Actual vs Predicted scatter (if actual values available)
            if actual_values is not None and len(actual_values) > 0:
                axes[1].scatter(actual_values, predictions, alpha=0.6, color='green', s=1)
                
                # Add perfect prediction line
                min_val = min(min(actual_values), min(predictions))
                max_val = max(max(actual_values), max(predictions))
                axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
                
                axes[1].set_xlabel(f'Actual {target_display}')
                axes[1].set_ylabel(f'Predicted {target_display}')
                axes[1].set_title('Actual vs Predicted')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
                # Add metrics text (EXACTLY like main testing GUI)
                metrics_text = (
                    f"MAE: {metrics['MAE']:.4f}\n"
                    f"RMSE: {metrics['RMSE']:.4f}\n"
                    f"R²: {metrics['R²']:.4f}\n"
                    f"MAPE: {metrics['MAPE']:.2f}%"
                )
                axes[1].text(0.05, 0.95, metrics_text, transform=axes[1].transAxes, 
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            else:
                # Just show predictions if no actual values
                axes[1].plot(x_axis, predictions, color='blue', linewidth=1.5)
                axes[1].set_title(f'{model_info} - Predictions Only')
                axes[1].set_xlabel('Sample Index')
                axes[1].set_ylabel(f'Predicted {target_display}')
                axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            print(f"[DEBUG] Plot displayed successfully for {model_info}")
            
        except ImportError:
            QMessageBox.warning(self, "Plot Error", 
                              "Matplotlib is required for plotting. Please install it with: pip install matplotlib")
        except Exception as e:
            QMessageBox.warning(self, "Plot Error", f"Error creating plot: {str(e)}")
            print(f"[DEBUG] Plot error: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main function to run the standalone testing GUI"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("VEstim Standalone Testing")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("McMaster University Battery Lab")
    
    # For testing purposes, create with a dummy path
    gui = VEstimStandaloneTestingGUI("test_job_folder")
    gui.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()