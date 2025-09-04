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
    QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, 
    QWidget, QTreeWidget, QTreeWidgetItem, QProgressBar, QMessageBox, 
    QGroupBox, QTextEdit, QFrame, QFileDialog
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
        """Create the results table section exactly matching main testing GUI columns"""
        results_group = QGroupBox("Testing Results")
        results_layout = QVBoxLayout(results_group)
        
        # Create results table with exact main testing GUI columns
        self.results_table = QTreeWidget()
        self.results_table.setHeaderLabels([
            "Model", "Architecture", "Task ID", "File", "#Params", 
            "Best Train Loss (mV)", "Best Val Loss (mV)", "Epochs Trained", "RMSE (mV)", "Plot"
        ])
        self.results_table.setRootIsDecorated(False)
        self.results_table.setAlternatingRowColors(True)
        
        # Set column widths to match main testing GUI
        header = self.results_table.header()
        header.resizeSection(0, 80)   # Model
        header.resizeSection(1, 120)  # Architecture  
        header.resizeSection(2, 150)  # Task ID
        header.resizeSection(3, 100)  # File
        header.resizeSection(4, 80)   # #Params
        header.resizeSection(5, 130)  # Best Train Loss (mV)
        header.resizeSection(6, 130)  # Best Val Loss (mV)
        header.resizeSection(7, 110)  # Epochs Trained
        header.resizeSection(8, 100)  # RMSE (mV)
        header.resizeSection(9, 80)   # Plot
        
        results_layout.addWidget(self.results_table)
        self.main_layout.addWidget(results_group)
        
        # Add Open Job Folder button
        button_layout = QHBoxLayout()
        self.open_folder_button = QPushButton("📁 Open Job Folder")
        self.open_folder_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                font-size: 12px;
                padding: 10px 20px;
                border-radius: 5px;
                border: none;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.open_folder_button.clicked.connect(self.open_job_folder)
        
        button_layout.addWidget(self.open_folder_button)
        button_layout.addStretch()  # Push button to left
        self.main_layout.addLayout(button_layout)
        
        # Progress bar at bottom
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(0)  # Indeterminate
        self.progress_bar.hide()  # Initially hidden
        self.main_layout.addWidget(self.progress_bar)
        
        # Load any existing results from the testing manager
        self.load_testing_results()
    
    def open_job_folder(self):
        """Open the job folder in file explorer"""
        try:
            import subprocess
            import platform
            
            if os.path.exists(self.job_folder_path):
                if platform.system() == 'Windows':
                    subprocess.Popen(['explorer', self.job_folder_path])
                elif platform.system() == 'Darwin':  # macOS
                    subprocess.Popen(['open', self.job_folder_path])
                else:  # Linux
                    subprocess.Popen(['xdg-open', self.job_folder_path])
                    
                print(f"[DEBUG] Opened job folder: {self.job_folder_path}")
            else:
                QMessageBox.warning(self, "Folder Not Found", 
                                  f"Job folder not found: {self.job_folder_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not open job folder: {e}")
            print(f"[DEBUG] Error opening job folder: {e}")

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
        """Add result row matching main testing GUI format exactly"""
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
            target_column = result.get('target_column', 'voltage')  # Default to voltage
            model_file_path = result.get('model_file_path', '')
            
            # Get RMSE in proper units (main testing loop logic)
            rmse = result.get('RMSE', 'N/A')
            
            # Determine error unit and convert RMSE (like main testing loop)
            error_unit = ""
            if "voltage" in target_column.lower():
                error_unit = "mV"
                if isinstance(rmse, (int, float)):
                    rmse = rmse * 1000  # Convert to mV
            elif "soc" in target_column.lower():
                error_unit = "%SOC"
                if isinstance(rmse, (int, float)):
                    rmse = rmse * 100  # Convert to %SOC
            elif "temperature" in target_column.lower():
                error_unit = "°C"
            
            # Get model parameters count (try from task info first)
            task_info = result.get('task_info', None)
            model_params = self.get_model_parameters(model_file_path, task_info)
            
            # Get training metrics in proper units (using target column for unit conversion)
            training_metrics = self.get_training_metrics(model_file_path, target_column)
            if training_metrics:
                train_loss = training_metrics.get('Best Train Loss', 'N/A')
                val_loss = training_metrics.get('Best Val Loss', 'N/A') 
                epochs_trained = training_metrics.get('Epochs Trained', 'N/A')
                train_unit = training_metrics.get('Best Train Loss Unit', error_unit)
                val_unit = training_metrics.get('Best Val Loss Unit', error_unit)
            else:
                train_loss = val_loss = epochs_trained = 'N/A'
                train_unit = val_unit = error_unit
            
            # Get prediction data for plotting
            predictions_file = result.get('predictions_file', '')
            
            # Get actual test file name instead of "Test Data"
            test_data_file = result.get('test_data_file', '')
            if test_data_file and os.path.exists(test_data_file):
                file_name = os.path.basename(test_data_file)
            else:
                # Fallback to predictions file directory structure
                if predictions_file:
                    # Extract test file name from predictions file path pattern
                    pred_basename = os.path.basename(predictions_file)
                    if '_predictions.csv' in pred_basename:
                        file_name = pred_basename.replace('_predictions.csv', '.csv')
                    else:
                        file_name = "Test Data"
                else:
                    file_name = "Test Data"
            
            print(f"[DEBUG] Adding result row: {model_type}/{architecture}/{task}")
            print(f"[DEBUG] Training metrics: Train={train_loss} {train_unit}, Val={val_loss} {val_unit}, Epochs={epochs_trained}")
            print(f"[DEBUG] RMSE: {rmse} {error_unit}, Params: {model_params}")
            
            # Create tree widget item with exact main GUI columns
            item = QTreeWidgetItem(self.results_table)
            item.setText(0, model_type)                                # Model
            item.setText(1, architecture)                             # Architecture
            item.setText(2, task)                                     # Task ID
            item.setText(3, file_name)                                # File
            item.setText(4, str(model_params))                       # #Params
            
            # Training metrics with proper units
            train_display = f"{train_loss:.6f} {train_unit}" if isinstance(train_loss, (int, float)) else "N/A"
            val_display = f"{val_loss:.6f} {val_unit}" if isinstance(val_loss, (int, float)) else "N/A"
            epochs_display = str(epochs_trained) if epochs_trained != 'N/A' else "N/A"
            
            item.setText(5, train_display)                           # Best Train Loss (mV)
            item.setText(6, val_display)                             # Best Val Loss (mV)
            item.setText(7, epochs_display)                          # Epochs Trained
            
            # RMSE in proper units
            rmse_display = f"{rmse:.4f} {error_unit}" if isinstance(rmse, (int, float)) else "N/A"
            item.setText(8, rmse_display)                            # RMSE (mV)
            
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
                'target_display': target_column,  # Use target_column directly
                'error_unit': error_unit,
                'metrics': {
                    'RMSE': rmse
                }
            }
            
            plot_button.clicked.connect(lambda: self.show_model_plot(plot_data))
            self.results_table.setItemWidget(item, 9, plot_button)  # Column 9 for Plot
            
            print(f"[DEBUG] Successfully added result row to table")
            
            # Save results to CSV (like main testing loop)
            self.save_result_to_csv(result, rmse, error_unit, train_loss, val_loss, epochs_trained, model_params)
            
        except Exception as e:
            print(f"[DEBUG] Error adding result row: {e}")
            import traceback
            traceback.print_exc()
    
    def save_result_to_csv(self, result, rmse, error_unit, train_loss, val_loss, epochs_trained, model_params):
        """Save testing results to CSV file in standalone_test_results directory"""
        try:
            import pandas as pd
            import datetime
            
            # Get job folder from model file path
            model_file_path = result.get('model_file_path', '')
            if not model_file_path:
                return
                
            job_folder = os.path.dirname(os.path.dirname(os.path.dirname(model_file_path)))
            standalone_results_dir = os.path.join(job_folder, 'standalone_test_results')
            os.makedirs(standalone_results_dir, exist_ok=True)
            
            # Create timestamped CSV filename
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            architecture = result.get('architecture', 'unknown')
            task = result.get('task', 'unknown')
            csv_filename = f"standalone_test_results_{architecture}_{task}_{timestamp}.csv"
            csv_path = os.path.join(standalone_results_dir, csv_filename)
            
            # Prepare data row exactly like main testing GUI
            result_data = {
                'Timestamp': datetime.datetime.now().isoformat(),
                'Model': result.get('model_type', 'N/A'),
                'Architecture': result.get('architecture', 'N/A'),
                'Task_ID': result.get('task', 'N/A'),
                'File': 'Test Data',
                'Model_Parameters': model_params,
                f'Best_Train_Loss_{error_unit}': train_loss if isinstance(train_loss, (int, float)) else 'N/A',
                f'Best_Val_Loss_{error_unit}': val_loss if isinstance(val_loss, (int, float)) else 'N/A',
                'Epochs_Trained': epochs_trained if epochs_trained != 'N/A' else 'N/A',
                f'RMSE_{error_unit}': rmse if isinstance(rmse, (int, float)) else 'N/A',
                'Target_Column': result.get('target_column', 'N/A'),
                'Predictions_File': os.path.basename(result.get('predictions_file', ''))
            }
            
            # Create DataFrame and save
            df = pd.DataFrame([result_data])
            df.to_csv(csv_path, index=False)
            
            print(f"[DEBUG] Results saved to CSV: {csv_filename}")
            
        except Exception as e:
            print(f"[DEBUG] Error saving results to CSV: {e}")
            import traceback
            traceback.print_exc()
            
        except Exception as e:
            print(f"[DEBUG] Error adding result row: {e}")
            import traceback
            traceback.print_exc()
    
    def show_model_plot(self, plot_data):
        """Show plot for the model results by reading from saved predictions file - EXACTLY like main testing GUI"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
            import pandas as pd
            import numpy as np
            
            predictions_file = plot_data['predictions_file']
            model_info = plot_data['model_info']
            target_display = plot_data['target_display']
            error_unit = plot_data['error_unit']
            target_column = plot_data['target_column']
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
            
            # Find columns EXACTLY like main testing GUI
            true_col, pred_col, error_col, timestamp_col = None, None, None, None
            for col in df.columns:
                col_lower = col.lower()
                if 'true' in col_lower and target_column.lower() in col_lower: true_col = col
                elif 'predicted' in col_lower and target_column.lower() in col_lower: pred_col = col
                elif 'error' in col_lower: error_col = col
                elif 'timestamp' in col_lower or 'time' in col_lower: timestamp_col = col
            
            if not true_col or not pred_col:
                QMessageBox.critical(self, "Error", f"Required columns not found in predictions file.\nAvailable columns: {list(df.columns)}")
                return
            
            # Unit handling EXACTLY like main testing GUI
            unit_display_long, error_unit = target_column, ""
            if "voltage" in target_column.lower():
                unit_display_long, error_unit = "Voltage (V)", "mV"
            elif "soc" in target_column.lower():
                unit_display_long, error_unit = "SOC (% SOC)", "% SOC"
            elif "temperature" in target_column.lower():
                unit_display_long, error_unit = "Temperature (°C)", "°C"
            
            # Error calculation EXACTLY like main testing GUI
            errors_for_plot = df[error_col] if error_col else np.abs(df[true_col] - df[pred_col])
            if not error_col:
                if "voltage" in target_column.lower(): errors_for_plot *= 1000
                elif "soc" in target_column.lower() and np.max(np.abs(df[true_col])) <= 1.0: errors_for_plot *= 100

            rms_error = np.sqrt(np.mean(errors_for_plot**2))
            max_error = np.max(errors_for_plot)
            mean_error = np.mean(errors_for_plot)
            std_error = np.std(errors_for_plot)

            # X-axis handling EXACTLY like main testing GUI
            x_axis, x_label = (df.index, "Sample Index")

            # 1) Prefer explicit seconds if present (no guessing)
            for cand in ("Time (s)", "Time_s", "Seconds", "time_s"):
                if cand in df.columns:
                    x_axis, x_label = df[cand], "Time (seconds)"
                    break
            else:
                # 2) Your existing logic, but with Excel-serial handling
                if timestamp_col:
                    try:
                        ts = df[timestamp_col]
                        if pd.api.types.is_numeric_dtype(ts):
                            ts_num = pd.to_numeric(ts, errors='coerce')
                            if ts_num.notna().any() and 10000 < ts_num.max() < 1_000_000:
                                t = pd.to_datetime(ts_num, unit="D", origin="1899-12-30")
                                x_axis = (t - t.iloc[0]).dt.total_seconds()
                            else:
                                x_axis = ts_num - ts_num.iloc[0]
                        else:
                            t = pd.to_datetime(ts, errors='coerce', format='%Y-%m-%d %H:%M:%S.%f')
                            if t.notna().any():
                                x_axis = (t - t.iloc[0]).dt.total_seconds()
                        x_label = "Time (seconds)"
                    except:
                        pass

            # Final fallback if degenerate
            try:
                xa = np.asarray(x_axis)
                if np.nanmax(xa) - np.nanmin(xa) == 0:
                    x_axis, x_label = (df.index, "Sample Index")
            except:
                x_axis, x_label = (df.index, "Sample Index")

            # Create figure EXACTLY like main testing GUI
            fig = Figure(figsize=(14, 10), dpi=100)
            canvas = FigureCanvas(fig)
            toolbar = NavigationToolbar(canvas, self)
            plt.style.use('seaborn-v0_8-darkgrid')
            
            # Plot 1: Predictions vs True Values
            ax1 = fig.add_subplot(3, 1, 1)
            ax1.plot(x_axis, df[true_col], label='True Values', color='#2E86AB', linewidth=2, alpha=0.8)
            ax1.plot(x_axis, df[pred_col], label='Predictions', color='#A23B72', linewidth=2, linestyle='--', alpha=0.8)
            ax1.set_title(f'Model Predictions vs. True Values\n{os.path.basename(predictions_file)}', fontsize=14, fontweight='bold', pad=20)
            ax1.set_ylabel(unit_display_long, fontsize=12)
            ax1.legend(fontsize=11, loc='upper right')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Error over time
            ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
            ax2.plot(x_axis, errors_for_plot, label=f'Absolute Error ({error_unit})', color='#F18F01', linewidth=1.5, alpha=0.7)
            ax2.set_title('Prediction Error Over Time', fontsize=12, fontweight='bold')
            ax2.set_xlabel(x_label, fontsize=12)
            ax2.set_ylabel(f'Error ({error_unit})', fontsize=12)
            ax2.legend(fontsize=11)
            ax2.grid(True, alpha=0.3)
            
            # Add stats text box EXACTLY like main testing GUI
            stats_text = f'RMS Error: {rms_error:.3f} {error_unit}\nMax Error: {max_error:.3f} {error_unit}\nMean Error: {mean_error:.3f} {error_unit}\nStd Error: {std_error:.3f} {error_unit}'
            ax2.text(0.98, 0.02, stats_text, transform=ax2.transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8, edgecolor='navy'))
            
            # Plot 3: Error distribution histogram
            ax3 = fig.add_subplot(3, 1, 3)
            ax3.hist(errors_for_plot, bins=50, alpha=0.7, color='#F18F01', edgecolor='black', linewidth=0.5)
            ax3.axvline(x=mean_error, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.3f}')
            ax3.set_title('Error Distribution', fontsize=12, fontweight='bold')
            ax3.set_xlabel(f'Error ({error_unit})', fontsize=12)
            ax3.set_ylabel('Frequency', fontsize=12)
            ax3.legend(fontsize=10)
            ax3.grid(True, alpha=0.3)
            
            fig.tight_layout(pad=3.0)

            # Create plot window EXACTLY like main testing GUI
            plot_window = QMainWindow()
            plot_window.setWindowTitle(f"Test Results: {os.path.basename(predictions_file)}")
            plot_window.setGeometry(100, 100, 1400, 1000)
            
            central_widget = QWidget()
            plot_window.setCentralWidget(central_widget)
            layout = QVBoxLayout(central_widget)
            
            # Add toolbar and canvas
            layout.addWidget(toolbar)
            layout.addWidget(canvas)
            
            # Add buttons
            button_layout = QHBoxLayout()
            save_button = QPushButton("Save Plot")
            save_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px; font-weight: bold;")
            save_button.clicked.connect(lambda: self.save_plot(fig, predictions_file))
            
            close_button = QPushButton("Close")
            close_button.setStyleSheet("background-color: #f44336; color: white; padding: 8px; font-weight: bold;")
            close_button.clicked.connect(plot_window.close)
            
            button_layout.addWidget(save_button)
            button_layout.addWidget(close_button)
            button_layout.addStretch()
            layout.addLayout(button_layout)
            
            # Show maximized by default (user can restore)
            plot_window.showMaximized()
            plot_window.raise_()
            plot_window.activateWindow()
            
            # Store reference to prevent garbage collection
            self._plot_window = plot_window
            
            print(f"[DEBUG] Plot displayed successfully for {model_info}")
            
        except ImportError:
            QMessageBox.warning(self, "Plot Error", 
                              "Matplotlib is required for plotting. Please install it with: pip install matplotlib")
        except Exception as e:
            QMessageBox.warning(self, "Plot Error", f"Error creating plot: {str(e)}")
            print(f"[DEBUG] Plot error: {e}")
            import traceback
            traceback.print_exc()
    
    def get_model_parameters(self, model_file_path, task_info=None):
        """Get number of model parameters from task info or model file"""
        try:
            # First try to get from task info if available
            if task_info and isinstance(task_info, dict):
                # Check if parameters are stored in hyperparams or task info
                hyperparams = task_info.get('hyperparams', {})
                if 'model_parameters' in hyperparams:
                    total_params = hyperparams['model_parameters']
                elif 'MODEL_PARAMETERS' in hyperparams:
                    total_params = hyperparams['MODEL_PARAMETERS']
                elif 'total_params' in task_info:
                    total_params = task_info['total_params']
                else:
                    total_params = None
                
                if total_params:
                    # Format in K/M notation like main testing GUI
                    if total_params >= 1_000_000:
                        return f"{total_params / 1_000_000:.1f}M"
                    elif total_params >= 1_000:
                        return f"{total_params / 1_000:.1f}K"
                    else:
                        return str(total_params)
            
            # Fallback: Calculate from model file
            try:
                import tensorflow as tf
            except ImportError:
                print("[DEBUG] TensorFlow not available, cannot count parameters")
                return "N/A"
            
            if not os.path.exists(model_file_path):
                return "N/A"
            
            # Load model and count parameters
            model = tf.keras.models.load_model(model_file_path, compile=False)
            total_params = model.count_params()
            
            # Format in K/M notation like main testing GUI
            if total_params >= 1_000_000:
                return f"{total_params / 1_000_000:.1f}M"
            elif total_params >= 1_000:
                return f"{total_params / 1_000:.1f}K"
            else:
                return str(total_params)
                
        except Exception as e:
            print(f"[DEBUG] Error getting model parameters: {e}")
            return "N/A"
    
    def get_training_metrics(self, model_file_path, target_column="voltage"):
        """Get training metrics from training_progress.csv in proper units (mV, %SOC, etc.)"""
        try:
            import pandas as pd
            
            # Look for training_progress.csv in the logs subdirectory (task structure)
            model_dir = os.path.dirname(model_file_path)
            logs_dir = os.path.join(model_dir, 'logs')
            training_csv_path = os.path.join(logs_dir, 'training_progress.csv')
            
            # Fallback to same directory as model
            if not os.path.exists(training_csv_path):
                training_csv_path = os.path.join(model_dir, 'training_progress.csv')
            
            if not os.path.exists(training_csv_path):
                print(f"[DEBUG] No training_progress.csv found at {training_csv_path}")
                return None
            
            # Read the training progress CSV
            df = pd.read_csv(training_csv_path, comment='#')  # Handle comment lines
            print(f"[DEBUG] Loaded training progress CSV with columns: {list(df.columns)}")
            
            if len(df) > 0:
                training_metrics = {}
                
                # Determine unit conversion factor based on target column (like main testing loop)
                unit_multiplier = 1.0
                unit_suffix = ""
                
                if "voltage" in target_column.lower():
                    unit_multiplier = 1000.0  # Convert to mV
                    unit_suffix = "mV"
                elif "soc" in target_column.lower():
                    unit_multiplier = 100.0   # Convert to %SOC 
                    unit_suffix = "%SOC"
                elif "temperature" in target_column.lower():
                    unit_multiplier = 1.0     # Keep as °C
                    unit_suffix = "°C"
                
                # Get best training loss (minimum value from train_loss_norm column)
                if 'train_loss_norm' in df.columns:
                    best_train_loss = df['train_loss_norm'].min()
                    # Convert to proper units (assuming losses are in normalized/raw units)
                    training_metrics['Best Train Loss'] = best_train_loss * unit_multiplier
                    training_metrics['Best Train Loss Unit'] = unit_suffix
                
                # Get best validation loss (minimum value from val_loss_norm column)  
                if 'val_loss_norm' in df.columns:
                    best_val_loss = df['val_loss_norm'].min()
                    training_metrics['Best Val Loss'] = best_val_loss * unit_multiplier
                    training_metrics['Best Val Loss Unit'] = unit_suffix
                
                # Get epochs trained (maximum epoch number)
                if 'epoch' in df.columns:
                    epochs_trained = int(df['epoch'].max())
                    training_metrics['Epochs Trained'] = epochs_trained
                
                # Alternative column names if the above don't exist
                if 'best_val_loss_norm' in df.columns and 'Best Val Loss' not in training_metrics:
                    best_val_loss = df['best_val_loss_norm'].min()
                    training_metrics['Best Val Loss'] = best_val_loss * unit_multiplier
                    training_metrics['Best Val Loss Unit'] = unit_suffix
                
                print(f"[DEBUG] Training metrics in proper units ({unit_suffix}): {training_metrics}")
                return training_metrics if training_metrics else None
            
            return None
            
        except Exception as e:
            print(f"[DEBUG] Error reading training metrics: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_plot(self, fig, predictions_file):
        """Save plot to test_results directory structure exactly like main testing loop"""
        try:
            from PyQt5.QtWidgets import QFileDialog
            import datetime
            
            # Create test results directory structure like main testing loop
            job_folder = os.path.dirname(os.path.dirname(predictions_file))  # Go up from model dir
            test_results_dir = os.path.join(job_folder, 'test_results')
            
            # Create timestamped results directory
            timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            timestamped_dir = os.path.join(test_results_dir, f'new_test_results_{timestamp}')
            plots_dir = os.path.join(timestamped_dir, 'plots')
            
            # Create directories if they don't exist
            os.makedirs(plots_dir, exist_ok=True)
            
            # Generate default filename
            file_name = os.path.splitext(os.path.basename(predictions_file))[0]
            default_filename = f"{file_name}_test_plot.png"
            default_path = os.path.join(plots_dir, default_filename)
            
            # Show save dialog with default path in plots directory
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Save Plot", default_path, 
                "PNG files (*.png);;PDF files (*.pdf);;SVG files (*.svg);;All files (*.*)"
            )
            
            if save_path:
                # Ensure the directory exists for the chosen path
                save_dir = os.path.dirname(save_path)
                os.makedirs(save_dir, exist_ok=True)
                
                # Save with high quality
                fig.savefig(save_path, dpi=300, bbox_inches='tight', 
                          facecolor='white', edgecolor='none')
                
                rel_path = os.path.relpath(save_path, job_folder)
                QMessageBox.information(self, "Plot Saved", 
                                      f"Plot saved successfully!\n\nLocation: {rel_path}")
                print(f"[DEBUG] Plot saved to: {save_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save plot: {e}")
            print(f"[DEBUG] Plot save error: {e}")
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