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
    QGroupBox, QTextEdit, QFrame, QFileDialog, QTabWidget, QAction
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
        
        self.headers_updated = False  # Track if headers have been updated for target type
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
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #0b6337; margin-bottom: 5px;")
        self.main_layout.addWidget(title_label)
        
        # Job directory subtitle
        job_dir_name = os.path.basename(self.job_folder_path) if self.job_folder_path else "Unknown"
        job_dir_label = QLabel(f"Job: {job_dir_name}")
        job_dir_label.setAlignment(Qt.AlignCenter)
        job_dir_label.setStyleSheet("font-size: 11pt; color: #666; margin-bottom: 20px;")
        self.main_layout.addWidget(job_dir_label)

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
    
    def update_progress_log(self, message):
        """Handle progress messages from testing manager"""
        print(f"[TESTING PROGRESS] {message}")
        
        # Update progress bar visibility based on message content
        if "Starting test" in message or "Loading test data" in message:
            self.progress_bar.show()
        elif "STANDALONE TESTING COMPLETE" in message or "Testing failed" in message:
            self.progress_bar.hide()
    
    def show_completion_message(self):
        """Show completion message when testing is finished"""
        # Hide progress bar
        self.progress_bar.hide()
        
        # Just print completion message instead of showing popup
        print("[TESTING COMPLETE] All models have been tested successfully!")
        print("[TESTING COMPLETE] Results are displayed in the table above")
        print("[TESTING COMPLETE] Click 'Plot' buttons to view detailed visualizations")
        print("[TESTING COMPLETE] Results saved to job folder")

    def open_job_folder(self):
        """Open the job folder in file explorer"""
        try:
            import subprocess
            import platform
            
            if self.job_folder_path and os.path.exists(self.job_folder_path):
                if platform.system() == 'Windows':
                    subprocess.Popen(['explorer', self.job_folder_path])
                elif platform.system() == 'Darwin':  # macOS
                    subprocess.Popen(['open', self.job_folder_path])
                else:  # Linux
                    subprocess.Popen(['xdg-open', self.job_folder_path])
                    

            else:
                print(f"[ERROR] Job folder not found or not set: {self.job_folder_path}")
                
        except Exception as e:
            print(f"[ERROR] Could not open job folder: {e}")

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
    
    def update_table_headers_for_target(self, target_column):
        """Update table headers based on target column type"""
        # Determine unit for headers
        if "voltage" in target_column.lower():
            loss_unit = "mV"
            rmse_unit = "mV"
        elif "soc" in target_column.lower():
            loss_unit = "%SOC"
            rmse_unit = "%SOC"
        elif "temperature" in target_column.lower() or "temp" in target_column.lower():
            loss_unit = "°C"
            rmse_unit = "°C"
        else:
            loss_unit = "units"
            rmse_unit = "units"
            
        # Update headers with correct units
        self.results_table.setHeaderLabels([
            "Model", "Architecture", "Task ID", "File", "#Params", 
            f"Best Train Loss ({loss_unit})", f"Best Val Loss ({loss_unit})", "Epochs Trained", f"RMSE ({rmse_unit})", "Plot"
        ])
    
    def add_result_row(self, result):
        """Add result row matching main testing GUI format exactly"""
        
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
            
            # Update headers dynamically based on target type (first result only)
            if not self.headers_updated:
                self.update_table_headers_for_target(target_column)
                self.headers_updated = True
            
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
                    background-color: #663399;
                    color: white;
                    font-weight: bold;
                    padding: 5px 15px;
                    border-radius: 3px;
                    border: none;
                }
                QPushButton:hover {
                    background-color: #7d4db3;
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
            

            
            # Save results to CSV (like main testing loop)
            self.save_result_to_csv(result, rmse, error_unit, train_loss, val_loss, epochs_trained, model_params)
            
        except Exception as e:
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
            

            
        except Exception as e:

            import traceback
            traceback.print_exc()
            
        except Exception as e:
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
                print(f"[ERROR] Predictions file not found: {predictions_file}")
                return
            
            # Read the predictions file (EXACTLY like main testing GUI)
            try:
                df = pd.read_csv(predictions_file)


            except Exception as e:
                print(f"[ERROR] Error reading predictions file: {str(e)}")
                return
            
            # Determine target display name (same logic as prediction generation)
            if "voltage" in target_column.lower():
                target_display = "voltage"
            elif "soc" in target_column.lower():
                target_display = "soc"
            elif "temperature" in target_column.lower() or "temp" in target_column.lower():
                target_display = "temperature"
            else:
                target_display = target_column.lower()
            
            # Find columns using target_display instead of original target_column
            true_col, pred_col, error_col, timestamp_col = None, None, None, None
            for col in df.columns:
                col_lower = col.lower()
                if 'true' in col_lower and target_display in col_lower: true_col = col
                elif 'predicted' in col_lower and target_display in col_lower: pred_col = col
                elif 'error' in col_lower: error_col = col
                elif 'timestamp' in col_lower or 'time' in col_lower: timestamp_col = col
            
            if not true_col or not pred_col:
                print(f"[ERROR] Required columns not found in predictions file. Available columns: {list(df.columns)}")
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

            # Create plot window EXACTLY like main testing GUI using PlotWindow class
            plot_window = PlotWindow(self, fig=fig, canvas=canvas, toolbar=toolbar)
            plot_window.setWindowTitle(f"Test Results: {os.path.basename(predictions_file)}")

            # Create Save/Close buttons with improved styling and centering
            save_button = QPushButton("Save Plot")
            save_button.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50; 
                    color: white; 
                    padding: 12px 30px; 
                    font-weight: bold; 
                    font-size: 11pt;
                    border-radius: 5px;
                    min-width: 120px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
            """)
            save_button.clicked.connect(lambda: self.save_plot(fig, predictions_file))
            
            close_button = QPushButton("Close")
            close_button.setStyleSheet("""
                QPushButton {
                    background-color: #f44336; 
                    color: white; 
                    padding: 12px 30px; 
                    font-weight: bold; 
                    font-size: 11pt;
                    border-radius: 5px;
                    min-width: 120px;
                }
                QPushButton:hover {
                    background-color: #da190b;
                }
            """)
            close_button.clicked.connect(plot_window.close)
            
            plot_window.set_buttons(save_button, close_button)

            # Provide data for "Export Data in Current View"
            import numpy as np
            xa = np.asarray(x_axis)
            def _provider():
                # returns (x, df, true_col, pred_col, err_series, x_label)
                return xa, df, true_col, pred_col, errors_for_plot if hasattr(errors_for_plot, 'values') else pd.Series(errors_for_plot), x_label
            plot_window._current_view_provider = _provider
            
            # Show maximized by default (user can restore)
            plot_window.showMaximized()
            plot_window.raise_()
            plot_window.activateWindow()
            
            # Store reference to prevent garbage collection
            self._plot_window = plot_window
            

            
        except ImportError:
            print("[ERROR] Matplotlib is required for plotting. Please install it with: pip install matplotlib")
        except Exception as e:
            print(f"[ERROR] Error creating plot: {str(e)}")
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

                return None
            
            # Read the training progress CSV
            df = pd.read_csv(training_csv_path, comment='#')  # Handle comment lines

            
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
                

                return training_metrics if training_metrics else None
            
            return None
            
        except Exception as e:

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
                print(f"[PLOT SAVED] Plot saved successfully to: {rel_path}")
                
        except Exception as e:
            print(f"[ERROR] Could not save plot: {e}")
            import traceback
            traceback.print_exc()

class PlotWindow(QMainWindow):
    def __init__(self, parent=None, fig=None, canvas=None, toolbar=None):
        super().__init__(parent)
        self.setWindowTitle("Test Results")
        self.setMinimumSize(900, 600)
        self.setWindowFlags(self.windowFlags() |
                            Qt.WindowMaximizeButtonHint |
                            Qt.WindowMinimizeButtonHint)
        # central widget with layout
        central = QWidget(self)
        self.setCentralWidget(central)
        self.vbox = QVBoxLayout(central)

        # optional tabs to keep things neat if you add diagnostics later
        self.tabs = QTabWidget(self)
        self.view_tab = QWidget()
        self.view_layout = QVBoxLayout(self.view_tab)
        self.view_layout.addWidget(toolbar)
        self.view_layout.addWidget(canvas)
        self.tabs.addTab(self.view_tab, "Overview")
        self.vbox.addWidget(self.tabs)

        # bottom buttons (keep your Save/Close buttons)
        self.btn_row = QHBoxLayout()
        self.vbox.addLayout(self.btn_row)

        self.fig = fig
        self.canvas = canvas

        self._build_menu()

    def _build_menu(self):
        menubar = self.menuBar()

        # File
        file_menu = menubar.addMenu("&File")
        act_save = QAction("Save Figure…", self)
        act_save.setShortcut("Ctrl+S")
        act_save.triggered.connect(self._save_full_figure)
        file_menu.addAction(act_save)

        act_save_view = QAction("Save Current View…", self)
        act_save_view.setShortcut("Ctrl+Shift+S")
        act_save_view.triggered.connect(self._save_current_view)
        file_menu.addAction(act_save_view)

        act_export_csv = QAction("Export Data in Current View (CSV)…", self)
        act_export_csv.triggered.connect(self._export_current_view_csv)
        file_menu.addAction(act_export_csv)

        file_menu.addSeparator()
        act_copy = QAction("Copy Figure to Clipboard", self)
        act_copy.setShortcut("Ctrl+C")
        act_copy.triggered.connect(self._copy_to_clipboard)
        file_menu.addAction(act_copy)

        file_menu.addSeparator()
        act_close = QAction("Close", self)
        act_close.setShortcut("Ctrl+W")
        act_close.triggered.connect(self.close)
        file_menu.addAction(act_close)

        # View
        view_menu = menubar.addMenu("&View")
        act_full = QAction("Toggle Full Screen", self)
        act_full.setShortcut("F11")
        act_full.triggered.connect(self._toggle_fullscreen)
        view_menu.addAction(act_full)

        act_reset = QAction("Reset View (All Axes)", self)
        act_reset.setShortcut("Ctrl+R")
        act_reset.triggered.connect(self._reset_limits_all_axes)
        view_menu.addAction(act_reset)

        act_grid = QAction("Toggle Grid", self)
        act_grid.setShortcut("G")
        act_grid.triggered.connect(self._toggle_grid_all_axes)
        view_menu.addAction(act_grid)

        act_legend = QAction("Toggle Legend", self)
        act_legend.setShortcut("L")
        act_legend.triggered.connect(self._toggle_legend_all_axes)
        view_menu.addAction(act_legend)

        # Help
        help_menu = menubar.addMenu("&Help")
        act_help = QAction("Tips", self)
        act_help.triggered.connect(lambda: print(
            "[PLOT TIPS] Zoom: toolbar magnifier | Pan: toolbar hand | "
            "Save current zoom via File → Save Current View | "
            "Export visible data as CSV via File → Export Data in Current View"
        ))
        help_menu.addAction(act_help)

    # ==== Helpers you'll call from your main widget ====
    def set_buttons(self, save_button, close_button):
        # Center the buttons and make them wider
        self.btn_row.addStretch()  # Left stretch
        self.btn_row.addWidget(save_button)
        self.btn_row.addWidget(close_button)
        self.btn_row.addStretch()  # Right stretch

    # ==== Actions ====
    def _get_primary_axes(self):
        # assumes your first subplot is ax1 (lines), 2nd ax2 (errors), 3rd ax3 (hist)
        # we'll use ax1's x-limits as the "view window"
        if self.fig is None:
            return None
        axes = self.fig.get_axes()
        return axes[0] if axes else None

    def _save_full_figure(self):
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(self, "Save Figure", "", "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if path:
            try:
                self.fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
                print(f"[PLOT SAVED] Full figure saved to: {path}")
            except Exception as e:
                print(f"[ERROR] Could not save figure: {e}")

    def _save_current_view(self):
        from PyQt5.QtWidgets import QFileDialog
        ax = self._get_primary_axes()
        if ax is None:
            return
        xlim = ax.get_xlim()
        # temporarily enforce xlim on all time-aligned axes
        axes = self.fig.get_axes()
        prev_lims = [a.get_xlim() for a in axes]
        try:
            for a in axes[:2]:  # ax1 and ax2 share x; leave hist alone
                a.set_xlim(xlim)
            self.canvas.draw_idle()
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Current View", "", "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)"
            )
            if path:
                self.fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
                print(f"[PLOT SAVED] Current view saved to: {path}")
        except Exception as e:
            print(f"[ERROR] Could not save current view: {e}")
        finally:
            for a, lim in zip(axes, prev_lims):
                a.set_xlim(lim)
            self.canvas.draw_idle()

    def _export_current_view_csv(self):
        from PyQt5.QtWidgets import QFileDialog
        # This will be filled by the parent with a callback that returns (x, df, true_col, pred_col, err_series, x_label)
        if not hasattr(self, "_current_view_provider"):
            print("[WARNING] Export provider not set")
            return
        data = self._current_view_provider()
        if data is None:
            print("[WARNING] Could not get current view data")
            return
        x, df, true_col, pred_col, err_series, x_label = data
        ax = self._get_primary_axes()
        if ax is None:
            return
        xmin, xmax = ax.get_xlim()
        try:
            import numpy as np
            mask = (x >= xmin) & (x <= xmax)
            sub = df.loc[mask, [true_col, pred_col]].copy()
            sub[x_label] = x[mask]
            if err_series is not None:
                sub["error_for_plot"] = err_series[mask].values
            path, _ = QFileDialog.getSaveFileName(self, "Export Visible Data", "", "CSV (*.csv)")
            if path:
                sub[[x_label, true_col, pred_col] + (["error_for_plot"] if "error_for_plot" in sub.columns else [])] \
                    .to_csv(path, index=False)
                print(f"[DATA EXPORTED] Visible data exported to: {path}")
        except Exception as e:
            print(f"[ERROR] Could not export data: {e}")

    def _copy_to_clipboard(self):
        try:
            # render to png bytes and put on clipboard
            import io
            from PyQt5.QtGui import QImage, QClipboard, QPixmap
            from PyQt5.QtWidgets import QApplication
            buf = io.BytesIO()
            self.fig.savefig(buf, format="png", dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
            buf.seek(0)
            qimg = QImage.fromData(buf.getvalue(), "PNG")
            QApplication.clipboard().setPixmap(QPixmap.fromImage(qimg), QClipboard.Clipboard)
            print("[PLOT COPIED] Figure copied to clipboard")
        except Exception as e:
            print(f"[ERROR] Could not copy to clipboard: {e}")

    def _toggle_fullscreen(self):
        if self.windowState() & Qt.WindowFullScreen:
            self.setWindowState(self.windowState() & ~Qt.WindowFullScreen)
        else:
            self.setWindowState(self.windowState() | Qt.WindowFullScreen)

    def _reset_limits_all_axes(self):
        for a in self.fig.get_axes():
            a.autoscale(enable=True, axis='both', tight=False)
        self.canvas.draw_idle()

    def _toggle_grid_all_axes(self):
        for a in self.fig.get_axes():
            grid_lines = a.get_xgridlines()
            is_grid_on = len(grid_lines) > 0 and grid_lines[0].get_visible()
            a.grid(not is_grid_on, alpha=0.3)
        self.canvas.draw_idle()

    def _toggle_legend_all_axes(self):
        for a in self.fig.get_axes():
            leg = a.get_legend()
            if leg is None:
                # try to create one if there are labeled artists
                a.legend(fontsize=10)
            else:
                leg.set_visible(not leg.get_visible())
        self.canvas.draw_idle()

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