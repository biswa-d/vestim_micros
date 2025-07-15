# ---------------------------------------------------------------------------------
# Author: Biswanath Dehury
# Date: 2025-07-14
# Version: 1.0.0
# Description: Optuna-based hyperparameter optimization GUI for VEstim
# ---------------------------------------------------------------------------------

import os
import json
import logging
import time
from typing import Dict, Any, List, Tuple
import threading

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, 
    QLineEdit, QMessageBox, QGroupBox, QComboBox, QSpinBox, QDoubleSpinBox,
    QProgressBar, QTextEdit, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QFrame, QScrollArea, QFormLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QIcon, QFont

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from vestim.gateway.src.job_manager_qt import JobManager
from vestim.gui.src.training_setup_gui_qt import VEstimTrainSetupGUI


class OptunaOptimizationThread(QThread):
    """Thread for running Optuna optimization in the background"""
    
    # Signals for communicating with the main GUI thread
    progress_updated = pyqtSignal(int, int)  # current_trial, total_trials
    trial_completed = pyqtSignal(dict)  # trial_info
    optimization_completed = pyqtSignal(list)  # best_configs
    error_occurred = pyqtSignal(str)  # error_message
    log_message = pyqtSignal(str)  # log_message
    
    def __init__(self, base_params, optimization_config, parent=None):
        super().__init__(parent)
        self.base_params = base_params
        self.optimization_config = optimization_config
        self.study = None
        self.should_stop = False
        
        # Extract parameter ranges in boundary format [min,max]
        self.param_ranges = {k: v for k, v in base_params.items() 
                           if isinstance(v, str) and '[' in v and ']' in v}
        
    def run(self):
        """Run the Optuna optimization"""
        try:
            self.log_message.emit("Starting Bayesian optimization with Optuna...")
            
            # Create sampler based on selection
            sampler_name = self.optimization_config.get('sampler', 'TPE (Recommended)')
            if 'Random' in sampler_name:
                sampler = optuna.samplers.RandomSampler()
            elif 'CMA-ES' in sampler_name:
                sampler = optuna.samplers.CmaEsSampler()
            else:  # TPE
                sampler = optuna.samplers.TPESampler()
            
            # Create pruner based on selection
            pruner_name = self.optimization_config.get('pruner', 'Median Pruner')
            if 'No Pruning' in pruner_name:
                pruner = optuna.pruners.NopPruner()
            elif 'Hyperband' in pruner_name:
                pruner = optuna.pruners.HyperbandPruner()
            else:  # Median Pruner
                pruner = optuna.pruners.MedianPruner()
            
            # Create study
            study_name = f"vestim_optimization_{int(time.time())}"
            self.study = optuna.create_study(
                direction='minimize',
                study_name=study_name,
                sampler=sampler,
                pruner=pruner
            )
            
            # Log optimization settings
            self.log_message.emit(f"Using {sampler_name} sampler and {pruner_name}")
            
            # Run optimization
            n_trials = self.optimization_config['n_trials']
            self.log_message.emit(f"Running {n_trials} optimization trials...")
            
            for trial_num in range(n_trials):
                if self.should_stop:
                    self.log_message.emit("Optimization stopped by user")
                    break
                    
                try:
                    # Run single trial
                    trial = self.study.ask()
                    params = self._suggest_params(trial)
                    
                    # Evaluate parameters (simulate training for now)
                    objective_value = self._evaluate_params(params, trial_num)
                    
                    # Tell study the result
                    self.study.tell(trial, objective_value)
                    
                    # Emit progress and trial info
                    self.progress_updated.emit(trial_num + 1, n_trials)
                    self.trial_completed.emit({
                        'trial_number': trial_num + 1,
                        'params': params,
                        'value': objective_value,
                        'state': 'COMPLETE'
                    })
                    
                    self.log_message.emit(f"Trial {trial_num + 1}: objective = {objective_value:.6f}")
                    
                except Exception as e:
                    self.log_message.emit(f"Trial {trial_num + 1} failed: {str(e)}")
                    self.trial_completed.emit({
                        'trial_number': trial_num + 1,
                        'params': {},
                        'value': float('inf'),
                        'state': 'FAIL'
                    })
            
            # Get best configurations
            if not self.should_stop:
                best_configs = self._get_best_configurations()
                self.log_message.emit(f"Optimization completed! Found {len(best_configs)} best configurations")
                self.optimization_completed.emit(best_configs)
                
        except Exception as e:
            self.error_occurred.emit(f"Optimization failed: {str(e)}")
    
    def _suggest_params(self, trial):
        """Suggest parameters for a trial based on boundary format from GUI"""
        params = self.base_params.copy()
        
        # Define which parameters should be treated as integers vs floats
        integer_params = {
            "LAYERS", "HIDDEN_UNITS", "GRU_LAYERS", "GRU_HIDDEN_UNITS", 
            "MAX_EPOCHS", "VALID_PATIENCE", "VALID_FREQUENCY", "LOOKBACK",
            "BATCH_SIZE", "LR_PERIOD", "PLATEAU_PATIENCE", "REPETITIONS"
        }
        
        # Process parameters that are in boundary format [min,max]
        for param_name, param_value in self.base_params.items():
            if isinstance(param_value, str) and param_value.startswith('[') and param_value.endswith(']'):
                try:
                    # Parse the boundary format [min,max]
                    inner = param_value[1:-1].strip()
                    parts = [part.strip() for part in inner.split(',')]
                    
                    if len(parts) == 2:
                        min_val = float(parts[0])
                        max_val = float(parts[1])
                        
                        if param_name in integer_params:
                            # Suggest integer value
                            suggested_value = trial.suggest_int(param_name.lower(), int(min_val), int(max_val))
                        else:
                            # Suggest float value
                            if param_name in ["INITIAL_LR", "LR_PARAM", "PLATEAU_FACTOR", "FNN_DROPOUT_PROB"]:
                                # Use log scale for learning rates and probabilities
                                suggested_value = trial.suggest_float(param_name.lower(), min_val, max_val, log=True)
                            else:
                                suggested_value = trial.suggest_float(param_name.lower(), min_val, max_val)
                        
                        params[param_name] = str(suggested_value)
                        
                except (ValueError, IndexError) as e:
                    self.log_message.emit(f"Could not parse boundary format for {param_name}: {param_value}")
                    # Keep original value if parsing fails
                    continue
        
        return params
        
        if 'learning_rate_range' in config:
            min_val, max_val = config['learning_rate_range']
            params['INITIAL_LR'] = str(trial.suggest_float('learning_rate', min_val, max_val, log=True))
        
        if 'lookback_range' in config and params.get('TRAINING_METHOD') == 'Sequence-to-Sequence':
            min_val, max_val = config['lookback_range']
            params['LOOKBACK'] = str(trial.suggest_int('lookback', min_val, max_val))
        
        if 'dropout_range' in config and self.base_params.get('MODEL_TYPE') == 'FNN':
            min_val, max_val = config['dropout_range']
            params['FNN_DROPOUT_PROB'] = str(trial.suggest_float('dropout', min_val, max_val))
        
        return params
    
    def _evaluate_params(self, params, trial_num):
        """Evaluate parameters by running real training and returning validation loss."""
        self.log_message.emit(f"--- Evaluating Trial {trial_num + 1} ---")
        import torch
        import numpy as np
        from vestim.services.model_training.src.LSTM_model_service_test import LSTMModelService
        from vestim.services.model_training.src.training_task_service import TrainingTaskService
        from vestim.services.model_training.src.data_loader_service import DataLoaderService
        import warnings
        warnings.filterwarnings("ignore")

        # --- 1. Prepare data loaders (CSV-based, not HDF5) ---
        lookback = int(float(params.get('LOOKBACK', 400)))
        batch_size = int(float(params.get('BATCH_SIZE', 100)))
        feature_columns = params.get('FEATURE_COLUMNS', ['SOC', 'Current', 'Temp'])
        target_column = params.get('TARGET_COLUMN', 'Voltage')
        train_folder = params.get('TRAIN_FOLDER')
        valid_folder = params.get('VALID_FOLDER')
        if not train_folder or not valid_folder:
            # Try to get from job manager
            job_manager = JobManager()
            train_folder = job_manager.get_train_folder()
            valid_folder = job_manager.get_val_folder()

        data_loader_service = DataLoaderService()
        training_method = params.get('TRAINING_METHOD', 'Sequence-to-Sequence')
        model_type = params.get('MODEL_TYPE', 'LSTM')
        try:
            job_manager = JobManager()
            job_folder_path = job_manager.get_job_folder()

            train_loader, val_loader = data_loader_service.create_data_loaders_from_separate_folders(
                job_folder_path=job_folder_path,
                training_method=training_method,
                feature_cols=feature_columns,
                target_col=target_column,
                batch_size=batch_size,
                num_workers=0,
                lookback=lookback,
                concatenate_raw_data=False,
                seed=trial_num + 42,
                model_type=model_type,
                create_test_loader=False
            )
        except Exception as e:
            self.log_message.emit(f"Data loading failed: {e}")
            return float('inf')

        # --- Check for empty loaders ---
        def is_loader_empty(loader):
            try:
                return len(loader) == 0
            except Exception:
                return True

        if is_loader_empty(train_loader):
            self.log_message.emit("No training data available for this trial. Skipping.")
            return float('inf')
        if is_loader_empty(val_loader):
            self.log_message.emit("No validation data available for this trial. Skipping.")
            return float('inf')

        # --- 2. Build model ---
        model_type = params.get('MODEL_TYPE', 'LSTM')
        input_size = len(feature_columns)
        hidden_units = int(float(params.get('HIDDEN_UNITS', 10)))
        layers = int(float(params.get('LAYERS', 1)))
        device_str = params.get('DEVICE_SELECTION', 'cpu')
        device = torch.device(device_str if 'cuda' in device_str and torch.cuda.is_available() else 'cpu')
        model_service = LSTMModelService()
        model_params = {
            'INPUT_SIZE': input_size,
            'HIDDEN_UNITS': hidden_units,
            'LAYERS': layers
        }
        model = model_service.build_lstm_model(model_params, device)
        model.to(device)

        # --- 3. Training setup ---
        max_epochs = int(float(params.get('MAX_EPOCHS', 10)))  # Use a reasonable number of epochs for Optuna to evaluate trials
        initial_lr = float(params.get('INITIAL_LR', 0.001))
        optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
        training_service = TrainingTaskService()
        stop_requested = False
        
        # Initialize hidden states for RNN models
        h_s_initial, h_c_initial = None, None
        if model_type in ['LSTM', 'GRU']:
            num_layers = int(float(params.get('LAYERS', 1)))
            hidden_units = int(float(params.get('HIDDEN_UNITS', 10)))
            # Use the actual batch size from the data loader
            actual_batch_size = train_loader.batch_size if train_loader else int(float(params.get('BATCH_SIZE', 100)))
            h_s_initial = torch.zeros(num_layers, actual_batch_size, hidden_units).to(device)
            if model_type == 'LSTM':
                h_c_initial = torch.zeros(num_layers, actual_batch_size, hidden_units).to(device)

        best_val_loss = float('inf')

        # --- 4. Training loop (short, for speed) ---
        for epoch in range(1, max_epochs+1):
            self.log_message.emit(f"  Trial {trial_num + 1}, Epoch {epoch}/{max_epochs}...")
            # Correctly unpack the tuple returned by train_epoch
            _, train_loss_norm, _, _ = training_service.train_epoch(
                model, model_type, train_loader, optimizer, h_s_initial, h_c_initial, epoch, device, stop_requested, task={
                    'hyperparams': params,
                    'log_frequency': 100,
                    'log_callback': self.log_message.emit
                }
            )
            # Use the dedicated validate_epoch function for validation
            val_loss, _, _ = training_service.validate_epoch(
                model, model_type, val_loader, h_s_initial, h_c_initial, epoch, device, stop_requested, task={
                    'hyperparams': params,
                    'log_frequency': 100,
                    'log_callback': self.log_message.emit
                }
            )
            self.log_message.emit(f"  Trial {trial_num + 1}, Epoch {epoch}: Train Loss = {train_loss_norm:.6f}, Val Loss = {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
        
        self.log_message.emit(f"--- Finished Trial {trial_num + 1} with loss: {best_val_loss:.6f} ---")
        # --- 5. Return best validation loss as objective ---
        return best_val_loss
    
    def _get_best_configurations(self):
        """Get the best N configurations from the study"""
        n_configs = self.optimization_config.get('n_best_configs', 5)
        
        # Get best trials
        best_trials = sorted(self.study.trials, key=lambda t: t.value if t.value is not None else float('inf'))
        best_trials = [t for t in best_trials if t.state == optuna.trial.TrialState.COMPLETE]
        best_trials = best_trials[:n_configs]
        
        best_configs = []
        for trial in best_trials:
            config = self.base_params.copy()
            
            # Update with optimized parameters
            for key, value in trial.params.items():
                if key == 'hidden_units':
                    if config.get('MODEL_TYPE') == 'LSTM':
                        config['HIDDEN_UNITS'] = str(value)
                    elif config.get('MODEL_TYPE') == 'GRU':
                        config['GRU_HIDDEN_UNITS'] = str(value)
                elif key == 'layers':
                    if config.get('MODEL_TYPE') == 'LSTM':
                        config['LAYERS'] = str(value)
                    elif config.get('MODEL_TYPE') == 'GRU':
                        config['GRU_LAYERS'] = str(value)
                elif key == 'batch_size':
                    config['BATCH_SIZE'] = str(value)
                elif key == 'learning_rate':
                    config['INITIAL_LR'] = str(value)
                elif key == 'lookback':
                    config['LOOKBACK'] = str(value)
                elif key == 'dropout':
                    config['FNN_DROPOUT_PROB'] = str(value)
            
            best_configs.append({
                'params': config,
                'objective_value': trial.value,
                'trial_number': trial.number
            })
        
        return best_configs
    
    def stop_optimization(self):
        """Stop the optimization"""
        self.should_stop = True


class VEstimOptunaOptimizationGUI(QWidget):
    """GUI for Optuna-based hyperparameter optimization"""
    
    def __init__(self, base_params):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.base_params = base_params
        self.optimization_thread = None
        self.best_configs = []
        
        # Extract parameter ranges from base_params
        self.param_ranges = {k: v for k, v in base_params.items() 
                           if isinstance(v, str) and '[' in v and ']' in v}
        
        # Check if Optuna is available
        if not OPTUNA_AVAILABLE:
            QMessageBox.critical(
                self, 
                "Optuna Not Available", 
                "Optuna is not installed. Please install it using:\npip install optuna"
            )
            self.close()
            return
        
        self.setup_window()
        self.build_gui()
        
    def setup_window(self):
        """Setup the main window"""
        self.setWindowTitle("VEstim - Optuna Hyperparameter Optimization")
        self.setGeometry(100, 100, 1200, 800)
        
        # Load the application icon
        resources_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources')
        icon_path = os.path.join(resources_path, 'icon.ico')
        
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
    
    def build_gui(self):
        """Build the main GUI layout"""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title_label = QLabel("Bayesian Hyperparameter Optimization")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 18pt; font-weight: bold; margin-bottom: 15px; color: #2E86AB;")
        main_layout.addWidget(title_label)
        
        # Subtitle
        subtitle_label = QLabel("Using Optuna for intelligent hyperparameter search")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("font-size: 12pt; color: gray; margin-bottom: 20px;")
        main_layout.addWidget(subtitle_label)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Configuration tab
        config_tab = self.create_configuration_tab()
        self.tab_widget.addTab(config_tab, "Configuration")
        
        # Optimization tab
        optimization_tab = self.create_optimization_tab()
        self.tab_widget.addTab(optimization_tab, "Optimization")
        
        # Results tab
        results_tab = self.create_results_tab()
        self.tab_widget.addTab(results_tab, "Results")
        
        main_layout.addWidget(self.tab_widget)
        
        # Bottom buttons
        button_layout = QHBoxLayout()
        
        back_button = QPushButton("← Back to Hyperparameters")
        back_button.setFixedHeight(35)
        back_button.clicked.connect(self.go_back)
        
        self.start_button = QPushButton("Start Optimization")
        self.start_button.setFixedHeight(35)
        self.start_button.setStyleSheet("background-color: #2E86AB; color: white; font-weight: bold;")
        self.start_button.clicked.connect(self.start_optimization)
        
        self.stop_button = QPushButton("Stop Optimization")
        self.stop_button.setFixedHeight(35)
        self.stop_button.setStyleSheet("background-color: #D32F2F; color: white; font-weight: bold;")
        self.stop_button.clicked.connect(self.stop_optimization)
        self.stop_button.setEnabled(False)
        
        self.proceed_button = QPushButton("Prepare Training Tasks →")
        self.proceed_button.setFixedHeight(35)
        self.proceed_button.setStyleSheet("background-color: #0b6337; color: white; font-weight: bold;")
        self.proceed_button.clicked.connect(self.proceed_to_training)
        self.proceed_button.setEnabled(False)
        
        button_layout.addWidget(back_button)
        button_layout.addStretch()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.proceed_button)
        
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)
    
    def create_configuration_tab(self):
        """Create the configuration tab"""
        scroll = QScrollArea()
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Optimization settings
        settings_group = QGroupBox("Bayesian Optimization Settings")
        settings_layout = QFormLayout()
        
        self.n_trials_spin = QSpinBox()
        self.n_trials_spin.setRange(5, 1000)
        self.n_trials_spin.setValue(50)
        self.n_trials_spin.setToolTip("Number of optimization trials to run\nMore trials = better optimization but longer time")
        
        self.n_best_configs_spin = QSpinBox()
        self.n_best_configs_spin.setRange(1, 20)
        self.n_best_configs_spin.setValue(5)
        self.n_best_configs_spin.setToolTip("Number of best configurations to select for final training")
        
        # Sampler selection
        self.sampler_combo = QComboBox()
        self.sampler_combo.addItems(["TPE (Recommended)", "Random", "CMA-ES"])
        self.sampler_combo.setToolTip("TPE (Tree-structured Parzen Estimator) is recommended for most cases")
        
        # Pruner selection
        self.pruner_combo = QComboBox()
        self.pruner_combo.addItems(["Median Pruner", "No Pruning", "Hyperband"])
        self.pruner_combo.setToolTip("Median Pruner stops unpromising trials early to save time")
        
        settings_layout.addRow("Number of Trials:", self.n_trials_spin)
        settings_layout.addRow("Best Configs to Select:", self.n_best_configs_spin)
        settings_layout.addRow("Sampling Algorithm:", self.sampler_combo)
        settings_layout.addRow("Pruning Strategy:", self.pruner_combo)
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Parameter ranges info (read-only display)
        ranges_group = QGroupBox("Parameter Search Ranges (from Hyperparameter GUI)")
        ranges_layout = QVBoxLayout()
        
        self.ranges_display = QTextEdit()
        self.ranges_display.setReadOnly(True)
        self.ranges_display.setMaximumHeight(150)
        self.ranges_display.setStyleSheet("background-color: #f5f5f5; border: 1px solid #ddd;")
        
        # Display the parameter ranges from hyperparameter GUI
        ranges_text = self._format_parameter_ranges()
        self.ranges_display.setText(ranges_text)
        
        ranges_layout.addWidget(self.ranges_display)
        ranges_group.setLayout(ranges_layout)
        layout.addWidget(ranges_group)
        
        # Batch size choices
        ranges_layout.addWidget(QLabel("Batch Size Options:"))
        self.batch_sizes = QLineEdit("32,64,128,256,512")
        self.batch_sizes.setToolTip("Comma-separated batch size options")
        ranges_layout.addWidget(self.batch_sizes)
        
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        return scroll
    
    def _format_parameter_ranges(self):
        """Format parameter ranges for display"""
        ranges_text = "Parameter search ranges configured in Hyperparameter GUI:\n\n"
        
        for param_name, param_value in self.param_ranges.items():
            if isinstance(param_value, str) and '[' in param_value and ']' in param_value:
                ranges_text += f"• {param_name}: {param_value}\n"
            else:
                ranges_text += f"• {param_name}: {param_value} (single value)\n"
        
        if not self.param_ranges:
            ranges_text += "No parameter ranges configured.\nPlease set boundary ranges [min,max] in the Hyperparameter GUI."
        
        return ranges_text
    
    def create_optimization_tab(self):
        """Create the optimization tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Progress section
        progress_group = QGroupBox("Optimization Progress")
        progress_layout = QVBoxLayout()
        
        self.progress_label = QLabel("Ready to start optimization")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Log section
        log_group = QGroupBox("Optimization Log")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(300)
        
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        # Current trial info
        trial_group = QGroupBox("Current Trial Information")
        trial_layout = QVBoxLayout()
        
        self.trial_table = QTableWidget()
        self.trial_table.setColumnCount(3)
        self.trial_table.setHorizontalHeaderLabels(["Trial", "Parameters", "Objective Value"])
        self.trial_table.horizontalHeader().setStretchLastSection(True)
        
        trial_layout.addWidget(self.trial_table)
        trial_group.setLayout(trial_layout)
        layout.addWidget(trial_group)
        
        return widget
    
    def create_results_tab(self):
        """Create the results tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Results table
        results_group = QGroupBox("Best Configurations")
        results_layout = QVBoxLayout()
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["Rank", "Objective Value", "Trial #", "Parameters"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        
        results_layout.addWidget(self.results_table)
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        # Export buttons
        export_layout = QHBoxLayout()
        
        export_button = QPushButton("Export Results to JSON")
        export_button.clicked.connect(self.export_results)
        
        export_layout.addWidget(export_button)
        export_layout.addStretch()
        
        layout.addLayout(export_layout)
        
        return widget
    
    def start_optimization(self):
        """Start the Optuna optimization"""
        try:
            # Get optimization configuration
            config = self._get_optimization_config()
            
            # Switch to Optimization tab to show progress
            self.tab_widget.setCurrentIndex(1)  # Optimization tab is at index 1
            
            # Disable start button, enable stop button
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.proceed_button.setEnabled(False)
            
            # Clear previous results
            self.trial_table.setRowCount(0)
            self.results_table.setRowCount(0)
            self.log_text.clear()
            
            # Show progress bar
            self.progress_bar.setVisible(True)
            self.progress_bar.setMaximum(config['n_trials'])
            self.progress_bar.setValue(0)
            
            # Start optimization thread
            self.optimization_thread = OptunaOptimizationThread(self.base_params, config)
            self.optimization_thread.progress_updated.connect(self.update_progress)
            self.optimization_thread.trial_completed.connect(self.trial_completed)
            self.optimization_thread.optimization_completed.connect(self.optimization_completed)
            self.optimization_thread.error_occurred.connect(self.optimization_error)
            self.optimization_thread.log_message.connect(self.add_log_message)
            
            self.optimization_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start optimization: {str(e)}")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
    
    def stop_optimization(self):
        """Stop the optimization"""
        if self.optimization_thread:
            self.optimization_thread.stop_optimization()
            self.add_log_message("Stopping optimization...")
    
    def _get_optimization_config(self):
        """Get optimization configuration from GUI"""
        config = {
            'n_trials': self.n_trials_spin.value(),
            'n_best_configs': self.n_best_configs_spin.value(),
            'sampler': self.sampler_combo.currentText(),
            'pruner': self.pruner_combo.currentText()
        }
        
        return config
        
        if hasattr(self, 'dropout_min'):
            config['dropout_range'] = (self.dropout_min.value(), self.dropout_max.value())
        
        return config
    
    def update_progress(self, current, total):
        """Update progress bar"""
        self.progress_bar.setValue(current)
        self.progress_label.setText(f"Optimization Progress: {current}/{total} trials completed")
    
    def trial_completed(self, trial_info):
        """Handle completed trial"""
        row = self.trial_table.rowCount()
        self.trial_table.insertRow(row)
        
        self.trial_table.setItem(row, 0, QTableWidgetItem(str(trial_info['trial_number'])))
        self.trial_table.setItem(row, 1, QTableWidgetItem(str(trial_info['params'])))
        self.trial_table.setItem(row, 2, QTableWidgetItem(f"{trial_info['value']:.6f}"))
        
        # Auto-scroll to latest trial
        self.trial_table.scrollToBottom()
    
    def optimization_completed(self, best_configs):
        """Handle completed optimization"""
        self.best_configs = best_configs
        
        # Switch to Results tab to show the final results
        self.tab_widget.setCurrentIndex(2)  # Results tab is at index 2
        
        # Reset buttons
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.proceed_button.setEnabled(True)
        
        # Hide progress bar
        self.progress_bar.setVisible(False)
        self.progress_label.setText("Optimization completed!")
        
        # Update results table
        self.results_table.setRowCount(len(best_configs))
        for i, config in enumerate(best_configs):
            self.results_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.results_table.setItem(i, 1, QTableWidgetItem(f"{config['objective_value']:.6f}"))
            self.results_table.setItem(i, 2, QTableWidgetItem(str(config['trial_number'])))
            self.results_table.setItem(i, 3, QTableWidgetItem(str(config['params'])))
        
        self.add_log_message(f"Optimization completed! Found {len(best_configs)} best configurations.")
    
    def optimization_error(self, error_msg):
        """Handle optimization error"""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setVisible(False)
        
        QMessageBox.critical(self, "Optimization Error", error_msg)
        self.add_log_message(f"ERROR: {error_msg}")
    
    def add_log_message(self, message):
        """Add message to log"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
    
    def export_results(self):
        """Export results to JSON file automatically in job folder"""
        if not self.best_configs:
            QMessageBox.warning(self, "No Results", "No optimization results to export.")
            return
        
        try:
            # Get the job folder automatically
            job_manager = JobManager()
            job_folder = job_manager.get_job_folder()
            
            if not job_folder or not os.path.exists(job_folder):
                QMessageBox.critical(self, "Export Error", "Could not find job folder.")
                return
            
            # Create filename with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(job_folder, f"optuna_best_configs_{timestamp}.json")
            
            # Export the results
            with open(filename, 'w') as f:
                json.dump(self.best_configs, f, indent=2)
            
            QMessageBox.information(self, "Export Successful", 
                                  f"Results automatically saved to:\n{filename}")
                
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export results: {str(e)}")
    
    def proceed_to_training(self):
        """Proceed to task preparation with selected configurations"""
        if not self.best_configs:
            QMessageBox.warning(self, "No Results", "No optimization results available.")
            return
        
        try:
            # Import the new Optuna task setup manager
            from vestim.gui.src.optuna_task_setup_manager_qt import OptunaPrepareTaskManager
            
            # Create the task setup manager with base params and best configs
            self.task_setup_manager = OptunaPrepareTaskManager(self.base_params, self.best_configs)
            self.task_setup_manager.show()
            self.close()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to proceed to task preparation: {str(e)}")
    
    def go_back(self):
        """Go back to hyperparameter GUI"""
        self.close()
        # Import here to avoid circular imports
        from vestim.gui.src.hyper_param_gui_qt import VEstimHyperParamGUI
        self.hyper_param_gui = VEstimHyperParamGUI()
        self.hyper_param_gui.show()


if __name__ == "__main__":
    app = QApplication([])
    
    # Test with sample parameters
    test_params = {
        'MODEL_TYPE': 'LSTM',
        'FEATURE_COLUMNS': ['SOC', 'Current', 'Temp'],
        'TARGET_COLUMN': 'Voltage',
        'TRAINING_METHOD': 'Sequence-to-Sequence',
        'HIDDEN_UNITS': '10',
        'LAYERS': '1'
    }
    
    gui = VEstimOptunaOptimizationGUI(test_params)
    gui.show()
    app.exec_()
