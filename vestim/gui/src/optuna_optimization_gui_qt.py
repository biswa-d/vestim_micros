# ---------------------------------------------------------------------------------
# Author: Biswanath Dehury
# Date: 2025-07-14
# Version: 1.0.0
# Description: Optuna-based hyperparameter optimization GUI for VEstim
# ---------------------------------------------------------------------------------

import torch
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
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject
from PyQt5.QtGui import QIcon, QFont

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from vestim.gateway.src.job_manager_qt import JobManager
from vestim.gui.src.training_setup_gui_qt import VEstimTrainSetupGUI
from vestim.gateway.src.training_setup_manager_qt import VEstimTrainingSetupManager


class OptunaOptimizationThread(QThread):
    """Thread for running Optuna optimization in the background"""
    
    # Signals for communicating with the main GUI thread
    progress_updated = pyqtSignal(int, int)  # current_trial, total_trials
    trial_completed = pyqtSignal(dict)  # trial_info
    optimization_completed = pyqtSignal(list, list)  # best_configs, all_completed_trials
    error_occurred = pyqtSignal(str)  # error_message
    log_message = pyqtSignal(str)  # log_message
    
    def __init__(self, base_params, optimization_config, job_manager, parent=None):
        super().__init__(parent)
        self.base_params = base_params
        self.optimization_config = optimization_config
        self.job_manager = job_manager
        self.study = None
        self.should_stop = False
        self.completed_trials_data = []
        self.current_task_manager = None
        self.pruning_initiated = False
        
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
                    
                trial = None
                params = {}
                try:
                    # Run single trial
                    trial = self.study.ask()
                    params = self._suggest_params(trial)
                    
                    # Evaluate parameters (simulate training for now)
                    objective_value = self._evaluate_params(params, trial)
                    
                    # Tell study the result
                    self.study.tell(trial, objective_value)
                    
                    # Store the complete data for this successful trial
                    trial_data = {
                        'params': params,
                        'objective_value': objective_value,
                        'trial_number': trial.number
                    }
                    self.completed_trials_data.append(trial_data)

                    # Emit progress and trial info
                    self.progress_updated.emit(trial_num + 1, n_trials)
                    self.trial_completed.emit({
                        'trial_number': trial_num + 1,
                        'params': params,
                        'value': objective_value,
                        'state': 'COMPLETE'
                    })
                    
                    self.log_message.emit(f"Trial {trial_num + 1}: objective = {objective_value:.6f}")

                except optuna.exceptions.TrialPruned:
                    self.log_message.emit(f"Trial {trial_num + 1} pruned.")
                    self.trial_completed.emit({
                        'trial_number': trial_num + 1,
                        'params': params,
                        'value': 'N/A',
                        'state': 'PRUNED'
                    })

                except Exception as e:
                    self.log_message.emit(f"Trial {trial_num + 1} failed: {str(e)}")
                    self.trial_completed.emit({
                        'trial_number': trial_num + 1,
                        'params': params,
                        'value': float('inf'),
                        'state': 'FAIL'
                    })
            
            # Get best configurations regardless of whether the optimization was stopped or completed
            best_configs = self._get_best_configurations()
            if self.should_stop:
                self.log_message.emit(f"Optimization stopped by user. Processing {len(best_configs)} best completed trials.")
            else:
                self.log_message.emit(f"Optimization completed! Found {len(best_configs)} best configurations.")
            self.optimization_completed.emit(best_configs, self.completed_trials_data)
                
        except Exception as e:
            self.error_occurred.emit(f"Optimization failed: {str(e)}")
    
    def _suggest_params(self, trial):
        """Suggest parameters for a trial, with special handling for dynamic FNN architecture search."""
        params = {}
        handled_params = set()

        model_type = self.base_params.get('MODEL_TYPE')

        # SOTA Dynamic FNN Architecture Search
        if model_type == 'FNN' and 'FNN_N_LAYERS' in self.base_params and 'FNN_UNITS' in self.base_params:
            try:
                # Suggest number of layers from the specified range
                n_layers_range_str = self.base_params['FNN_N_LAYERS']
                n_layers_range = json.loads(n_layers_range_str)
                n_layers = trial.suggest_int('FNN_N_LAYERS', n_layers_range[0], n_layers_range[1])

                # Suggest number of units for each layer
                units_range_str = self.base_params['FNN_UNITS']
                units_range = json.loads(units_range_str)
                
                hidden_layers_config = []
                for i in range(n_layers):
                    unit_param_name = f'FNN_UNITS_L{i}'
                    units = trial.suggest_int(unit_param_name, units_range[0], units_range[1])
                    hidden_layers_config.append(str(units))
                
                # This is the parameter the rest of the system uses
                params['FNN_HIDDEN_LAYERS'] = ",".join(hidden_layers_config)
                
                # Mark these as handled so the main loop doesn't process them again
                handled_params.update(['FNN_N_LAYERS', 'FNN_UNITS', 'FNN_HIDDEN_LAYERS'])

            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
                self.log_message.emit(f"Could not parse FNN dynamic ranges: {e}. Falling back to predefined architectures.")

        # General parameter suggestion loop
        integer_params = {
            "LAYERS", "HIDDEN_UNITS", "GRU_LAYERS", "GRU_HIDDEN_UNITS",
            "MAX_EPOCHS", "VALID_PATIENCE", "VALID_FREQUENCY", "LOOKBACK",
            "BATCH_SIZE", "LR_PERIOD", "PLATEAU_PATIENCE", "REPETITIONS"
        }
        float_log_params = {"INITIAL_LR", "LR_PARAM", "PLATEAU_FACTOR", "FNN_DROPOUT_PROB"}
        categorical_params = {"FNN_HIDDEN_LAYERS"}

        for param_name, param_value in self.base_params.items():
            if param_name in handled_params:
                continue

            if param_name in categorical_params and isinstance(param_value, str) and ';' in param_value:
                choices = [choice.strip() for choice in param_value.split(';')]
                suggested_value = trial.suggest_categorical(param_name, choices)
                params[param_name] = suggested_value
            elif isinstance(param_value, str) and param_value.startswith('[') and param_value.endswith(']'):
                try:
                    inner = param_value[1:-1].strip()
                    parts = [part.strip() for part in inner.split(',')]
                    
                    if len(parts) == 2:
                        min_val, max_val = float(parts[0]), float(parts[1])
                        
                        if param_name in integer_params:
                            suggested_value = trial.suggest_int(param_name, int(min_val), int(max_val))
                        elif param_name in float_log_params:
                            suggested_value = trial.suggest_float(param_name, min_val, max_val, log=True)
                        else:
                            suggested_value = trial.suggest_float(param_name, min_val, max_val)
                        
                        params[param_name] = str(suggested_value)
                    else:
                        params[param_name] = param_value
                except (ValueError, IndexError):
                    params[param_name] = param_value
            else:
                params[param_name] = param_value
                
        return params
    
    def _evaluate_params(self, params, trial):
        """
        Evaluate parameters by running a full, robust training task for a single configuration.
        This now uses the main application's TrainingTaskManager to ensure a perfect simulation.
        """
        self.log_message.emit(f"--- Evaluating Trial {trial.number} ---")
        
        from vestim.gateway.src.training_task_manager_qt import TrainingTaskManager
        from vestim.services.model_training.src.training_task_service import TrainingTaskService

        try:
            # Create a temporary, in-memory task for evaluation
            task_service = TrainingTaskService()
            model_service_map = {
                "LSTM": "vestim.services.model_training.src.LSTM_model_service.LSTMModelService",
                "GRU": "vestim.services.model_training.src.GRU_model_service.GRUModelService",
                "FNN": "vestim.services.model_training.src.FNN_model_service.FNNModelService",
            }
            model_service_class = self._get_class_from_string(model_service_map[params['MODEL_TYPE']])
            model_service = model_service_class()

            device = torch.device(params.get('DEVICE_SELECTION', 'cuda:0') if torch.cuda.is_available() else 'cpu')
            model = model_service.create_model(params, device=device)
            
            training_task = {
                'task_id': f"optuna_trial_{trial.number}",
                'model': model,
                'hyperparams': params,
                'optuna_trial': trial,
                'log_callback': self.log_message.emit,
                'job_folder_augmented_from': self.job_manager.get_job_folder(),
                'data_loader_params': {
                    'lookback': int(params['LOOKBACK']),
                    'batch_size': int(params['BATCH_SIZE']),
                    'feature_columns': self.base_params['FEATURE_COLUMNS'],
                    'target_column': self.base_params['TARGET_COLUMN'],
                    'num_workers': 4
                },
                'training_params': {
                    'early_stopping': True,
                    'early_stopping_patience': int(params['VALID_PATIENCE']),
                    'save_best_model': False, # Do not save models during trials
                },
                'results': {}
            }

            task_manager = TrainingTaskManager(global_params=self.base_params)
            
            class SignalEmitter(QObject):
                progress_signal = pyqtSignal(dict)
            
            emitter = SignalEmitter()
            
            self.current_task_manager = task_manager
            try:
                task_manager.process_task(training_task, emitter.progress_signal)
            finally:
                self.current_task_manager = None

            final_results = training_task.get('results', {})
            best_val_loss = final_results.get('best_validation_loss_normalized', float('inf'))

            if best_val_loss == float('inf'):
                self.log_message.emit(f"Trial {trial.number} finished, but a valid validation loss was not found.")
            else:
                self.log_message.emit(f"--- Finished Trial {trial.number} with best validation loss: {best_val_loss:.6f} ---")

            return best_val_loss

        except optuna.exceptions.TrialPruned:
            self.log_message.emit(f"--- Trial {trial.number} pruned. ---")
            raise
        except Exception as e:
            self.log_message.emit(f"An error occurred during trial {trial.number}: {e}")
            import traceback
            self.log_message.emit(traceback.format_exc())
            return float('inf')
    
    def _get_class_from_string(self, class_path):
        """Dynamically import a class from a string path."""
        module_path, class_name = class_path.rsplit('.', 1)
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    def _get_best_configurations(self):
        """Get the best N configurations from the collected trial data, ignoring pruned trials."""
        n_configs = self.optimization_config.get('n_best_configs', 5)
        
        # Filter out pruned or failed trials (where objective_value is not a float)
        completed_trials = [t for t in self.completed_trials_data if isinstance(t['objective_value'], float)]
        
        # Sort the completed trials by the objective value
        sorted_trials = sorted(completed_trials, key=lambda t: t['objective_value'])
        
        # Return the top N configurations.
        return sorted_trials[:n_configs]
    
    def stop_optimization(self):
        """Stop the optimization"""
        self.should_stop = True
        if self.current_task_manager:
            self.current_task_manager.stop_task()


class VEstimOptunaOptimizationGUI(QWidget):
    """GUI for Optuna-based hyperparameter optimization"""
    
    def __init__(self, base_params):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.base_params = base_params
        self.job_manager = JobManager()
        self.optimization_thread = None
        self.best_configs = []
        self.completed_trials_data = []
        self.all_trials_data = []
        self.auto_proceed_timer = QTimer(self)
        self.auto_proceed_timer.setSingleShot(True)
        self.auto_proceed_timer.timeout.connect(self.proceed_to_training_setup)
        
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
        
        self.proceed_button = QPushButton("Proceed to Training Setup →")
        self.proceed_button.setFixedHeight(35)
        self.proceed_button.setStyleSheet("background-color: #0b6337; color: white; font-weight: bold;")
        self.proceed_button.clicked.connect(self.proceed_to_training_setup)
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
        self.trial_table.setColumnCount(4)
        self.trial_table.setHorizontalHeaderLabels(["Trial", "State", "Objective Value", "Parameters"])
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
            self.all_trials_data.clear()
            
            # Show progress bar
            self.progress_bar.setVisible(True)
            self.progress_bar.setMaximum(config['n_trials'])
            self.progress_bar.setValue(0)
            
            # Start optimization thread
            self.optimization_thread = OptunaOptimizationThread(self.base_params, config, self.job_manager)
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
        self.all_trials_data.append(trial_info)
        row = self.trial_table.rowCount()
        self.trial_table.insertRow(row)
        
        self.trial_table.setItem(row, 0, QTableWidgetItem(str(trial_info['trial_number'])))
        self.trial_table.setItem(row, 1, QTableWidgetItem(trial_info['state']))
        
        value_str = f"{trial_info['value']:.6f}" if isinstance(trial_info['value'], float) else str(trial_info['value'])
        self.trial_table.setItem(row, 2, QTableWidgetItem(value_str))
        
        self.trial_table.setItem(row, 3, QTableWidgetItem(str(trial_info['params'])))
        
        # Auto-scroll to latest trial
        self.trial_table.scrollToBottom()
    
    def optimization_completed(self, best_configs, completed_trials_data):
        """Handle completed optimization"""
        self.best_configs = best_configs
        self.completed_trials_data = completed_trials_data
        
        # Switch to Results tab to show the final results
        self.tab_widget.setCurrentIndex(2)  # Results tab is at index 2
        
        # Reset buttons
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.proceed_button.setEnabled(True)
        self.auto_proceed_timer.start(60000)  # 60 seconds
        
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
        
        # Provide a more detailed summary
        total_trials_run = len(self.all_trials_data)
        completed_count = len([t for t in self.all_trials_data if t.get('state') == 'COMPLETE'])
        pruned_count = len([t for t in self.all_trials_data if t.get('state') == 'PRUNED'])
        failed_count = len([t for t in self.all_trials_data if t.get('state') == 'FAIL'])

        summary_message = (
            f"Optimization finished! Found {len(best_configs)} best configurations from "
            f"{total_trials_run} trials run "
            f"({completed_count} completed, {pruned_count} pruned, {failed_count} failed)."
        )
        self.add_log_message(summary_message)
    
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
    
    def proceed_to_training_setup(self):
        """Proceed to the training setup GUI with the best configurations."""
        self.auto_proceed_timer.stop()  # Stop timer if manually clicked or auto-triggered
        if not self.best_configs:
            QMessageBox.warning(self, "No Results", "No optimization results available.")
            return
        
        try:
            self.logger.info("Proceeding to training setup with the following configurations:")
            self.logger.info(json.dumps(self.best_configs, indent=2))
            self.close()
            # Pass both the original base_params (for display) and the best_configs (for task creation)
            self.training_setup_gui = VEstimTrainSetupGUI(
                params=self.base_params,
                optuna_configs=self.best_configs,
                job_manager=self.job_manager
            )
            self.training_setup_gui.show()
        except Exception as e:
            self.logger.error(f"Error proceeding to training setup: {e}")
            QMessageBox.critical(self, "Error", f"Failed to open training setup: {str(e)}")
    
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
