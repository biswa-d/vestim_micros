import os
import json
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton,
    QLineEdit, QFileDialog, QMessageBox, QGroupBox, QComboBox, QListWidget, QAbstractItemView, QCheckBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon
import pandas as pd

# Make PyTorch optional in the frontend
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available in frontend. This is normal - PyTorch is only required in the backend.")

from vestim.backend.src.services.job_service import JobService
from vestim.backend.src.managers.hyper_param_manager_qt import VEstimHyperParamManager
import logging

class VEstimHyperParamWidget(QWidget):
    params_configured = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.params = {}
        self.job_id = None
        self.job_service = JobService()
        self.hyper_param_manager = VEstimHyperParamManager()
        self.param_entries = {}
        self.build_gui()

    def set_job_id(self, job_id):
        self.job_id = job_id
        self.job_folder = self.job_service.get_job_folder(job_id)
        self.update_feature_target_selection()

    def build_gui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)

        title_label = QLabel("Select Hyperparameters for Model Training")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 16pt; font-weight: bold; margin-bottom: 10px;")
        main_layout.addWidget(title_label)

        hyperparam_section = QGridLayout()

        feature_target_group = QGroupBox()
        feature_target_layout = QVBoxLayout()
        self.add_feature_target_selection(feature_target_layout)
        feature_target_group.setLayout(feature_target_layout)
        hyperparam_section.addWidget(feature_target_group, 0, 0)

        model_selection_group = QGroupBox()
        model_selection_layout = QVBoxLayout()
        self.add_model_selection(model_selection_layout)
        model_selection_group.setLayout(model_selection_layout)
        hyperparam_section.addWidget(model_selection_group, 0, 1)

        training_method_group = QGroupBox()
        training_method_layout = QVBoxLayout()
        self.add_training_method_selection(training_method_layout)
        training_method_group.setLayout(training_method_layout)
        hyperparam_section.addWidget(training_method_group, 0, 2)

        scheduler_group = QGroupBox()
        scheduler_layout = QVBoxLayout()
        self.add_scheduler_selection(scheduler_layout)
        scheduler_group.setLayout(scheduler_layout)
        hyperparam_section.addWidget(scheduler_group, 1, 0)

        validation_group = QGroupBox()
        validation_criteria_layout = QVBoxLayout()
        self.add_validation_criteria(validation_criteria_layout)
        validation_group.setLayout(validation_criteria_layout)
        hyperparam_section.addWidget(validation_group, 1, 1)

        device_selection_group = QGroupBox()
        device_selection_layout = QVBoxLayout()
        self.add_device_selection(device_selection_layout)
        device_selection_group.setLayout(device_selection_layout)
        hyperparam_section.addWidget(device_selection_group, 1, 2)

        main_layout.addLayout(hyperparam_section)

        button_layout = QVBoxLayout()
        start_button = QPushButton("Create Training Tasks")
        start_button.clicked.connect(self.proceed_to_training)
        button_layout.addWidget(start_button, alignment=Qt.AlignCenter)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

    def add_feature_target_selection(self, layout):
        feature_label = QLabel("Feature Columns (Input):")
        self.feature_list = QListWidget()
        self.feature_list.setSelectionMode(QAbstractItemView.MultiSelection)
        target_label = QLabel("Target Column (Output):")
        self.target_combo = QComboBox()
        self.param_entries["FEATURE_COLUMNS"] = self.feature_list
        self.param_entries["TARGET_COLUMN"] = self.target_combo
        v_layout = QVBoxLayout()
        v_layout.addWidget(feature_label)
        v_layout.addWidget(self.feature_list)
        v_layout.addWidget(target_label)
        v_layout.addWidget(self.target_combo)
        layout.addLayout(v_layout)

    def update_feature_target_selection(self):
        column_names = self.load_column_names()
        self.feature_list.clear()
        self.feature_list.addItems(column_names)
        self.target_combo.clear()
        self.target_combo.addItems(column_names)

    def add_training_method_selection(self, layout):
        training_layout = QVBoxLayout()
        training_method_label = QLabel("Training Method:")
        self.training_method_combo = QComboBox()
        self.training_method_combo.addItems(["Sequence-to-Sequence", "Whole Sequence"])
        self.lookback_label = QLabel("Lookback Window:")
        self.lookback_entry = QLineEdit("400")
        self.batch_training_checkbox = QCheckBox("Enable Batch Training")
        self.batch_training_checkbox.setChecked(True)
        self.batch_size_label = QLabel("Batch Size:")
        self.batch_size_entry = QLineEdit("100")
        train_val_split_label = QLabel("Train-Valid Split:")
        self.train_val_split_entry = QLineEdit("0.8")
        self.param_entries["TRAINING_METHOD"] = self.training_method_combo
        self.param_entries["LOOKBACK"] = self.lookback_entry
        self.param_entries["BATCH_TRAINING"] = self.batch_training_checkbox
        self.param_entries["BATCH_SIZE"] = self.batch_size_entry
        self.param_entries["TRAIN_VAL_SPLIT"] = self.train_val_split_entry
        training_layout.addWidget(training_method_label)
        training_layout.addWidget(self.training_method_combo)
        training_layout.addWidget(self.lookback_label)
        training_layout.addWidget(self.lookback_entry)
        training_layout.addWidget(self.batch_training_checkbox)
        training_layout.addWidget(self.batch_size_label)
        training_layout.addWidget(self.batch_size_entry)
        training_layout.addWidget(train_val_split_label)
        training_layout.addWidget(self.train_val_split_entry)
        layout.addLayout(training_layout)

    def add_model_selection(self, layout):
        model_layout = QVBoxLayout()
        model_label = QLabel("Select Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["LSTM", "FNN", "GRU"])
        self.model_param_container = QVBoxLayout()
        self.param_entries["MODEL_TYPE"] = self.model_combo
        self.model_combo.currentIndexChanged.connect(self.update_model_params)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        model_layout.addLayout(self.model_param_container)
        layout.addLayout(model_layout)
        self.update_model_params()

    def update_model_params(self):
        selected_model = self.model_combo.currentText()
        while self.model_param_container.count():
            item = self.model_param_container.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        
        model_params = {}
        if selected_model == "LSTM":
            lstm_layers_label = QLabel("LSTM Layers:")
            self.lstm_layers_entry = QLineEdit("1")
            hidden_units_label = QLabel("Hidden Units:")
            self.hidden_units_entry = QLineEdit("10")
            self.model_param_container.addWidget(lstm_layers_label)
            self.model_param_container.addWidget(self.lstm_layers_entry)
            self.model_param_container.addWidget(hidden_units_label)
            self.model_param_container.addWidget(self.hidden_units_entry)
            model_params["LAYERS"] = self.lstm_layers_entry
            model_params["HIDDEN_UNITS"] = self.hidden_units_entry
        self.param_entries.update(model_params)

    def add_scheduler_selection(self, layout):
        scheduler_layout = QVBoxLayout()
        scheduler_label = QLabel("Learning Rate Scheduler:")
        self.scheduler_combo = QComboBox()
        self.scheduler_combo.addItems(["StepLR", "ReduceLROnPlateau"])
        initial_lr_label = QLabel("Initial Learning Rate:")
        self.initial_lr_entry = QLineEdit("0.0001")
        self.lr_param_label = QLabel("LR Drop Factor:")
        self.lr_param_entry = QLineEdit("0.1")
        self.param_entries["SCHEDULER_TYPE"] = self.scheduler_combo
        self.param_entries["INITIAL_LR"] = self.initial_lr_entry
        self.param_entries["LR_DROP_FACTOR"] = self.lr_param_entry
        scheduler_layout.addWidget(scheduler_label)
        scheduler_layout.addWidget(self.scheduler_combo)
        scheduler_layout.addWidget(initial_lr_label)
        scheduler_layout.addWidget(self.initial_lr_entry)        
        scheduler_layout.addWidget(self.lr_param_label)
        scheduler_layout.addWidget(self.lr_param_entry)
        layout.addLayout(scheduler_layout)

    def add_validation_criteria(self, layout):
        validation_layout = QVBoxLayout()
        epochs_label = QLabel("Epochs:")
        self.epochs_entry = QLineEdit("1")
        patience_label = QLabel("Patience:")
        self.patience_entry = QLineEdit("10")
        self.param_entries["EPOCHS"] = self.epochs_entry
        self.param_entries["PATIENCE"] = self.patience_entry
        validation_layout.addWidget(epochs_label)
        validation_layout.addWidget(self.epochs_entry)
        validation_layout.addWidget(patience_label)
        validation_layout.addWidget(self.patience_entry)
        layout.addLayout(validation_layout)
        
    def add_device_selection(self, layout):
        device_layout = QVBoxLayout()
        
        device_label = QLabel("Device:")
        self.device_combo = QComboBox()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.device_combo.addItems(["cuda", "cpu"])
        else:
            self.device_combo.addItems(["cpu"])
        self.param_entries["DEVICE"] = self.device_combo
        device_layout.addWidget(device_label)
        device_layout.addWidget(self.device_combo)
        layout.addLayout(device_layout)

    def proceed_to_training(self):
        try:
            params = self.collect_parameters()
            self.hyper_param_manager.save_params(self.job_id, params)
            self.params_configured.emit(self.job_id)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to proceed to training: {e}")

    def load_column_names(self):
        if not self.job_folder:
            return []
        train_data_path = os.path.join(self.job_folder, "train_data", "processed_data")
        if not os.path.exists(train_data_path):
            return []
        
        for file in os.listdir(train_data_path):
            if file.endswith(".csv"):
                try:
                    df = pd.read_csv(os.path.join(train_data_path, file))
                    return df.columns.tolist()
                except Exception as e:
                    self.logger.error(f"Error reading csv file {file}: {e}")
        return []

    def collect_parameters(self):
        params = {}
        for key, widget in self.param_entries.items():
            if isinstance(widget, QLineEdit):
                params[key] = widget.text()
            elif isinstance(widget, QComboBox):
                params[key] = widget.currentText()
            elif isinstance(widget, QListWidget):
                params[key] = [item.text() for item in widget.selectedItems()]
            elif isinstance(widget, QCheckBox):
                params[key] = widget.isChecked()
        return params

if __name__ == '__main__':
    app = QApplication([])
    # A dummy job_id is required for the widget to initialize.
    # In a real application, this would be provided by the dashboard.
    widget = VEstimHyperParamWidget()
    widget.set_job_id("dummy_job_id") 
    widget.show()
    app.exec_()