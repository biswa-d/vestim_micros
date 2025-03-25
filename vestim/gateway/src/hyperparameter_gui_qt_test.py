from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, 
                           QLabel, QLineEdit, QComboBox, QCheckBox, QGroupBox,
                           QSpinBox)  # Add QSpinBox to imports

# ... rest of imports ...

class VEstimHyperParameterGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def setup_training_params(self):
        """Setup training parameters section."""
        training_group = QGroupBox("Training Parameters")
        training_layout = QFormLayout()

        # Batch Size
        self.batch_size_input = QLineEdit()
        self.batch_size_input.setPlaceholderText("100")
        training_layout.addRow("Batch Size:", self.batch_size_input)

        # Lookback
        self.lookback_input = QLineEdit()
        self.lookback_input.setPlaceholderText("400")
        training_layout.addRow("Lookback:", self.lookback_input)

        # Learning Rate
        self.lr_input = QLineEdit()
        self.lr_input.setPlaceholderText("0.0001")
        training_layout.addRow("Initial Learning Rate:", self.lr_input)

        # Learning Rate Parameters
        self.lr_param_input = QLineEdit()
        self.lr_param_input.setPlaceholderText("0.1")
        training_layout.addRow("Learning Rate Parameter:", self.lr_param_input)

        self.lr_period_input = QLineEdit()
        self.lr_period_input.setPlaceholderText("1000")
        training_layout.addRow("Learning Rate Period:", self.lr_period_input)

        # Plateau Parameters
        self.plateau_patience_input = QLineEdit()
        self.plateau_patience_input.setPlaceholderText("10")
        training_layout.addRow("Plateau Patience:", self.plateau_patience_input)

        self.plateau_factor_input = QLineEdit()
        self.plateau_factor_input.setPlaceholderText("0.1")
        training_layout.addRow("Plateau Factor:", self.plateau_factor_input)

        # Validation Parameters
        self.valid_patience_input = QLineEdit()
        self.valid_patience_input.setPlaceholderText("10")
        training_layout.addRow("Validation Patience:", self.valid_patience_input)

        self.valid_frequency_input = QLineEdit()
        self.valid_frequency_input.setPlaceholderText("3")
        training_layout.addRow("Validation Frequency:", self.valid_frequency_input)

        # Add Max Epochs input with spinbox
        self.max_epochs_spinbox = QSpinBox()
        self.max_epochs_spinbox.setRange(1, 10000)
        self.max_epochs_spinbox.setValue(100)
        self.max_epochs_spinbox.setToolTip("Maximum number of training epochs")
        training_layout.addRow("Max Epochs:", self.max_epochs_spinbox)

        training_group.setLayout(training_layout)
        return training_group

    def get_current_params(self):
        """Get current parameters from GUI inputs."""
        params = {
            'FEATURE_COLUMNS': self.get_feature_columns(),
            'TARGET_COLUMN': self.target_column_combo.currentText(),
            'MODEL_TYPE': self.model_type_combo.currentText(),
            'LAYERS': self.layers_input.text(),
            'HIDDEN_UNITS': self.hidden_units_input.text(),
            'TRAINING_METHOD': self.training_method_combo.currentText(),
            'LOOKBACK': self.lookback_input.text(),
            'BATCH_TRAINING': self.batch_training_checkbox.isChecked(),
            'BATCH_SIZE': self.batch_size_input.text(),
            'TRAIN_VAL_SPLIT': self.train_val_split_input.text(),
            'SCHEDULER_TYPE': self.scheduler_type_combo.currentText(),
            'INITIAL_LR': self.lr_input.text(),
            'LR_PARAM': self.lr_param_input.text(),
            'LR_PERIOD': self.lr_period_input.text(),
            'PLATEAU_PATIENCE': self.plateau_patience_input.text(),
            'PLATEAU_FACTOR': self.plateau_factor_input.text(),
            'VALID_PATIENCE': self.valid_patience_input.text(),
            'VALID_FREQUENCY': self.valid_frequency_input.text(),
            'MAX_EPOCHS': str(self.max_epochs_spinbox.value()),  # Add this line
        }
        return params

    def set_params(self, params):
        """Set parameters in GUI inputs."""
        # ... existing parameter settings ...
        if 'MAX_EPOCHS' in params:
            try:
                self.max_epochs_spinbox.setValue(int(params['MAX_EPOCHS']))
            except (ValueError, TypeError):
                self.max_epochs_spinbox.setValue(100)  # Default value if conversion fails
        # ... rest of the settings ... 