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

    # Max Epochs
    self.max_epochs_input = QLineEdit()
    self.max_epochs_input.setPlaceholderText("100")
    training_layout.addRow("Max Epochs:", self.max_epochs_input)

    # Learning Rate
    self.lr_input = QLineEdit()
    self.lr_input.setPlaceholderText("0.0001")
    training_layout.addRow("Initial Learning Rate:", self.lr_input)

    # Other existing parameters...

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
        'MAX_EPOCHS': self.max_epochs_input.text(),
    }
    return params

def set_params(self, params):
    """Set parameters in GUI inputs."""
    # ... existing parameter settings ...
    self.max_epochs_input.setText(str(params.get('MAX_EPOCHS', '100')))
    # ... other parameter settings ... 