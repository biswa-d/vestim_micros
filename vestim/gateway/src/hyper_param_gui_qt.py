def setup_training_params(self):
    """Setup training parameters section."""
    training_group = QGroupBox("Training Parameters")
    training_layout = QFormLayout()

    # Existing training parameter inputs...

    # Add Max Epochs input
    self.max_epochs_spinbox = QSpinBox()
    self.max_epochs_spinbox.setRange(1, 10000)  # Reasonable range for epochs
    self.max_epochs_spinbox.setValue(100)  # Default value
    self.max_epochs_spinbox.setToolTip("Maximum number of training epochs")
    training_layout.addRow("Max Epochs:", self.max_epochs_spinbox)

    # Add to existing layout
    training_group.setLayout(training_layout)
    return training_group

def get_current_params(self):
    """Get current parameters from GUI inputs."""
    params = {
        # Existing parameters...
        'MAX_EPOCHS': self.max_epochs_spinbox.value(),
        # Other parameters...
    }
    return params

def set_params(self, params):
    """Set parameters in GUI inputs."""
    # Existing parameter settings...
    self.max_epochs_spinbox.setValue(int(params.get('MAX_EPOCHS', 100)))
    # Other parameter settings... 