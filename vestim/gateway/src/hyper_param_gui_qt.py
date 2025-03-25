def setup_training_params(self):
    """Set up the training parameters section of the GUI."""
    training_group = QGroupBox("Training Parameters")
    training_layout = QFormLayout()

    # ... existing parameters ...

    # Add Max Epochs input
    self.max_epochs_input = QLineEdit()
    self.max_epochs_input.setPlaceholderText("100")
    self.max_epochs_input.setText(str(self.hyper_params.get('MAX_EPOCHS', '100')))
    training_layout.addRow("Max Epochs:", self.max_epochs_input)

    # ... rest of the setup ...

def get_current_params(self):
    """Get current parameters from GUI inputs."""
    params = {
        # ... existing parameters ...
        'MAX_EPOCHS': self.max_epochs_input.text(),
        # ... other parameters ...
    }
    return params

def set_params(self, params):
    """Set parameters in GUI inputs."""
    # ... existing parameters ...
    self.max_epochs_input.setText(str(params.get('MAX_EPOCHS', '100')))
    # ... other parameters ... 