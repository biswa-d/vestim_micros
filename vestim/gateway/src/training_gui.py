def setup_hyperparameter_inputs(self):
    """Setup the hyperparameter input fields."""
    # ... existing input fields ...
    
    # Add max epochs input
    self.max_epochs_label = QLabel("Max Epochs:")
    self.max_epochs_input = QSpinBox()
    self.max_epochs_input.setRange(1, 10000)
    self.max_epochs_input.setValue(100)  # Default value
    
    # Add to layout
    self.form_layout.addRow(self.max_epochs_label, self.max_epochs_input) 