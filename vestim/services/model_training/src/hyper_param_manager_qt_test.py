def get_default_hyper_params(self):
    """Return default hyperparameters."""
    return {
        # ... existing parameters ...
        'INITIAL_LR': '0.0001',
        'LR_PARAM': '0.1',
        'LR_PERIOD': '1000',
        'PLATEAU_PATIENCE': '10',
        'PLATEAU_FACTOR': '0.1',
        'VALID_PATIENCE': '10',
        'VALID_FREQUENCY': '3',
        'MAX_EPOCHS': '100',  # Add default max epochs
        # ... other parameters ...
    } 