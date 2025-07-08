def get_default_hyper_params(self):
    """Return default hyperparameters."""
    return {
        'FEATURE_COLUMNS': ['Current', 'SOC', 'Temp'],
        'TARGET_COLUMN': 'Voltage',
        'MODEL_TYPE': 'LSTM',
        'LAYERS': '1',
        'HIDDEN_UNITS': '10',
        'TRAINING_METHOD': 'Sequence-to-Sequence',
        'LOOKBACK': '400',
        'BATCH_TRAINING': True,
        'BATCH_SIZE': '100',
        # Removed TRAIN_VAL_SPLIT as we now use separate train/val/test folders
        'SCHEDULER_TYPE': 'StepLR',
        'INITIAL_LR': '0.0001',
        'LR_PARAM': '0.1',
        'LR_PERIOD': '1000',
        'PLATEAU_PATIENCE': '10',
        'PLATEAU_FACTOR': '0.1',
        'VALID_PATIENCE': '10',
        'VALID_FREQUENCY': '3',
        'MAX_EPOCHS': '100',
    } 