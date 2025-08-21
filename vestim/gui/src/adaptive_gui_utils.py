from PyQt5.QtWidgets import QLabel, QGridLayout, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt

def scale_font(font):
    """Scales font size based on screen resolution."""
    return font

def scale_widget_size(width, height=None):
    """Scales widget size based on screen resolution."""
    if height is None:
        return int(width)
    return int(width), int(height)

def get_adaptive_stylesheet(base_style):
    """Returns an adaptive stylesheet."""
    return base_style

def display_hyperparameters(gui, params):
    if not params:
        return

    frame_layout = gui.hyperparam_frame.layout()
    
    # Clear any existing widgets from the frame's layout
    if frame_layout is not None:
        while frame_layout.count():
            item = frame_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
    else:
        # If the frame has no layout, create one. This is good practice.
        frame_layout = QVBoxLayout()
        gui.hyperparam_frame.setLayout(frame_layout)

    # Create a container widget for the grid layout
    grid_container = QWidget()
    grid_layout = QGridLayout(grid_container) # Set layout on the container
    
    # Add the container widget to the frame's main layout
    frame_layout.addWidget(grid_container)

    grid_layout.setContentsMargins(20, 20, 20, 20)
    grid_layout.setHorizontalSpacing(15)
    grid_layout.setVerticalSpacing(10)

    model_type = params.get('MODEL_TYPE', 'LSTM')
    scheduler_type = params.get('SCHEDULER_TYPE', 'StepLR')

    def is_parameter_relevant(param_key, model_type, scheduler_type):
        # Exclude INPUT_SIZE and OUTPUT_SIZE from the display
        if param_key in ['INPUT_SIZE', 'OUTPUT_SIZE']:
            return False

        always_relevant = {
            'MODEL_TYPE', 'NUM_LEARNABLE_PARAMS',
            'TRAINING_METHOD', 'BATCH_TRAINING', 'BATCH_SIZE',
            'MAX_EPOCHS', 'INITIAL_LR', 'SCHEDULER_TYPE', 'VALID_PATIENCE', 
            'VALID_FREQUENCY', 'REPETITIONS', 'DEVICE_SELECTION', 'USE_MIXED_PRECISION',
            'MAX_TRAINING_TIME_SECONDS', 'FEATURE_COLUMNS', 'TARGET_COLUMN'
        }
        
        if param_key in always_relevant:
            return True
        
        if model_type in ['LSTM', 'GRU', 'LSTM_EMA', 'LSTM_LPF']:
            lstm_gru_params = {'LAYERS', 'HIDDEN_UNITS', 'LOOKBACK'}
            if param_key in lstm_gru_params:
                return True
        elif model_type == 'FNN':
            fnn_params = {'FNN_HIDDEN_LAYERS', 'FNN_DROPOUT_PROB', 'HIDDEN_LAYER_SIZES', 'DROPOUT_PROB'}
            if param_key in fnn_params:
                return True
            if param_key in {'LOOKBACK', 'LAYERS', 'HIDDEN_UNITS'}:
                return False
        
        if scheduler_type == 'StepLR':
            if param_key in {'LR_DROP_PERIOD', 'LR_PERIOD', 'LR_DROP_FACTOR', 'LR_PARAM'}:
                return True
            if param_key in {'PLATEAU_PATIENCE', 'PLATEAU_FACTOR', 'COSINE_T0', 'COSINE_T_MULT', 'COSINE_ETA_MIN'}:
                return False
        elif scheduler_type == 'ReduceLROnPlateau':
            if param_key in {'PLATEAU_PATIENCE', 'PLATEAU_FACTOR'}:
                return True
            if param_key in {'LR_DROP_PERIOD', 'LR_PERIOD', 'LR_DROP_FACTOR', 'LR_PARAM', 'COSINE_T0', 'COSINE_T_MULT', 'COSINE_ETA_MIN'}:
                return False
        elif scheduler_type == 'CosineAnnealingWarmRestarts':
            if param_key in {'COSINE_T0', 'COSINE_T_MULT', 'COSINE_ETA_MIN'}:
                return True
            if param_key in {'LR_DROP_PERIOD', 'LR_PERIOD', 'LR_DROP_FACTOR', 'LR_PARAM', 'PLATEAU_PATIENCE', 'PLATEAU_FACTOR'}:
                return False
        
        return True

    filtered_params = {k: v for k, v in params.items() if is_parameter_relevant(k, model_type, scheduler_type)}
    
    # Reordered sections for better logical flow
    data_keys = ['FEATURE_COLUMNS', 'TARGET_COLUMN']

    model_arch_keys = ['MODEL_TYPE', 'NUM_LEARNABLE_PARAMS']
    if model_type in ['LSTM', 'GRU', 'LSTM_EMA', 'LSTM_LPF']:
        model_arch_keys.extend(['LAYERS', 'HIDDEN_UNITS'])
    elif model_type == 'FNN':
        model_arch_keys.extend(['FNN_HIDDEN_LAYERS', 'HIDDEN_LAYER_SIZES', 'FNN_DROPOUT_PROB', 'DROPOUT_PROB'])
    
    train_method_keys = ['TRAINING_METHOD', 'BATCH_TRAINING', 'BATCH_SIZE']
    if model_type in ['LSTM', 'GRU', 'LSTM_EMA', 'LSTM_LPF']:
        train_method_keys.insert(1, 'LOOKBACK')
    
    train_control_keys = ['MAX_EPOCHS', 'INITIAL_LR', 'SCHEDULER_TYPE']
    if scheduler_type == 'StepLR':
        train_control_keys.extend(['LR_DROP_PERIOD', 'LR_PERIOD', 'LR_DROP_FACTOR', 'LR_PARAM'])
    elif scheduler_type == 'ReduceLROnPlateau':
        train_control_keys.extend(['PLATEAU_PATIENCE', 'PLATEAU_FACTOR'])
    elif scheduler_type == 'CosineAnnealingWarmRestarts':
        train_control_keys.extend(['COSINE_T0', 'COSINE_T_MULT', 'COSINE_ETA_MIN'])
    train_control_keys.extend(['VALID_PATIENCE', 'VALID_FREQUENCY', 'REPETITIONS'])
    
    # Grouped device and precision settings
    exec_env_keys = ['DEVICE_SELECTION', 'USE_MIXED_PRECISION', 'CURRENT_DEVICE', 'MAX_TRAINING_TIME_SECONDS']
    
    # New order: Data -> Model -> Method -> Control -> Environment
    all_ordered_keys = data_keys + model_arch_keys + train_method_keys + train_control_keys + exec_env_keys
    
    ordered_items = []
    used_keys = set()
    
    for key in all_ordered_keys:
        if key in filtered_params:
            ordered_items.append((key, filtered_params[key]))
            used_keys.add(key)
    
    for key, value in filtered_params.items():
        if key not in used_keys:
            ordered_items.append((key, value))

    items = ordered_items
    # Reduced to 3 columns to give more space and prevent text cutoff
    num_cols = 3
    num_rows = (len(items) + num_cols - 1) // num_cols

    for i, (param, value) in enumerate(items):
        row = i % num_rows
        col = (i // num_rows) * 2

        label_text = gui.param_labels.get(param, param.replace("_", " ").title())
        value_str = str(value)

        param_label = QLabel(f"{label_text}:")
        param_label.setStyleSheet("font-size: 9pt; color: #333;")
        param_label.setAlignment(Qt.AlignRight | Qt.AlignTop)

        value_label = QLabel(value_str)
        value_label.setStyleSheet("font-size: 9pt; color: #000000; font-weight: bold;")
        value_label.setWordWrap(True)
        value_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        grid_layout.addWidget(param_label, row, col)
        grid_layout.addWidget(value_label, row, col + 1)

    for c in range(num_cols):
        grid_layout.setColumnStretch(c * 2, 0)
        grid_layout.setColumnStretch(c * 2 + 1, 1)
