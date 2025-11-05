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

def _get_value_or_zero(params, key):
    """Helper to safely get numeric value or return 0"""
    try:
        return float(params.get(key, 0))
    except (ValueError, TypeError):
        return 0

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
    
    # Get critical values for conditional display
    max_train_time = _get_value_or_zero(params, 'MAX_TRAINING_TIME_SECONDS')
    exploit_reps = _get_value_or_zero(params, 'EXPLOIT_REPETITIONS')
    inference_filter = params.get('INFERENCE_FILTER_TYPE')
    
    # Logical order as requested by user:
    # 1. Feature & Target columns
    # 2. Model & Layers
    # 3. Mixed Precision
    # 4. Training Method
    # 5. For LSTM: Sequence Length
    # 6. Batch Size
    # 7. Max Epochs
    # 8. Validation Freq
    # 9. Validation Patience
    # 10. Optimizer
    # 11. Weight Decay
    # 12. LR Scheduler & relevant params only
    # 13. Exploit params (only if exploit_reps > 0)
    # 14. Training time (only if > 0)
    # 15. Inference filter params (only if filter != None)
    
    ordered_keys = []
    
    # 1. Data columns
    ordered_keys.extend(['FEATURE_COLUMNS', 'TARGET_COLUMN'])
    
    # 2. Model architecture
    ordered_keys.append('MODEL_TYPE')
    if model_type in ['LSTM', 'GRU', 'LSTM_EMA', 'LSTM_LPF']:
        ordered_keys.append('RNN_LAYER_SIZES')
    elif model_type == 'FNN':
        ordered_keys.append('HIDDEN_LAYER_SIZES')
    # 2b. Place # Params and Current Device right after layer sizes for visual consistency
    ordered_keys.append('NUM_LEARNABLE_PARAMS')
    ordered_keys.append('CURRENT_DEVICE')
    
    # 3. Mixed Precision
    ordered_keys.append('USE_MIXED_PRECISION')
    
    # 4. Training Method
    ordered_keys.append('TRAINING_METHOD')
    
    # 5. For LSTM/GRU: Sequence Length (renamed from LOOKBACK)
    if model_type in ['LSTM', 'GRU', 'LSTM_EMA', 'LSTM_LPF']:
        ordered_keys.append('LOOKBACK')
    
    # 6. Batch Size
    ordered_keys.append('BATCH_SIZE')
    
    # 7. Max Epochs
    ordered_keys.append('MAX_EPOCHS')
    
    # 8. Validation Frequency
    ordered_keys.append('VALID_FREQUENCY')
    
    # 9. Validation Patience
    ordered_keys.append('VALID_PATIENCE')
    
    # 10. Optimizer
    ordered_keys.append('OPTIMIZER_TYPE')
    
    # 11. Weight Decay
    ordered_keys.append('WEIGHT_DECAY')
    
    # 12. LR Scheduler and relevant params
    ordered_keys.append('INITIAL_LR')
    ordered_keys.append('SCHEDULER_TYPE')
    
    if scheduler_type == 'StepLR':
        ordered_keys.extend(['LR_DROP_PERIOD', 'LR_DROP_FACTOR'])
    elif scheduler_type == 'ReduceLROnPlateau':
        ordered_keys.extend(['PLATEAU_PATIENCE', 'PLATEAU_FACTOR'])
    elif scheduler_type == 'CosineAnnealingWarmRestarts':
        ordered_keys.extend(['COSINE_T0', 'COSINE_T_MULT', 'COSINE_ETA_MIN'])
    
    # 13. Exploit params (only if EXPLOIT_REPETITIONS > 0)
    if exploit_reps > 0:
        ordered_keys.extend(['EXPLOIT_LR', 'EXPLOIT_EPOCHS', 'EXPLOIT_REPETITIONS'])
    
    # 14. Training time (only if > 0)
    if max_train_time > 0:
        ordered_keys.append('MAX_TRAINING_TIME_SECONDS')
    
    # 15. Inference filter params (only if filter is not None)
    if inference_filter and str(inference_filter).strip().lower() != 'none':
        ordered_keys.append('INFERENCE_FILTER_TYPE')
        # Add specific filter params based on filter type
        if 'INFERENCE_FILTER_WINDOW_SIZE' in params:
            ordered_keys.append('INFERENCE_FILTER_WINDOW_SIZE')
        if 'INFERENCE_FILTER_ALPHA' in params:
            ordered_keys.append('INFERENCE_FILTER_ALPHA')
        if 'INFERENCE_FILTER_POLYORDER' in params:
            ordered_keys.append('INFERENCE_FILTER_POLYORDER')
    
    # Additional useful params (device already placed above next to model arch)
    ordered_keys.extend(['PIN_MEMORY', 'DROPOUT_PROB'])
    
    # Build filtered items list in order
    items = []
    used_keys = set()
    
    for key in ordered_keys:
        if key in params:
            items.append((key, params[key]))
            used_keys.add(key)
    
    # Add any remaining params not explicitly ordered (fallback)
    excluded_always = {'INPUT_SIZE', 'OUTPUT_SIZE', 'NUM_WORKERS', 'BATCH_TRAINING', 
                      'DEVICE_SELECTION', 'REPETITIONS', 'CURRENT_REPETITION', 
                      'LAYERS', 'HIDDEN_UNITS', 'LR_PERIOD', 'LR_PARAM'}

    def _include_in_fallback(k: str) -> bool:
        # Hide inference filter details when filter is None
        if k in {'INFERENCE_FILTER_WINDOW_SIZE', 'INFERENCE_FILTER_ALPHA', 'INFERENCE_FILTER_POLYORDER'}:
            if not inference_filter or str(inference_filter).strip().lower() == 'none':
                return False
        # Hide exploit params when repetitions are zero
        if k in {'EXPLOIT_LR', 'EXPLOIT_EPOCHS', 'EXPLOIT_REPETITIONS'} and exploit_reps <= 0:
            return False
        # Hide max training time when zero
        if k == 'MAX_TRAINING_TIME_SECONDS' and max_train_time <= 0:
            return False
        return True

    for key, value in params.items():
        if key not in used_keys and key not in excluded_always and _include_in_fallback(key):
            items.append((key, value))
    # Revert to 4 columns as requested
    num_cols = 4
    num_rows = (len(items) + num_cols - 1) // num_cols

    for i, (param, value) in enumerate(items):
        row = i % num_rows
        col = (i // num_rows) * 2

        label_text = gui.param_labels.get(param, param.replace("_", " ").title())

        # Friendly formatting with special handling for integers, booleans, and zero-as-None fields
        boolean_like_keys = {"normalization_applied", "PIN_MEMORY", "USE_MIXED_PRECISION"}
        integer_keys = {
            'BATCH_SIZE','MAX_EPOCHS','VALID_FREQUENCY','VALID_PATIENCE','PLATEAU_PATIENCE',
            'COSINE_T0','LOOKBACK','NUM_LEARNABLE_PARAMS','EXPLOIT_EPOCHS','INFERENCE_FILTER_WINDOW_SIZE',
            'INFERENCE_FILTER_POLYORDER'
        }
        zero_as_none_keys = {'EXPLOIT_REPETITIONS'}

        if isinstance(value, bool):
            value_str = "True" if value else "False"
        elif str(param) in boolean_like_keys or str(param).lower() in boolean_like_keys:
            truthy = {True, 1, "1", "true", "True", "YES", "yes"}
            value_str = "True" if value in truthy else "False"
        elif param in zero_as_none_keys:
            try:
                v = int(float(value))
                value_str = "None" if v == 0 else str(v)
            except (ValueError, TypeError):
                value_str = str(value)
        elif param in integer_keys:
            try:
                value_str = str(int(float(value)))
            except (ValueError, TypeError):
                value_str = str(value)
        else:
            try:
                float_val = float(value)
                if 0 < abs(float_val) <= 0.01:
                    value_str = f"{float_val:.1e}"
                elif abs(float_val) < 1.0 and float_val != int(float_val):
                    value_str = f"{float_val:.1f}"
                else:
                    if float_val == int(float_val):
                        value_str = str(int(float_val))
                    else:
                        value_str = f"{float_val:.1f}"
            except (ValueError, TypeError):
                value_str = str(value)
        
        # Truncate very long parameter values to prevent GUI distortion
        MAX_DISPLAY_LENGTH = 60
        if len(value_str) > MAX_DISPLAY_LENGTH:
            value_str = value_str[:MAX_DISPLAY_LENGTH-3] + "..."

        param_label = QLabel(f"{label_text}:")
        param_label.setStyleSheet("font-size: 9pt; color: #333;")
        param_label.setAlignment(Qt.AlignRight | Qt.AlignTop)

        value_label = QLabel(value_str)
        value_label.setStyleSheet("font-size: 9pt; color: #000000; font-weight: bold;")
        value_label.setWordWrap(True)
        value_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        grid_layout.addWidget(param_label, row, col)
        grid_layout.addWidget(value_label, row, col + 1)

    # Custom column stretch for 4 columns.
    # The grid has 8 columns total (4 label/value pairs).
    # We give the first value column (index 1) a higher stretch factor
    # to make it wider for long text like the feature list.
    grid_layout.setColumnStretch(0, 0) # Col 1 Label
    grid_layout.setColumnStretch(1, 2) # Col 1 Value (weight 2)
    grid_layout.setColumnStretch(2, 0) # Col 2 Label
    grid_layout.setColumnStretch(3, 1) # Col 2 Value (weight 1)
    grid_layout.setColumnStretch(4, 0) # Col 3 Label
    grid_layout.setColumnStretch(5, 1) # Col 3 Value (weight 1)
    grid_layout.setColumnStretch(6, 0) # Col 4 Label
    grid_layout.setColumnStretch(7, 1) # Col 4 Value (weight 1)
