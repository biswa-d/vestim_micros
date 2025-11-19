from PyQt5.QtWidgets import QLabel, QGridLayout, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt

# Updated: 2025-11-05 - Fixed column-first filling order

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

    def is_parameter_relevant(param_key, model_type, scheduler_type, current_params):
        # Exclude parameters that are redundant or internal
        excluded_params = ['INPUT_SIZE', 'OUTPUT_SIZE', 'NUM_WORKERS', 'BATCH_TRAINING', 
                  'DEVICE_SELECTION', 'REPETITIONS', 'CURRENT_REPETITION',
                  # Hide deprecated time-split inputs; we display the canonical seconds field instead
                  'MAX_TRAIN_HOURS', 'MAX_TRAIN_MINUTES', 'MAX_TRAIN_SECONDS']
        if param_key in excluded_params:
            return False
        
        # Only show LOOKBACK for sequence training
        if param_key == 'LOOKBACK':
            training_method = current_params.get('TRAINING_METHOD', '')
            return str(training_method) == 'Sequence-to-Sequence'
        
        # Hide exploitation parameters when exploitation is disabled (EXPLOIT_REPETITIONS = 0)
        exploit_repetitions = current_params.get('EXPLOIT_REPETITIONS', 0)
        try:
            exploit_reps_val = int(exploit_repetitions)
        except (ValueError, TypeError):
            exploit_reps_val = 0
        
        if param_key in ['EXPLOIT_LR', 'EXPLOIT_EPOCHS', 'EXPLOIT_REPETITIONS'] and exploit_reps_val == 0:
            return False

        always_relevant = {
            'MODEL_TYPE', 'NUM_LEARNABLE_PARAMS',
            'TRAINING_METHOD', 'BATCH_SIZE', 'LOOKBACK',
            'MAX_EPOCHS', 'INITIAL_LR', 'OPTIMIZER_TYPE', 'SCHEDULER_TYPE', 
            'VALID_PATIENCE', 'VALID_FREQUENCY',
            'CURRENT_DEVICE', 'USE_MIXED_PRECISION',
            'MAX_TRAINING_TIME_SECONDS', 'FEATURE_COLUMNS', 'TARGET_COLUMN',
            'INFERENCE_FILTER_TYPE', 'PIN_MEMORY', 'PREFETCH_FACTOR',
            'DROPOUT_PROB', 'WEIGHT_DECAY', 'NORMALIZATION_APPLIED', 'FINAL_LR'
        }

        if param_key in always_relevant:
            return True

        # Conditionally hide inference filter parameters if filter type is None
        inference_filter_type = current_params.get('INFERENCE_FILTER_TYPE')
        if param_key in ['INFERENCE_FILTER_WINDOW_SIZE', 'INFERENCE_FILTER_ALPHA', 'INFERENCE_FILTER_POLYORDER']:
            if inference_filter_type is None or str(inference_filter_type).strip().lower() == 'none':
                return False

        if model_type in ['LSTM', 'GRU', 'LSTM_EMA', 'LSTM_LPF']:
            # Explicitly hide legacy redundancy for RNNs
            if param_key in {'LAYERS', 'HIDDEN_UNITS'}:
                return False
            lstm_gru_params = {'LOOKBACK', 'RNN_LAYER_SIZES'}
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

    filtered_params = {k: v for k, v in params.items() if is_parameter_relevant(k, model_type, scheduler_type, params)}

    # Sequential ordering with key aliases: fill top-to-bottom in a single list.
    # Each entry in preferred_key_groups is a list of aliases; the first present wins.
    preferred_key_groups = [
        ['FEATURE_COLUMNS'],
        ['TARGET_COLUMN'],
        ['NORMALIZATION_APPLIED', 'normalization_applied'],
        ['MODEL_TYPE'],
    ]

    # Model-specific architecture
    if model_type == 'FNN':
        preferred_key_groups += [
            ['HIDDEN_LAYER_SIZES', 'FNN_HIDDEN_LAYERS'],
            ['FNN_ACTIVATION'],
            ['DROPOUT_PROB', 'FNN_DROPOUT_PROB'],
            ['WEIGHT_DECAY']
        ]
    elif model_type in ['LSTM', 'GRU', 'LSTM_EMA', 'LSTM_LPF']:
        preferred_key_groups += [
            ['RNN_LAYER_SIZES'],
            ['DROPOUT_PROB'],
            ['WEIGHT_DECAY']
        ]

    preferred_key_groups += [
        ['NUM_LEARNABLE_PARAMS'],
        ['CURRENT_DEVICE'],
        ['USE_MIXED_PRECISION'],
        ['TRAINING_METHOD'],
    ]

    # Show LOOKBACK only for sequence training
    if str(params.get('TRAINING_METHOD', '')) == 'Sequence-to-Sequence':
        preferred_key_groups.append(['LOOKBACK'])

    preferred_key_groups += [
        ['BATCH_SIZE'],
        ['OPTIMIZER_TYPE'],
        ['INITIAL_LR'],
        ['SCHEDULER_TYPE'],
        ['FINAL_LR'],
    ]

    # Scheduler-specific params
    if scheduler_type == 'StepLR':
        preferred_key_groups += [
            ['LR_DROP_PERIOD', 'LR_PERIOD'],
            ['LR_DROP_FACTOR', 'LR_PARAM']
        ]
    elif scheduler_type == 'ReduceLROnPlateau':
        preferred_key_groups += [
            ['PLATEAU_PATIENCE'],
            ['PLATEAU_FACTOR']
        ]
    elif scheduler_type == 'CosineAnnealingWarmRestarts':
        preferred_key_groups += [
            ['COSINE_T0'],
            ['COSINE_T_MULT'],
            ['COSINE_ETA_MIN']
        ]

    # Exploit + validation/limits
    preferred_key_groups += [
        ['EXPLOIT_LR'],
        ['EXPLOIT_EPOCHS'],
        ['EXPLOIT_REPETITIONS'],
        ['MAX_EPOCHS'],
        ['VALID_FREQUENCY'],
        ['VALID_PATIENCE'],
        ['MAX_TRAINING_TIME_SECONDS'],
        ['INFERENCE_FILTER_TYPE']
    ]

    ordered_items = []
    used_keys = set()

    # Place items according to preferred order, respecting aliases
    for aliases in preferred_key_groups:
        actual_key = next((k for k in aliases if k in filtered_params), None)
        if actual_key is not None and actual_key not in used_keys:
            ordered_items.append((actual_key, filtered_params[actual_key]))
            used_keys.add(actual_key)
    
    for key, value in filtered_params.items():
        if key not in used_keys:
            ordered_items.append((key, value))

    items = ordered_items
    # Use 4 columns (each column has label + value, so 8 grid columns total)
    num_cols = 4
    num_rows = (len(items) + num_cols - 1) // num_cols

    for i, (param, value) in enumerate(items):
        # Fill DOWN columns first (top-to-bottom), then move RIGHT
        col = (i // num_rows) * 2  # Which column pair (0, 2, 4, 6)
        row = i % num_rows          # Which row within that column
        
        # Debug prints removed once layout confirmed

        label_text = gui.param_labels.get(param, param.replace("_", " ").title())
        
        # Friendly formatting with special handling for boolean-like fields
        boolean_like_keys = {"normalization_applied", "PIN_MEMORY", "USE_MIXED_PRECISION"}
        if isinstance(value, bool):
            value_str = "True" if value else "False"
        elif str(param) in boolean_like_keys or str(param).lower() in boolean_like_keys:
            # Treat 1/0 or truthy strings as booleans
            truthy = {True, 1, "1", "true", "True", "YES", "yes"}
            value_str = "True" if value in truthy else "False"
        else:
            # Convert to float for formatting check, handle potential errors
            try:
                float_val = float(value)
                # Clean integer representation for any integer-valued number (including 0.0)
                if float_val.is_integer():
                    value_str = str(int(float_val))
                # Use scientific notation for small non-zero values (learning rates, etc.)
                elif 0 < abs(float_val) <= 0.01:
                    value_str = f"{float_val:.1e}"
                elif abs(float_val) < 1.0:
                    # Format to 1 decimal place for values less than 1
                    value_str = f"{float_val:.1f}"
                else:
                    # Non-integer values >= 1
                    value_str = f"{float_val:.1f}"
            except (ValueError, TypeError):
                value_str = str(value)
        
        # Consistent formatting for architecture parameters
        is_arch_param = param in ['HIDDEN_LAYER_SIZES', 'FNN_HIDDEN_LAYERS', 'RNN_LAYER_SIZES']
        if is_arch_param:
            if isinstance(value, list):
                # Handles cases where it's already a list (e.g., from task_info.json)
                value_str = f"[{', '.join(map(str, value))}]"
            elif isinstance(value, str):
                # Handles string representations (e.g., from user input in setup GUI)
                # This makes "128, 64" or "32,16,10" display as "[128, 64]" etc.
                value_str = f"[{value}]"

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
