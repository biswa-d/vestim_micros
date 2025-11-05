from PyQt5.QtWidgets import QLabel, QGridLayout, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox
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

def _create_section_box(title, items, gui, color="#3498db"):
    """Create a styled QGroupBox for a parameter section"""
    group = QGroupBox(title)
    group.setStyleSheet(f"""
        QGroupBox {{
            font-weight: bold;
            font-size: 10pt;
            color: {color};
            border: 2px solid {color};
            border-radius: 5px;
            margin-top: 12px;
            padding: 15px 10px 10px 10px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 5px;
            background-color: white;
        }}
    """)
    
    layout = QGridLayout()
    layout.setHorizontalSpacing(10)
    layout.setVerticalSpacing(8)
    layout.setContentsMargins(5, 10, 5, 5)
    
    # Arrange items in 2 columns within each box
    for i, (param, value) in enumerate(items):
        row = i // 2
        col = (i % 2) * 2
        
        label_text = gui.param_labels.get(param, param.replace("_", " ").title())
        
        lbl = QLabel(f"{label_text}:")
        lbl.setStyleSheet("font-size: 9pt; color: #555; font-weight: normal;")
        lbl.setAlignment(Qt.AlignRight | Qt.AlignTop)
        
        val = QLabel(str(value))
        val.setStyleSheet("font-size: 9pt; font-weight: bold; color: #000;")
        val.setWordWrap(True)
        val.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        
        layout.addWidget(lbl, row, col)
        layout.addWidget(val, row, col + 1)
    
    # Set column stretches to make value columns wider
    layout.setColumnStretch(1, 2)  # First value column
    layout.setColumnStretch(3, 2)  # Second value column
    
    group.setLayout(layout)
    return group

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
    
    # Build grouped sections with formatted values
    # Better logical grouping as per user feedback
    data_model_items = []      # Data + Model architecture together
    training_config_items = []  # Training method, seq len, batch, epochs, validation
    optimizer_items = []        # Optimizer, LR scheduler and its params
    refinement_items = []       # Exploit params (only if active)
    inference_items = []        # Inference filter params (only if active)
    
    def _format_value(param, value):
        """Format parameter value for display"""
        boolean_like_keys = {"normalization_applied", "PIN_MEMORY", "USE_MIXED_PRECISION"}
        integer_keys = {
            'BATCH_SIZE','MAX_EPOCHS','VALID_FREQUENCY','VALID_PATIENCE','PLATEAU_PATIENCE',
            'COSINE_T0','LOOKBACK','NUM_LEARNABLE_PARAMS','EXPLOIT_EPOCHS','INFERENCE_FILTER_WINDOW_SIZE',
            'INFERENCE_FILTER_POLYORDER'
        }
        zero_as_none_keys = {'EXPLOIT_REPETITIONS'}

        if isinstance(value, bool):
            return "True" if value else "False"
        elif str(param) in boolean_like_keys or str(param).lower() in boolean_like_keys:
            truthy = {True, 1, "1", "true", "True", "YES", "yes"}
            return "True" if value in truthy else "False"
        elif param in zero_as_none_keys:
            try:
                v = int(float(value))
                return "None" if v == 0 else str(v)
            except (ValueError, TypeError):
                return str(value)
        elif param in integer_keys:
            try:
                return str(int(float(value)))
            except (ValueError, TypeError):
                return str(value)
        else:
            try:
                float_val = float(value)
                if 0 < abs(float_val) <= 0.01:
                    return f"{float_val:.1e}"
                elif abs(float_val) < 1.0 and float_val != int(float_val):
                    return f"{float_val:.1f}"
                else:
                    if float_val == int(float_val):
                        return str(int(float_val))
                    else:
                        return f"{float_val:.1f}"
            except (ValueError, TypeError):
                # Truncate very long strings
                value_str = str(value)
                if len(value_str) > 60:
                    return value_str[:57] + "..."
                return value_str
    
    # Populate sections with better logical grouping
    for key in ordered_keys:
        if key not in params:
            continue
        formatted_value = _format_value(key, params[key])
        
        # DATA + MODEL section (Features, Target, Model Type, Layers, # Params, Device, Mixed Precision)
        if key in ['FEATURE_COLUMNS', 'TARGET_COLUMN', 'MODEL_TYPE', 'RNN_LAYER_SIZES', 'HIDDEN_LAYER_SIZES', 
                   'NUM_LEARNABLE_PARAMS', 'CURRENT_DEVICE', 'USE_MIXED_PRECISION', 'DROPOUT_PROB']:
            data_model_items.append((key, formatted_value))
        
        # TRAINING CONFIGURATION section (Method, Seq Len, Batch, Epochs, Val Freq, Val Patience)
        elif key in ['TRAINING_METHOD', 'LOOKBACK', 'BATCH_SIZE', 'MAX_EPOCHS', 'VALID_FREQUENCY', 'VALID_PATIENCE']:
            training_config_items.append((key, formatted_value))
        
        # OPTIMIZER SETTINGS section (Optimizer, Weight Decay, Initial LR, LR Scheduler and all scheduler params)
        elif key in ['OPTIMIZER_TYPE', 'WEIGHT_DECAY', 'INITIAL_LR', 'SCHEDULER_TYPE', 
                     'LR_DROP_PERIOD', 'LR_DROP_FACTOR', 'PLATEAU_PATIENCE', 'PLATEAU_FACTOR',
                     'COSINE_T0', 'COSINE_T_MULT', 'COSINE_ETA_MIN']:
            optimizer_items.append((key, formatted_value))
        
        # TRAINING REFINEMENT section (Exploit params - only if reps > 0)
        elif key in ['EXPLOIT_LR', 'EXPLOIT_EPOCHS', 'EXPLOIT_REPETITIONS'] and exploit_reps > 0:
            refinement_items.append((key, formatted_value))
        
        # INFERENCE SETTINGS section (Filter params - only if filter != None)
        elif key.startswith('INFERENCE_FILTER') and inference_filter and str(inference_filter).strip().lower() != 'none':
            inference_items.append((key, formatted_value))
        
        # Pin Memory and other misc go to training config
        elif key in ['PIN_MEMORY']:
            training_config_items.append((key, formatted_value))
        
        # Max training time (only if > 0) goes to training config
        elif key == 'MAX_TRAINING_TIME_SECONDS' and max_train_time > 0:
            training_config_items.append((key, formatted_value))
    
    # Add any remaining params to appropriate fallback section
    excluded_always = {'INPUT_SIZE', 'OUTPUT_SIZE', 'NUM_WORKERS', 'BATCH_TRAINING', 
                      'DEVICE_SELECTION', 'REPETITIONS', 'CURRENT_REPETITION', 
                      'LAYERS', 'HIDDEN_UNITS', 'LR_PERIOD', 'LR_PARAM'}
    
    used_keys = set(key for key in ordered_keys if key in params)
    for key, value in params.items():
        if key in used_keys or key in excluded_always:
            continue
        # Apply fallback filtering
        if key in {'INFERENCE_FILTER_WINDOW_SIZE', 'INFERENCE_FILTER_ALPHA', 'INFERENCE_FILTER_POLYORDER'}:
            if not inference_filter or str(inference_filter).strip().lower() == 'none':
                continue
        if key in {'EXPLOIT_LR', 'EXPLOIT_EPOCHS', 'EXPLOIT_REPETITIONS'} and exploit_reps <= 0:
            continue
        if key == 'MAX_TRAINING_TIME_SECONDS' and max_train_time <= 0:
            continue
        
        formatted_value = _format_value(key, value)
        # Add to training config as fallback
        training_config_items.append((key, formatted_value))

    # Create grouped box layout with better organization
    # Row 1: Data+Model, Training Config, Optimizer Settings
    row1_layout = QHBoxLayout()
    row1_layout.setSpacing(10)
    
    if data_model_items:
        row1_layout.addWidget(_create_section_box("DATA & MODEL", data_model_items, gui, "#3498db"))
    
    if training_config_items:
        row1_layout.addWidget(_create_section_box("TRAINING CONFIGURATION", training_config_items, gui, "#e74c3c"))
    
    if optimizer_items:
        row1_layout.addWidget(_create_section_box("OPTIMIZER SETTINGS", optimizer_items, gui, "#f39c12"))
    
    # Row 2: Training Refinement (Exploit) and Inference Settings (only if active)
    row2_layout = QHBoxLayout()
    row2_layout.setSpacing(10)
    
    if refinement_items:
        refinement_box = _create_section_box("TRAINING REFINEMENT", refinement_items, gui, "#9b59b6")
        row2_layout.addWidget(refinement_box)
    
    if inference_items:
        inference_box = _create_section_box("INFERENCE SETTINGS", inference_items, gui, "#1abc9c")
        row2_layout.addWidget(inference_box)
    
    # Add rows to main container
    frame_layout.addLayout(row1_layout)
    if refinement_items or inference_items:  # Only add row 2 if there's content
        frame_layout.addLayout(row2_layout)
