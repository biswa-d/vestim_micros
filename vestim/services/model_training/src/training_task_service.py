# filepath: c:\Users\dehuryb\OneDrive - McMaster University\Models\ML_LiB_Models\vestim_micros\vestim\services\model_training\src\training_task_service.py.fixed
import torch, numpy as np
import torch.nn as nn
import torch.optim as optim
import json, csv, sqlite3, os
import time


class TrainingTaskService:
    def __init__(self, device=None):
        self.criterion = nn.MSELoss()  # Assuming you're using Mean Squared Error Loss for regression tasks
        if device is not None:
            self.device = device
            print(f"TrainingTaskService: Using specified device: {device} (type: {type(device)})")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.device = device
            print(f"TrainingTaskService: Auto-detected device: {device}")
        
        # Setup performance optimizations
        self._setup_performance_optimizations()
    
    def _setup_performance_optimizations(self):
        """Setup PyTorch performance optimizations that are safe and stable."""
        # Enable cuDNN benchmark mode for faster convolutions/RNN operations
        # This auto-tunes algorithms for your specific hardware/input size
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            print("Enabled cuDNN benchmark mode for optimized CUDA operations")

            # Enable TensorFloat-32 (TF32) for NVIDIA Ampere GPUs (RTX 30xx, RTX 40xx, RTX 50xx, A100)
            # TF32 provides ~10-20% speedup with minimal accuracy impact
            # Uses 19 bits precision instead of full FP32, but maintains same range
            try:
                torch.set_float32_matmul_precision('high')  # or 'highest' for more precision
                print("Enabled TensorFloat-32 (TF32) for faster matrix operations on Ampere+ GPUs")
            except AttributeError:
                # PyTorch < 1.12 doesn't have this feature
                print("TF32 not available (requires PyTorch >= 1.12)")
        
        # Optimize CPU threading for PyTorch operations
        # Prevents oversubscription when using multiple DataLoader workers
        import os
        cpu_count = os.cpu_count() or 4
        # Reserve some threads for DataLoader workers
        torch_threads = max(1, cpu_count // 2)
        torch.set_num_threads(torch_threads)
        # Set environment variables for BLAS libraries
        os.environ['OMP_NUM_THREADS'] = str(torch_threads)
        os.environ['MKL_NUM_THREADS'] = str(torch_threads)
        print(f"Optimized CPU threading: {torch_threads} threads for PyTorch operations")
    
    def log_to_csv(self, task, epoch, batch_idx, batch_time, phase):
        """Log batch timing data to a CSV file."""
        csv_log_file = task['csv_log_file']  # Fetch the CSV log file path from the task
        fieldnames = ['Epoch', 'Batch', 'Batch Time', 'Phase']
        file_exists = os.path.isfile(csv_log_file)

        with open(csv_log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()  # Write header only once
            writer.writerow({
                'Epoch': epoch,
                'Batch': batch_idx,
                'Batch Time': batch_time,
                'Phase': phase
            })

    def log_to_sqlite(self, task, epoch, batch_idx, batch_time, phase, device):
        """Log batch timing data to a SQLite database."""
        sqlite_db_file = task['db_log_file']  # Fetch the SQLite DB file path from the task
        conn = sqlite3.connect(sqlite_db_file)
        cursor = conn.cursor()

        # Insert batch-level data into batch_logs table
        cursor.execute('''INSERT INTO batch_logs (task_id, epoch, batch_idx, batch_time, phase, learning_rate, num_learnable_params, batch_size, lookback, device)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?,?,?)''',
                    (task['task_id'], epoch, batch_idx, batch_time, phase, 
                        task['hyperparams']['INITIAL_LR'], task['hyperparams']['NUM_LEARNABLE_PARAMS'],
                        task['hyperparams']['BATCH_SIZE'], task['hyperparams']['LOOKBACK'], device))

        conn.commit()
        conn.close()

    def _as_bool(self, value):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ['true', '1', 'yes']

    def _is_hybrid_loss_enabled(self, task):
        hyperparams = task.get('hyperparams', {}) if isinstance(task, dict) else {}
        enabled = self._as_bool(hyperparams.get('PHYSICS_DTDT_CONSTRAINT_ENABLED', False))
        if not enabled:
            return False
        try:
            first_order_weight = float(hyperparams.get('PHYSICS_DTDT_LOSS_WEIGHT', 0.0))
        except Exception:
            first_order_weight = 0.0
        try:
            second_order_weight = float(hyperparams.get('PHYSICS_D2YDT2_LOSS_WEIGHT', 0.0))
        except Exception:
            second_order_weight = 0.0
        return (first_order_weight > 0) or (second_order_weight > 0)

    def _is_smoothness_enabled(self, task):
        hyperparams = task.get('hyperparams', {}) if isinstance(task, dict) else {}
        try:
            weight = float(hyperparams.get('PHYSICS_SMOOTHNESS_LOSS_WEIGHT', 0.0))
        except Exception:
            weight = 0.0
        return weight > 0

    def _prediction_smoothness_penalty(self, y_pred, task):
        hyperparams = task.get('hyperparams', {}) if isinstance(task, dict) else {}
        try:
            smooth_weight = float(hyperparams.get('PHYSICS_SMOOTHNESS_LOSS_WEIGHT', 0.0))
        except Exception:
            smooth_weight = 0.0

        if smooth_weight <= 0:
            return torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)

        pred_seq = y_pred
        if pred_seq.ndim > 1 and pred_seq.shape[-1] == 1:
            pred_seq = pred_seq.squeeze(-1)

        if pred_seq.ndim == 1:
            if pred_seq.shape[0] < 2:
                return torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)
            diffs = torch.diff(pred_seq, dim=0)
        else:
            if pred_seq.shape[1] < 2:
                return torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)
            diffs = torch.diff(pred_seq, dim=1)

        return smooth_weight * torch.mean(diffs * diffs)

    def _is_temperature_target(self, task):
        if not isinstance(task, dict):
            return False
        target_col = str(task.get('data_loader_params', {}).get('target_column', '')).lower()
        if not target_col:
            target_col = str(task.get('hyperparams', {}).get('TARGET_COLUMN', '')).lower()
        return ('temp' in target_col) or ('temperature' in target_col)

    def _get_initial_temp_feature_index(self, task):
        if not isinstance(task, dict):
            return None
        if '_initial_temp_feature_index_cache' in task:
            return task.get('_initial_temp_feature_index_cache')

        feature_columns = task.get('data_loader_params', {}).get('feature_columns', [])
        if isinstance(feature_columns, str):
            feature_columns = [c.strip() for c in feature_columns.split(',') if c.strip()]
        if not isinstance(feature_columns, list):
            task['_initial_temp_feature_index_cache'] = None
            return None

        idx = None
        for i, col in enumerate(feature_columns):
            name = str(col).strip().lower()
            if 'temp' in name and 'initial' in name:
                idx = i
                break

        task['_initial_temp_feature_index_cache'] = idx
        return idx

    def _apply_batch_initial_temperature_anchor(self, X_batch, y_batch, task):
        if not self._is_hybrid_loss_enabled(task):
            return X_batch
        if not self._is_temperature_target(task):
            return X_batch

        feature_idx = self._get_initial_temp_feature_index(task)
        if feature_idx is None:
            return X_batch

        if y_batch.numel() == 0:
            return X_batch

        if y_batch.ndim == 0:
            anchor_value = y_batch
        elif y_batch.ndim == 1:
            anchor_value = y_batch[0]
        else:
            anchor_value = y_batch[0, 0]

        try:
            if X_batch.ndim == 2 and feature_idx < X_batch.shape[1]:
                X_batch[:, feature_idx] = anchor_value
            elif X_batch.ndim == 3 and feature_idx < X_batch.shape[2]:
                X_batch[:, :, feature_idx] = anchor_value
        except Exception:
            return X_batch

        if isinstance(task, dict) and not task.get('_batch_anchor_logged_once', False):
            task['_batch_anchor_logged_once'] = True
            msg = (
                f"Hybrid anchor enabled: using first target in each batch as dynamic initial-temp feature "
                f"(feature index {feature_idx})."
            )
            log_callback = task.get('log_callback')
            if log_callback:
                log_callback(msg)
            print(msg)

        return X_batch

    def _log_hybrid_loss_configuration_once(self, task):
        if not isinstance(task, dict):
            return
        if task.get('_hybrid_loss_logged_once', False):
            return
        task['_hybrid_loss_logged_once'] = True

        hyperparams = task.get('hyperparams', {})
        enabled = self._is_hybrid_loss_enabled(task)
        mode = str(hyperparams.get('PHYSICS_DTDT_MAX_ABS_MODE', 'absolute')).strip().lower()
        message = (
            f"Hybrid loss mode: {'ENABLED' if enabled else 'DISABLED'} | "
            f"lambda_d1={hyperparams.get('PHYSICS_DTDT_LOSS_WEIGHT', 0.0)} | "
            f"mu_d2={hyperparams.get('PHYSICS_D2YDT2_LOSS_WEIGHT', 0.0)} | "
            f"smooth={hyperparams.get('PHYSICS_SMOOTHNESS_LOSS_WEIGHT', 0.0)} | "
            f"dt={hyperparams.get('PHYSICS_DTDT_DT_SECONDS', 1.0)}s | "
            f"bound_mode={mode}"
        )
        if mode in ('fraction_of_true_batch_max', 'fraction'):
            message += f" | fraction={hyperparams.get('PHYSICS_DTDT_MAX_ABS_FRACTION', 1.0)}"
        else:
            message += f" | max_abs={hyperparams.get('PHYSICS_DTDT_MAX_ABS', 0.0)}"

        log_callback = task.get('log_callback')
        if log_callback:
            log_callback(message)
        print(message)

    def _physics_dtdt_penalty(self, y_pred, y_true, task):
        hyperparams = task.get('hyperparams', {}) if isinstance(task, dict) else {}
        enabled = self._as_bool(hyperparams.get('PHYSICS_DTDT_CONSTRAINT_ENABLED', False))
        if not enabled:
            return torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)

        target_column = str(hyperparams.get('TARGET_COLUMN', '')).lower()
        target_only_temp = self._as_bool(hyperparams.get('PHYSICS_DTDT_TARGET_ONLY_TEMPERATURE', False))
        if target_only_temp and ('temp' not in target_column and 'temperature' not in target_column):
            return torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)

        first_order_weight = float(hyperparams.get('PHYSICS_DTDT_LOSS_WEIGHT', 0.0))
        second_order_weight = float(hyperparams.get('PHYSICS_D2YDT2_LOSS_WEIGHT', 0.0))
        if first_order_weight <= 0 and second_order_weight <= 0:
            return torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)

        dt_seconds = float(hyperparams.get('PHYSICS_DTDT_DT_SECONDS', 1.0))
        if dt_seconds <= 0:
            return torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)

        pred_seq = y_pred
        true_seq = y_true

        if pred_seq.ndim > 1 and pred_seq.shape[-1] == 1:
            pred_seq = pred_seq.squeeze(-1)
        if true_seq.ndim > 1 and true_seq.shape[-1] == 1:
            true_seq = true_seq.squeeze(-1)

        derivative_dim = None
        if pred_seq.ndim == 1 and true_seq.ndim == 1:
            if pred_seq.shape[0] < 2 or true_seq.shape[0] < 2:
                return torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)
            derivative_dim = 0
            pred_dtdt = torch.diff(pred_seq, dim=0) / dt_seconds
            true_dtdt = torch.diff(true_seq, dim=0) / dt_seconds
        else:
            if pred_seq.ndim < 2 or true_seq.ndim < 2:
                return torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)
            if pred_seq.shape[1] < 2 or true_seq.shape[1] < 2:
                return torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)
            derivative_dim = 1
            pred_dtdt = torch.diff(pred_seq, dim=1) / dt_seconds
            true_dtdt = torch.diff(true_seq, dim=1) / dt_seconds

        total_physics_penalty = torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)
        derivative_match_loss = self.criterion(pred_dtdt, true_dtdt)

        max_abs_mode = str(hyperparams.get('PHYSICS_DTDT_MAX_ABS_MODE', 'absolute')).strip().lower()
        max_abs_dtdt_value = float(hyperparams.get('PHYSICS_DTDT_MAX_ABS', 0.0))
        max_abs_fraction = float(hyperparams.get('PHYSICS_DTDT_MAX_ABS_FRACTION', 1.0))
        max_abs_weight = float(hyperparams.get('PHYSICS_DTDT_MAX_ABS_WEIGHT', 1.0))

        max_abs_dtdt = None
        if max_abs_mode in ('fraction_of_true_batch_max', 'fraction'):
            fraction = min(max(max_abs_fraction, 0.0), 1.0)
            if fraction > 0:
                true_batch_max_abs = torch.amax(torch.abs(true_dtdt))
                max_abs_dtdt = fraction * true_batch_max_abs
        elif max_abs_dtdt_value > 0:
            max_abs_dtdt = torch.tensor(max_abs_dtdt_value, device=pred_dtdt.device, dtype=pred_dtdt.dtype)

        if max_abs_dtdt is not None and first_order_weight > 0:
            excess = torch.relu(torch.abs(pred_dtdt) - max_abs_dtdt)
            slope_bound_loss = torch.mean(excess * excess)
            derivative_match_loss = derivative_match_loss + (max_abs_weight * slope_bound_loss)

        if first_order_weight > 0:
            total_physics_penalty = total_physics_penalty + (first_order_weight * derivative_match_loss)

        if second_order_weight > 0 and derivative_dim is not None:
            if pred_dtdt.shape[derivative_dim] >= 2 and true_dtdt.shape[derivative_dim] >= 2:
                pred_d2ydt2 = torch.diff(pred_dtdt, dim=derivative_dim) / dt_seconds
                true_d2ydt2 = torch.diff(true_dtdt, dim=derivative_dim) / dt_seconds
                curvature_match_loss = self.criterion(pred_d2ydt2, true_d2ydt2)
                total_physics_penalty = total_physics_penalty + (second_order_weight * curvature_match_loss)

        return total_physics_penalty
        
    def train_epoch(self, model, model_type, train_loader, optimizer, h_s_initial, h_c_initial, epoch, device, stop_requested, task, verbose=True):
        """Train the model for a single epoch, adapting to model type."""
        model.train()
        total_train_loss = []
        all_train_y_pred_normalized = [] # To store all predictions from the epoch
        all_train_y_true_normalized = [] # To store all true values from the epoch
        batch_times = []
        log_freq = task.get('log_frequency', 100)
        hybrid_loss_enabled = self._is_hybrid_loss_enabled(task)
        self._log_hybrid_loss_configuration_once(task)
        
        # Check if mixed precision training is enabled
        use_mixed_precision = task['hyperparams'].get('USE_MIXED_PRECISION', False) and device.type == 'cuda'
        # FIXED: Use torch.cuda.amp.GradScaler instead of torch.amp.GradScaler
        scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
        
        if use_mixed_precision:
            print(f"Using mixed precision training (AMP) for epoch {epoch}")
        
        # PERFORMANCE OPTIMIZATION: Pre-allocate hidden state tensors to avoid repeated allocations
        # These will be reset to zeros each batch (reference code behavior)
        max_batch_size = train_loader.batch_size
        if model_type in ["LSTM", "LSTM_EMA", "LSTM_LPF"]:
            h_s_buffer = torch.zeros(model.num_layers, max_batch_size, model.hidden_units, device=device)
            h_c_buffer = torch.zeros(model.num_layers, max_batch_size, model.hidden_units, device=device)
        elif model_type == "GRU":
            h_s_buffer = torch.zeros(model.num_layers, max_batch_size, model.hidden_units, device=device)
        
        # Reference code approach: hidden states will be reset to zeros at START of each batch
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            # Check stop signal every 10 batches for faster response (~every 1-2 seconds)
            if batch_idx % 10 == 0 and stop_requested:
                print(f"Stop requested during training at batch {batch_idx}")
                break
                
            # RESET hidden states to zeros for EVERY batch (reference code behavior)
            h_s, h_c = None, None
            z = None  # Initialize filter state for LPF models

            start_batch_time = time.time()
            # Use the 'device' argument passed to the method, not self.device
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            X_batch = self._apply_batch_initial_temperature_anchor(X_batch, y_batch, task)
            
            optimizer.zero_grad()

            # Use AMP autocast context manager when mixed precision is enabled
            if use_mixed_precision:
                with torch.amp.autocast('cuda'):
                    # Forward pass with mixed precision
                    if model_type == "LSTM_LPF":
                        # Reset pre-allocated tensors to zeros (faster than reallocating)
                        actual_batch_size = X_batch.size(0)
                        h_s_buffer[:, :actual_batch_size, :].zero_()
                        h_c_buffer[:, :actual_batch_size, :].zero_()
                        y_pred, (h_s, h_c), z = model(X_batch, h_s_buffer[:, :actual_batch_size, :], h_c_buffer[:, :actual_batch_size, :], z)
                    if model_type in ["LSTM", "LSTM_EMA"]:
                        # Reset pre-allocated tensors to zeros (faster than reallocating)
                        actual_batch_size = X_batch.size(0)
                        h_s_buffer[:, :actual_batch_size, :].zero_()
                        h_c_buffer[:, :actual_batch_size, :].zero_()
                        y_pred, (h_s, h_c) = model(X_batch, h_s_buffer[:, :actual_batch_size, :], h_c_buffer[:, :actual_batch_size, :])
                    elif model_type == "GRU":
                        # Reset pre-allocated tensor to zeros (faster than reallocating)
                        actual_batch_size = X_batch.size(0)
                        h_s_buffer[:, :actual_batch_size, :].zero_()
                        y_pred, h_s = model(X_batch, h_s_buffer[:, :actual_batch_size, :])
                    elif model_type == "FNN":
                        y_pred = model(X_batch)
                    else:
                        raise ValueError(f"Unsupported model_type in train_epoch: {model_type}")

                    # Ensure y_pred and y_batch have compatible shapes for loss calculation
                    if y_pred.ndim > y_batch.ndim and y_pred.shape[-1] == 1 and y_batch.ndim == 1:
                        y_pred = y_pred.squeeze(-1)
                    elif y_pred.ndim > y_batch.ndim and y_pred.shape[0] == y_batch.shape[0] and y_pred.shape[-1] == y_batch.shape[-1]:
                        pass # Assume model output is appropriate

                    if y_batch.ndim == 1 and y_pred.ndim == 2 and y_pred.shape[1] == 1:
                        y_batch = y_batch.unsqueeze(1)

                    base_loss = self.criterion(y_pred, y_batch)
                    physics_penalty = self._physics_dtdt_penalty(y_pred, y_batch, task)
                    loss = base_loss + physics_penalty
                
                # Check for invalid loss before backpropagation
                if not torch.isfinite(loss):
                    print(f"WARNING: Epoch {epoch}, Batch {batch_idx}: Invalid loss detected (NaN/Inf)")
                    # PERFORMANCE: Removed .item()/.min()/.max() calls - they sync GPU→CPU and are very slow
                    optimizer.zero_grad()
                    if use_mixed_precision:
                        scaler.update()
                    # Detach hidden states to break computation graph
                    if h_s is not None:
                        h_s = h_s.detach()
                    if h_c is not None:
                        h_c = h_c.detach()
                    continue
                
                # Mixed precision backward and optimizer step
                scaler.scale(loss).backward()
                
                # No gradient clipping (reference code: Junran Chen)
                
                scaler.step(optimizer)
                scaler.update()
                # No need to detach - states are reset to zeros every batch (reference code)
            else:
                # Standard precision training
                if model_type == "LSTM_LPF":
                    # Reset pre-allocated tensors to zeros (faster than reallocating)
                    actual_batch_size = X_batch.size(0)
                    h_s_buffer[:, :actual_batch_size, :].zero_()
                    h_c_buffer[:, :actual_batch_size, :].zero_()
                    y_pred, (h_s, h_c), z = model(X_batch, h_s_buffer[:, :actual_batch_size, :], h_c_buffer[:, :actual_batch_size, :], z)
                elif model_type in ["LSTM", "LSTM_EMA"]:
                    # Reset pre-allocated tensors to zeros (faster than reallocating)
                    actual_batch_size = X_batch.size(0)
                    h_s_buffer[:, :actual_batch_size, :].zero_()
                    h_c_buffer[:, :actual_batch_size, :].zero_()
                    y_pred, (h_s, h_c) = model(X_batch, h_s_buffer[:, :actual_batch_size, :], h_c_buffer[:, :actual_batch_size, :])
                elif model_type == "GRU":
                    # Reset pre-allocated tensor to zeros (faster than reallocating)
                    actual_batch_size = X_batch.size(0)
                    h_s_buffer[:, :actual_batch_size, :].zero_()
                    y_pred, h_s = model(X_batch, h_s_buffer[:, :actual_batch_size, :])
                elif model_type == "FNN":
                    y_pred = model(X_batch)
                else:
                    raise ValueError(f"Unsupported model_type in train_epoch: {model_type}")

                # Ensure y_pred and y_batch have compatible shapes for loss calculation
                if y_pred.ndim > y_batch.ndim and y_pred.shape[-1] == 1 and y_batch.ndim == 1:
                    y_pred = y_pred.squeeze(-1)
                elif y_pred.ndim > y_batch.ndim and y_pred.shape[0] == y_batch.shape[0] and y_pred.shape[-1] == y_batch.shape[-1]:
                    pass # Assume model output is appropriate

                if y_batch.ndim == 1 and y_pred.ndim == 2 and y_pred.shape[1] == 1:
                    y_batch = y_batch.unsqueeze(1)

                base_loss = self.criterion(y_pred, y_batch)
                physics_penalty = self._physics_dtdt_penalty(y_pred, y_batch, task)
                smoothness_penalty = self._prediction_smoothness_penalty(y_pred, task)
                loss = base_loss + physics_penalty + smoothness_penalty
                
                # Check for invalid loss before backpropagation
                if not torch.isfinite(loss):
                    print(f"WARNING: Epoch {epoch}, Batch {batch_idx}: Invalid loss detected (NaN/Inf)")
                    # PERFORMANCE: Removed .item()/.min()/.max() calls - they sync GPU→CPU and are very slow
                    optimizer.zero_grad()
                    # Detach hidden states to break computation graph
                    if h_s is not None:
                        h_s = h_s.detach()
                    if h_c is not None:
                        h_c = h_c.detach()
                    continue
                
                # Backward pass FIRST to compute gradients
                loss.backward()
                
                # No gradient clipping (reference code: Junran Chen)
                
                # Finally update weights
                optimizer.step()
            
            # CRITICAL: Detach hidden states immediately after successful backward pass
            # This prevents gradient graph accumulation across batches which causes:
            # - Memory buildup
            # - Gradient instability 
            # - Training spikes/explosions in RNNs
            # Do this BEFORE appending to loss history to ensure clean state
            if model_type in ["LSTM", "LSTM_EMA", "LSTM_LPF"]:
                if h_s is not None:
                    h_s = h_s.detach()
                if h_c is not None:
                    h_c = h_c.detach()
            elif model_type == "GRU":
                if h_s is not None:
                    h_s = h_s.detach()
                
            # Keep loss on GPU to avoid synchronization - will sync once at end of epoch
            total_train_loss.append(loss.detach())
            
            # Store predictions and true values - keep on GPU for speed
            all_train_y_pred_normalized.append(y_pred.detach())
            all_train_y_true_normalized.append(y_batch.detach())

            end_batch_time = time.time()
            batch_time = end_batch_time - start_batch_time
            batch_times.append(batch_time)

            # OPTIMIZATION: Reduce terminal logging frequency to avoid GPU→CPU sync overhead
            # GUI updates happen every epoch (not affected by this)
            # Only log terminal messages every 200 batches instead of 100
            if verbose and batch_idx % (log_freq * 2) == 0 and batch_times:
                log_callback = task.get('log_callback')
                if log_callback:
                    # GUI callback - keep this frequent for user feedback
                    if hybrid_loss_enabled or self._is_smoothness_enabled(task):
                        log_callback(
                            f"  Epoch: {epoch}, Batch: {batch_idx}/{len(train_loader)}, "
                            f"Base(MSE): {base_loss.item():.4f}, PhysicsPenalty: {physics_penalty.item():.4f}, SmoothPenalty: {smoothness_penalty.item():.4f}, Total: {loss.item():.4f}"
                        )
                    else:
                        log_callback(f"  Epoch: {epoch}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
                else:
                    # Terminal logging - reduce frequency
                    if hybrid_loss_enabled or self._is_smoothness_enabled(task):
                        print(
                            f"Epoch: {epoch}, Batch: {batch_idx}/{len(train_loader)}, "
                            f"Base(MSE): {base_loss.item():.4f}, PhysicsPenalty: {physics_penalty.item():.4f}, SmoothPenalty: {smoothness_penalty.item():.4f}, Total: {loss.item():.4f}"
                        )
                    else:
                        print(f"Epoch: {epoch}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            # CRITICAL: For shuffled sequence training, RESET hidden states to None after each batch
            # Shuffled sequences are temporally disconnected, so carrying hidden states
            # (even detached ones) causes gradient instability and training explosions.
            # The memory allocation cost is worth it for training stability!
            if model_type == "LSTM_LPF":
                h_s, h_c, z = None, None, None
            elif model_type in ["LSTM", "LSTM_EMA"]:
                h_s, h_c = None, None
            elif model_type == "GRU":
                h_s = None

        # Calculate average batch time
        avg_epoch_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0

        # Convert loss tensors to scalar - single GPU→CPU sync at end of epoch
        if total_train_loss and torch.is_tensor(total_train_loss[0]):
            total_train_loss = [l.item() for l in total_train_loss]
        avg_loss = sum(total_train_loss) / len(total_train_loss) if total_train_loss else float('nan')
        
        # Concatenate all batch tensors
        if all_train_y_pred_normalized:
            all_train_y_pred_normalized = torch.cat(all_train_y_pred_normalized, dim=0)
        if all_train_y_true_normalized:
            all_train_y_true_normalized = torch.cat(all_train_y_true_normalized, dim=0)
            
        return avg_epoch_batch_time, avg_loss, all_train_y_pred_normalized, all_train_y_true_normalized

    def validate_epoch(self, model, model_type, val_loader, h_s_initial, h_c_initial, epoch, device, stop_requested, task, verbose=True):
        """Validate the model for a single epoch, adapting to model type."""
        model.eval()
        total_val_loss = []
        all_val_y_pred_normalized = [] # To store all predictions from the epoch
        all_val_y_true_normalized = [] # To store all true values from the epoch
        log_freq = task.get('log_frequency', 100)
        
        # Check if mixed precision training is enabled for validation as well
        use_mixed_precision = task['hyperparams'].get('USE_MIXED_PRECISION', False) and device.type == 'cuda'
        
        if use_mixed_precision:
            print(f"Using mixed precision for validation in epoch {epoch}")
        
        # PERFORMANCE OPTIMIZATION: Pre-allocate hidden state tensors to avoid repeated allocations
        max_batch_size = val_loader.batch_size
        if model_type in ["LSTM", "LSTM_EMA", "LSTM_LPF"]:
            h_s_buffer = torch.zeros(model.num_layers, max_batch_size, model.hidden_units, device=device)
            h_c_buffer = torch.zeros(model.num_layers, max_batch_size, model.hidden_units, device=device)
        elif model_type == "GRU":
            h_s_buffer = torch.zeros(model.num_layers, max_batch_size, model.hidden_units, device=device)
        
        # Reference code approach: hidden states will be reset to zeros at START of each batch
        with torch.no_grad():
            for batch_idx, (X_batch, y_batch) in enumerate(val_loader):
                # RESET hidden states to zeros for EVERY batch (reference code behavior)
                h_s, h_c = None, None
                z = None  # Initialize filter state for LPF models
                if stop_requested:
                    print("Stop requested during validation")
                    break
                
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                X_batch = self._apply_batch_initial_temperature_anchor(X_batch, y_batch, task)

                # Use AMP autocast context manager when mixed precision is enabled
                if use_mixed_precision:
                    with torch.amp.autocast('cuda'):
                        if model_type == "LSTM_LPF":
                            # Reset pre-allocated tensors to zeros
                            actual_batch_size = X_batch.size(0)
                            h_s_buffer[:, :actual_batch_size, :].zero_()
                            h_c_buffer[:, :actual_batch_size, :].zero_()
                            y_pred, (h_s, h_c), z = model(X_batch, h_s_buffer[:, :actual_batch_size, :], h_c_buffer[:, :actual_batch_size, :], z)
                        elif model_type in ["LSTM", "LSTM_EMA"]:
                            # Reset pre-allocated tensors to zeros
                            actual_batch_size = X_batch.size(0)
                            h_s_buffer[:, :actual_batch_size, :].zero_()
                            h_c_buffer[:, :actual_batch_size, :].zero_()
                            y_pred, (h_s, h_c) = model(X_batch, h_s_buffer[:, :actual_batch_size, :], h_c_buffer[:, :actual_batch_size, :])
                        elif model_type == "GRU":
                            # Reset pre-allocated tensor to zeros
                            actual_batch_size = X_batch.size(0)
                            h_s_buffer[:, :actual_batch_size, :].zero_()
                            y_pred, h_s = model(X_batch, h_s_buffer[:, :actual_batch_size, :])
                        elif model_type == "FNN":
                            y_pred = model(X_batch)
                        else:
                            raise ValueError(f"Unsupported model_type in validate_epoch: {model_type}")
                        
                        if y_pred.ndim > y_batch.ndim and y_pred.shape[-1] == 1 and y_batch.ndim == 1:
                            y_pred = y_pred.squeeze(-1)
                        elif y_pred.ndim > y_batch.ndim and y_pred.shape[0] == y_batch.shape[0] and y_pred.shape[-1] == y_batch.shape[-1]:
                            pass  # Assume model output is appropriate

                        if y_batch.ndim == 1 and y_pred.ndim == 2 and y_pred.shape[1] == 1:
                            y_batch = y_batch.unsqueeze(1)

                        base_loss = self.criterion(y_pred, y_batch)
                        physics_penalty = self._physics_dtdt_penalty(y_pred, y_batch, task)
                        smoothness_penalty = self._prediction_smoothness_penalty(y_pred, task)
                        loss = base_loss + physics_penalty + smoothness_penalty
                else:
                    if model_type == "LSTM_LPF":
                        # Reset pre-allocated tensors to zeros
                        actual_batch_size = X_batch.size(0)
                        h_s_buffer[:, :actual_batch_size, :].zero_()
                        h_c_buffer[:, :actual_batch_size, :].zero_()
                        y_pred, (h_s, h_c), z = model(X_batch, h_s_buffer[:, :actual_batch_size, :], h_c_buffer[:, :actual_batch_size, :], z)
                    elif model_type in ["LSTM", "LSTM_EMA"]:
                        # Reset pre-allocated tensors to zeros
                        actual_batch_size = X_batch.size(0)
                        h_s_buffer[:, :actual_batch_size, :].zero_()
                        h_c_buffer[:, :actual_batch_size, :].zero_()
                        y_pred, (h_s, h_c) = model(X_batch, h_s_buffer[:, :actual_batch_size, :], h_c_buffer[:, :actual_batch_size, :])
                    elif model_type == "GRU":
                        # Reset pre-allocated tensor to zeros
                        actual_batch_size = X_batch.size(0)
                        h_s_buffer[:, :actual_batch_size, :].zero_()
                        y_pred, h_s = model(X_batch, h_s_buffer[:, :actual_batch_size, :])
                    elif model_type == "FNN":
                        y_pred = model(X_batch)
                    else:
                        raise ValueError(f"Unsupported model_type in validate_epoch: {model_type}")
                    
                    if y_pred.ndim > y_batch.ndim and y_pred.shape[-1] == 1 and y_batch.ndim == 1:
                        y_pred = y_pred.squeeze(-1)
                    elif y_pred.ndim > y_batch.ndim and y_pred.shape[0] == y_batch.shape[0] and y_pred.shape[-1] == y_batch.shape[-1]:
                        pass  # Assume model output is appropriate

                    if y_batch.ndim == 1 and y_pred.ndim == 2 and y_pred.shape[1] == 1:
                        y_batch = y_batch.unsqueeze(1)

                    base_loss = self.criterion(y_pred, y_batch)
                    physics_penalty = self._physics_dtdt_penalty(y_pred, y_batch, task)
                smoothness_penalty = self._prediction_smoothness_penalty(y_pred, task)
                loss = base_loss + physics_penalty + smoothness_penalty

                # Keep loss on GPU to avoid synchronization
                total_val_loss.append(loss.detach())
                
                # Store predictions and true values - keep on GPU for speed
                all_val_y_pred_normalized.append(y_pred.detach())
                all_val_y_true_normalized.append(y_batch.detach())

                # OPTIMIZATION: Reduce validation terminal logging to minimize GPU→CPU syncs
                # GUI updates happen every validation epoch (not affected)
                if verbose and batch_idx % (log_freq * 2) == 0:
                    log_callback = task.get('log_callback')
                    if log_callback:
                        # GUI callback - keep frequent for user feedback
                        log_callback(f"  Validation Epoch: {epoch}, Batch: {batch_idx}/{len(val_loader)}, Loss: {loss.item():.4f}")
                    else:
                        # Terminal logging - reduced frequency
                        print(f"Validation Epoch: {epoch}, Batch: {batch_idx}/{len(val_loader)}, Loss: {loss.item():.4f}")
                
                # CRITICAL: For shuffled sequence validation, RESET hidden states to None
                # Shuffled sequences are temporally disconnected, so carrying hidden states
                # causes validation instability. Reset for clean batch-to-batch evaluation.
                if model_type == "LSTM_LPF":
                    h_s, h_c, z = None, None, None
                elif model_type in ["LSTM", "LSTM_EMA"]:
                    h_s, h_c = None, None
                elif model_type == "GRU":
                    h_s = None
        
        # Convert loss tensors to scalar - single GPU→CPU sync at end of validation
        if total_val_loss and torch.is_tensor(total_val_loss[0]):
            total_val_loss = [l.item() for l in total_val_loss]
        avg_loss = sum(total_val_loss) / len(total_val_loss) if total_val_loss else float('nan')
        
        # Concatenate all batch tensors
        if all_val_y_pred_normalized:
            all_val_y_pred_normalized = torch.cat(all_val_y_pred_normalized, dim=0)
        if all_val_y_true_normalized:
            all_val_y_true_normalized = torch.cat(all_val_y_true_normalized, dim=0)
            
        return avg_loss, all_val_y_pred_normalized, all_val_y_true_normalized

    def save_model(self, model, model_path):
        """Save the model to disk."""
        torch.save(model.state_dict(), model_path)

        # Save hyperparameters as well
        with open(model_path + '_hyperparams.json', 'w') as f:
            json.dump(model.hyperparams, f, indent=4)

    def get_optimizer(self, model, lr, optimizer_type: str = 'Adam', weight_decay: float = 0.0, capturable: bool = False):
        """Initialize the optimizer for the model.

        Args:
            model: The model whose parameters will be optimized.
            lr (float): Initial learning rate.
            optimizer_type (str): 'Adam' or 'AdamW' (default: 'Adam').
            weight_decay (float): Weight decay coefficient. For AdamW this is decoupled; for Adam it behaves like L2.
            capturable (bool): Set capturable=True when using CUDA Graphs.

        Returns:
            torch.optim.Optimizer
        """
        optimizer_type = (optimizer_type or 'Adam').strip()
        try:
            if optimizer_type.lower() == 'adamw':
                return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, capturable=capturable)
            else:
                return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, capturable=capturable)
        except TypeError:
            # Fallback for older PyTorch without capturable kw
            if optimizer_type.lower() == 'adamw':
                return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            else:
                return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def get_scheduler(self, optimizer, lr_drop_period):
        """Initialize the learning rate scheduler."""
        # Create a learning rate scheduler that reduces the LR by 10% every lr_drop_period epochs
        return optim.lr_scheduler.StepLR(optimizer, step_size=lr_drop_period, gamma=0.1)
