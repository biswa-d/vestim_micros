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
        
    def train_epoch(self, model, model_type, train_loader, optimizer, h_s_initial, h_c_initial, epoch, device, stop_requested, task, verbose=True):
        """Train the model for a single epoch, adapting to model type."""
        model.train()
        total_train_loss = []
        all_train_y_pred_normalized = [] # To store all predictions from the epoch
        all_train_y_true_normalized = [] # To store all true values from the epoch
        batch_times = []
        log_freq = task.get('log_frequency', 100)
        
        # Check if mixed precision training is enabled
        use_mixed_precision = task['hyperparams'].get('USE_MIXED_PRECISION', False) and device.type == 'cuda'
        scaler = torch.amp.GradScaler('cuda') if use_mixed_precision else None
        
        if use_mixed_precision:
            print(f"Using mixed precision training (AMP) for epoch {epoch}")
        
        # Make copies of initial hidden states to be used and modified in the loop if RNN
        h_s, h_c = None, None
        z = None # Initialize filter state for LPF models
        if model_type in ["LSTM", "GRU", "LSTM_EMA", "LSTM_LPF"]: # Or any other RNN type needing hidden states
            if h_s_initial is not None:
                h_s = h_s_initial.detach().clone().to(device)  # Ensure cloned states are moved to the correct device
            if model_type in ["LSTM", "LSTM_EMA", "LSTM_LPF"] and h_c_initial is not None:
                h_c = h_c_initial.detach().clone().to(device)  # Ensure cloned states are moved to the correct device

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            if stop_requested:
                print("Stop requested during training")
                break

            start_batch_time = time.time()
            # Use the 'device' argument passed to the method, not self.device
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()

            # Use AMP autocast context manager when mixed precision is enabled
            if use_mixed_precision:
                with torch.amp.autocast('cuda'):
                    # Forward pass with mixed precision
                    if model_type == "LSTM_LPF":
                        if h_s is None or h_c is None: # Ensure hidden states are initialized
                            h_s = torch.zeros(model.num_layers, X_batch.size(0), model.hidden_units, device=device)
                            h_c = torch.zeros(model.num_layers, X_batch.size(0), model.hidden_units, device=device)
                        y_pred, (h_s, h_c), z = model(X_batch, h_s, h_c, z)
                    elif model_type in ["LSTM", "LSTM_EMA"]:
                        if h_s is None or h_c is None: # Ensure hidden states are initialized
                            h_s = torch.zeros(model.num_layers, X_batch.size(0), model.hidden_units, device=device)
                            h_c = torch.zeros(model.num_layers, X_batch.size(0), model.hidden_units, device=device)
                        else:
                            # Ensure existing hidden states are on the correct device
                            h_s = h_s.to(device)
                            h_c = h_c.to(device)
                        y_pred, (h_s, h_c) = model(X_batch, h_s, h_c)
                    elif model_type == "GRU":
                        if h_s is None: # Ensure hidden state is initialized
                            h_s = torch.zeros(model.num_layers, X_batch.size(0), model.hidden_units, device=device)
                        else:
                            # Ensure existing hidden state is on the correct device
                            h_s = h_s.to(device)
                        y_pred, h_s = model(X_batch, h_s)
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

                    loss = self.criterion(y_pred, y_batch)
                
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
                
                # Add gradient clipping to prevent exploding gradients (especially for RNNs)
                if model_type in ["LSTM", "GRU"]:
                    scaler.unscale_(optimizer)
                    # Clip gradients more aggressively with max_norm=0.5
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    # Check for invalid gradients after clipping
                    if not torch.isfinite(grad_norm):
                        print(f"WARNING: Epoch {epoch}, Batch {batch_idx}: Invalid gradient norm detected (NaN/Inf)")
                        # PERFORMANCE: Removed .item() calls - they sync GPU→CPU and are very slow
                        optimizer.zero_grad()  # Clear bad gradients before skipping
                        scaler.update()  # Update scaler state even when skipping
                        # Detach hidden states to break computation graph
                        if h_s is not None:
                            h_s = h_s.detach()
                        if h_c is not None:
                            h_c = h_c.detach()
                        continue  # Skip this batch update
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard precision training
                if model_type == "LSTM_LPF":
                    if h_s is None or h_c is None:
                        h_s = torch.zeros(model.num_layers, X_batch.size(0), model.hidden_units).to(device)
                        h_c = torch.zeros(model.num_layers, X_batch.size(0), model.hidden_units).to(device)
                    y_pred, (h_s, h_c), z = model(X_batch, h_s, h_c, z)
                elif model_type in ["LSTM", "LSTM_EMA"]:
                    if h_s is None or h_c is None:
                        h_s = torch.zeros(model.num_layers, X_batch.size(0), model.hidden_units).to(device)
                        h_c = torch.zeros(model.num_layers, X_batch.size(0), model.hidden_units).to(device)
                    else:
                        # Ensure existing hidden states are on the correct device
                        h_s = h_s.to(device)
                        h_c = h_c.to(device)
                    y_pred, (h_s, h_c) = model(X_batch, h_s, h_c)
                elif model_type == "GRU":
                    if h_s is None:
                        h_s = torch.zeros(model.num_layers, X_batch.size(0), model.hidden_units).to(device)
                    else:
                        # Ensure existing hidden state is on the correct device
                        h_s = h_s.to(device)
                    y_pred, h_s = model(X_batch, h_s)
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

                loss = self.criterion(y_pred, y_batch)
                
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
                
                # THEN clip gradients to prevent exploding gradients (especially for RNNs)
                if model_type in ["LSTM", "GRU"]:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    # Check for invalid gradients after clipping
                    if not torch.isfinite(grad_norm):
                        print(f"WARNING: Epoch {epoch}, Batch {batch_idx}: Invalid gradient norm detected (NaN/Inf)")
                        # PERFORMANCE: Removed .item() calls - they sync GPU→CPU and are very slow
                        optimizer.zero_grad()  # Clear the bad gradients
                        # Detach hidden states to break computation graph
                        if h_s is not None:
                            h_s = h_s.detach()
                        if h_c is not None:
                            h_c = h_c.detach()
                        continue
                
                # Finally update weights
                optimizer.step()
                
            total_train_loss.append(loss.item())
            
            # Store predictions and true values - keep on GPU for speed
            all_train_y_pred_normalized.append(y_pred.detach())
            all_train_y_true_normalized.append(y_batch.detach())

            end_batch_time = time.time()
            batch_time = end_batch_time - start_batch_time
            batch_times.append(batch_time)

            if verbose and batch_idx % log_freq == 0 and batch_times:
                log_callback = task.get('log_callback')
                if log_callback:
                    log_callback(f"  Epoch: {epoch}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
                else:
                    print(f"Epoch: {epoch}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Input: {X_batch.shape}, Pred: {y_pred.shape}")

            del X_batch, y_batch, y_pred, loss
            
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
            
            # Only clear CUDA cache periodically to avoid overhead
            if device.type == 'cuda' and batch_idx % 50 == 0:
                torch.cuda.empty_cache()

        avg_epoch_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
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
        
        h_s, h_c = None, None
        z = None # Initialize filter state for LPF models
        if model_type in ["LSTM", "GRU", "LSTM_EMA", "LSTM_LPF"]:
            if h_s_initial is not None:
                h_s = h_s_initial.detach().clone().to(device)  # Ensure cloned states are moved to the correct device
            if model_type in ["LSTM", "LSTM_EMA", "LSTM_LPF"] and h_c_initial is not None:
                h_c = h_c_initial.detach().clone().to(device)  # Ensure cloned states are moved to the correct device

        with torch.no_grad():
            for batch_idx, (X_batch, y_batch) in enumerate(val_loader):
                if stop_requested:
                    print("Stop requested during validation")
                    break
                
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                # Use AMP autocast context manager when mixed precision is enabled
                if use_mixed_precision:
                    with torch.amp.autocast('cuda'):
                        if model_type == "LSTM_LPF":
                            if h_s is None or h_c is None:
                                h_s = torch.zeros(model.num_layers, X_batch.size(0), model.hidden_units, device=device)
                                h_c = torch.zeros(model.num_layers, X_batch.size(0), model.hidden_units, device=device)
                            y_pred, (h_s, h_c), z = model(X_batch, h_s, h_c, z)
                        elif model_type in ["LSTM", "LSTM_EMA"]:
                            if h_s is None or h_c is None:
                                h_s = torch.zeros(model.num_layers, X_batch.size(0), model.hidden_units, device=device)
                                h_c = torch.zeros(model.num_layers, X_batch.size(0), model.hidden_units, device=device)
                            else:
                                # Ensure existing hidden states are on the correct device
                                h_s = h_s.to(device)
                                h_c = h_c.to(device)
                            y_pred, (h_s, h_c) = model(X_batch, h_s, h_c)
                        elif model_type == "GRU":
                            if h_s is None:
                                h_s = torch.zeros(model.num_layers, X_batch.size(0), model.hidden_units, device=device)
                            else:
                                # Ensure existing hidden state is on the correct device
                                h_s = h_s.to(device)
                            y_pred, h_s = model(X_batch, h_s)
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

                        loss = self.criterion(y_pred, y_batch)
                else:
                    if model_type == "LSTM_LPF":
                        if h_s is None or h_c is None:
                            h_s = torch.zeros(model.num_layers, X_batch.size(0), model.hidden_units).to(device)
                            h_c = torch.zeros(model.num_layers, X_batch.size(0), model.hidden_units).to(device)
                        y_pred, (h_s, h_c), z = model(X_batch, h_s, h_c, z)
                    elif model_type in ["LSTM", "LSTM_EMA"]:
                        if h_s is None or h_c is None:
                            h_s = torch.zeros(model.num_layers, X_batch.size(0), model.hidden_units).to(device)
                            h_c = torch.zeros(model.num_layers, X_batch.size(0), model.hidden_units).to(device)
                        else:
                            # Ensure existing hidden states are on the correct device
                            h_s = h_s.to(device)
                            h_c = h_c.to(device)
                        y_pred, (h_s, h_c) = model(X_batch, h_s, h_c)
                    elif model_type == "GRU":
                        if h_s is None:
                            h_s = torch.zeros(model.num_layers, X_batch.size(0), model.hidden_units).to(device)
                        else:
                            # Ensure existing hidden state is on the correct device
                            h_s = h_s.to(device)
                        y_pred, h_s = model(X_batch, h_s)
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

                    loss = self.criterion(y_pred, y_batch)

                total_val_loss.append(loss.item()) # Accumulate loss per batch (matches training approach)
                
                # Store predictions and true values - keep on GPU for speed
                all_val_y_pred_normalized.append(y_pred.detach())
                all_val_y_true_normalized.append(y_batch.detach())

                if verbose and batch_idx % log_freq == 0:
                    log_callback = task.get('log_callback')
                    if log_callback:
                        log_callback(f"  Validation Epoch: {epoch}, Batch: {batch_idx}/{len(val_loader)}, Loss: {loss.item():.4f}")
                    else:
                        print(f"Validation Epoch: {epoch}, Batch: {batch_idx}/{len(val_loader)}, Loss: {loss.item():.4f}")

                del X_batch, y_batch, y_pred, loss
                
                # CRITICAL: For shuffled sequence validation, RESET hidden states to None
                # Shuffled sequences are temporally disconnected, so carrying hidden states
                # causes validation instability. Reset for clean batch-to-batch evaluation.
                if model_type == "LSTM_LPF":
                    h_s, h_c, z = None, None, None
                elif model_type in ["LSTM", "LSTM_EMA"]:
                    h_s, h_c = None, None
                elif model_type == "GRU":
                    h_s = None
                
                # Only clear CUDA cache periodically to avoid overhead
                if device.type == 'cuda' and batch_idx % 50 == 0:
                    torch.cuda.empty_cache()
        
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

    def get_optimizer(self, model, lr):
        """Initialize the optimizer for the model."""
        return optim.Adam(model.parameters(), lr=lr)

    def get_scheduler(self, optimizer, lr_drop_period):
        """Initialize the learning rate scheduler."""
        # Create a learning rate scheduler that reduces the LR by 10% every lr_drop_period epochs
        return optim.lr_scheduler.StepLR(optimizer, step_size=lr_drop_period, gamma=0.1)
