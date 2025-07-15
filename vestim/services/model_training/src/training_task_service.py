# filepath: c:\Users\dehuryb\OneDrive - McMaster University\Models\ML_LiB_Models\vestim_micros\vestim\services\model_training\src\training_task_service.py.fixed
import torch, numpy as np
import torch.nn as nn
import torch.optim as optim
import json, csv, sqlite3, os
import time


class TrainingTaskService:
    def __init__(self):
        self.criterion = nn.MSELoss()  # Assuming you're using Mean Squared Error Loss for regression tasks
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
    
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
        
    def train_epoch(self, model, model_type, train_loader, optimizer, h_s_initial, h_c_initial, epoch, device, stop_requested, task):
        """Train the model for a single epoch, adapting to model type."""
        model.train()
        total_train_loss = []
        all_train_y_pred_normalized = [] # To store all predictions from the epoch
        all_train_y_true_normalized = [] # To store all true values from the epoch
        batch_times = []
        log_freq = task.get('log_frequency', 100)
        
        # Check if mixed precision training is enabled
        use_mixed_precision = task['hyperparams'].get('USE_MIXED_PRECISION', False) and device.type == 'cuda'
        scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
        
        if use_mixed_precision:
            print(f"Using mixed precision training (AMP) for epoch {epoch}")
        
        # Make copies of initial hidden states to be used and modified in the loop if RNN
        h_s, h_c = None, None
        if model_type in ["LSTM", "GRU"]: # Or any other RNN type needing hidden states
            if h_s_initial is not None:
                h_s = h_s_initial.detach().clone() # Detach and clone for each epoch start
            if model_type == "LSTM" and h_c_initial is not None:
                h_c = h_c_initial.detach().clone()

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
                with torch.cuda.amp.autocast():
                    # Forward pass with mixed precision
                    if model_type == "LSTM":
                        y_pred, (h_s, h_c) = model(X_batch, h_s, h_c)
                    elif model_type == "GRU":
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
                
                # Mixed precision backward and optimizer step
                scaler.scale(loss).backward()
                
                # Add gradient clipping to prevent exploding gradients (especially for RNNs)
                if model_type in ["LSTM", "GRU"]:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard precision training
                if model_type == "LSTM":
                    if h_s is None:
                        h_s = torch.zeros(model.num_layers, X_batch.size(0), model.hidden_units).to(device)
                        h_c = torch.zeros(model.num_layers, X_batch.size(0), model.hidden_units).to(device)
                    y_pred, (h_s, h_c) = model(X_batch, h_s, h_c)
                elif model_type == "GRU":
                    if h_s is None:
                        h_s = torch.zeros(model.num_layers, X_batch.size(0), model.hidden_units).to(device)
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
                
                # Add gradient clipping to prevent exploding gradients (especially for RNNs)
                if model_type in ["LSTM", "GRU"]:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                loss.backward()
                optimizer.step()
                
            total_train_loss.append(loss.item())
            
            # Store predictions and true values
            all_train_y_pred_normalized.append(y_pred.detach().cpu())
            all_train_y_true_normalized.append(y_batch.detach().cpu()) # y_batch is already on device, move to cpu

            end_batch_time = time.time()
            batch_time = end_batch_time - start_batch_time
            batch_times.append(batch_time)

            if batch_idx % log_freq == 0 and batch_times:
                avg_recent_batch_time = sum(batch_times[-log_freq:]) / len(batch_times[-log_freq:])
                print(f"Epoch: {epoch}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Input: {X_batch.shape}, Pred: {y_pred.shape}")
                if use_mixed_precision:
                    print(f"  Using mixed precision (AMP)")

            del X_batch, y_batch, y_pred, loss
            if model_type == "LSTM":
                h_s = h_s.detach()
                h_c = h_c.detach()
            elif model_type == "GRU":
                h_s = h_s.detach()
            torch.cuda.empty_cache() if device.type == 'cuda' else None

        avg_epoch_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
        avg_loss = sum(total_train_loss) / len(total_train_loss) if total_train_loss else float('nan')
        
        # Concatenate all batch tensors
        if all_train_y_pred_normalized:
            all_train_y_pred_normalized = torch.cat(all_train_y_pred_normalized, dim=0)
        if all_train_y_true_normalized:
            all_train_y_true_normalized = torch.cat(all_train_y_true_normalized, dim=0)
            
        return avg_epoch_batch_time, avg_loss, all_train_y_pred_normalized, all_train_y_true_normalized

    def validate_epoch(self, model, model_type, val_loader, h_s_initial, h_c_initial, epoch, device, stop_requested, task):
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
        if model_type in ["LSTM", "GRU"]:
            if h_s_initial is not None:
                h_s = h_s_initial.detach().clone()
            if model_type == "LSTM" and h_c_initial is not None:
                h_c = h_c_initial.detach().clone()

        with torch.no_grad():
            for batch_idx, (X_batch, y_batch) in enumerate(val_loader):
                if stop_requested:
                    print("Stop requested during validation")
                    break
                
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                # Use AMP autocast context manager when mixed precision is enabled
                if use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        if model_type == "LSTM":
                            y_pred, (_, _) = model(X_batch, h_s, h_c)
                        elif model_type == "GRU":
                            y_pred, _ = model(X_batch, h_s)
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
                    if model_type == "LSTM":
                        if h_s is None:
                            h_s = torch.zeros(model.num_layers, X_batch.size(0), model.hidden_units).to(device)
                            h_c = torch.zeros(model.num_layers, X_batch.size(0), model.hidden_units).to(device)
                        y_pred, (_, _) = model(X_batch, h_s, h_c)
                    elif model_type == "GRU":
                        if h_s is None:
                            h_s = torch.zeros(model.num_layers, X_batch.size(0), model.hidden_units).to(device)
                        y_pred, _ = model(X_batch, h_s)
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
                
                # Store predictions and true values
                all_val_y_pred_normalized.append(y_pred.detach().cpu())
                all_val_y_true_normalized.append(y_batch.detach().cpu())

                if batch_idx % log_freq == 0:
                    print(f"Validation Epoch: {epoch}, Batch: {batch_idx}/{len(val_loader)}, Loss: {loss.item():.4f}")
                    if use_mixed_precision:
                        print(f"  Using mixed precision (AMP)")

                del X_batch, y_batch, y_pred, loss
                if model_type == "LSTM":
                    h_s = h_s.detach()
                    h_c = h_c.detach()
                elif model_type == "GRU":
                    h_s = h_s.detach()
                torch.cuda.empty_cache() if device.type == 'cuda' else None
        
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
