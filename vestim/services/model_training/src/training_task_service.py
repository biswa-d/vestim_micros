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
        batch_times = []
        log_freq = task.get('log_frequency', 100)
        
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
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            if model_type == "LSTM":
                # Re-initialize hidden states for each batch if stateful=False (typical for many short sequences)
                # If stateful=True (one long sequence), this h_s, h_c would carry over.
                # For now, assume we re-initialize or the model handles its own state per call for stateless.
                # The h_s, h_c passed are the initial states for the *epoch* or *sequence segment*.
                # If the model is stateful across batches, this needs more sophisticated handling.
                # Assuming for now, each batch is independent or model handles its state.
                # The passed h_s_initial, h_c_initial are for the start of the epoch/sequence.
                # For simplicity, let's assume the model's forward pass can take None for initial hidden states
                # and will initialize them if None. Or, we pass the epoch's initial hidden states.
                # The passed h_s, h_c are for the *start* of the sequence for this batch.
                # The model's forward method should handle if h_s, h_c are None.
                # For now, we assume TrainingTaskManager provides appropriate h_s_initial, h_c_initial
                current_h_s = h_s_initial.detach().clone() if h_s_initial is not None else None
                current_h_c = h_c_initial.detach().clone() if h_c_initial is not None else None
                y_pred, (h_s_out, h_c_out) = model(X_batch, current_h_s, current_h_c) # Model returns new states
                # h_s, h_c = h_s_out.detach(), h_c_out.detach() # For stateful training across batches (not implemented yet)
            elif model_type == "GRU":
                current_h_s = h_s_initial.detach().clone() if h_s_initial is not None else None
                y_pred, h_s_out = model(X_batch, current_h_s)
                # h_s = h_s_out.detach() # For stateful
            elif model_type == "FNN":
                y_pred = model(X_batch)
            else:
                raise ValueError(f"Unsupported model_type in train_epoch: {model_type}")

            # Ensure y_pred and y_batch have compatible shapes for loss calculation
            # y_batch is likely [batch_size, 1] or [batch_size, num_targets]
            # y_pred from RNNs might be [batch_size, sequence_length, num_outputs] -> take last step or average
            # y_pred from FNNs is likely [batch_size, num_outputs]
            # Assuming model's forward pass and DataLoaderService ensure y_pred is [batch_size, num_outputs]
            # and y_batch is [batch_size, num_outputs] or can be squeezed.
            if y_pred.ndim > y_batch.ndim and y_pred.shape[-1] == 1 and y_batch.ndim == 1 : # e.g. y_pred [B,1], y_batch [B]
                 y_pred = y_pred.squeeze(-1)
            elif y_pred.ndim > y_batch.ndim and y_pred.shape[0] == y_batch.shape[0] and y_pred.shape[-1] == y_batch.shape[-1]: # e.g. y_pred [B,S,F] y_batch [B,F] -> needs model to output last step
                 # This case needs careful handling based on model output. For now, assume model output is appropriate.
                 # If RNN outputs all sequence steps, select the relevant one (e.g., last one for many-to-one)
                 # For now, assume y_pred is already [batch_size, num_output_features]
                 pass


            if y_batch.ndim == 1 and y_pred.ndim == 2 and y_pred.shape[1] == 1: # If y_batch is [B] and y_pred is [B,1]
                y_batch = y_batch.unsqueeze(1)


            loss = self.criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss.append(loss.item())
            
            end_batch_time = time.time()
            batch_time = end_batch_time - start_batch_time
            batch_times.append(batch_time)

            if batch_idx % log_freq == 0 and batch_times:
                avg_recent_batch_time = sum(batch_times[-log_freq:]) / len(batch_times[-log_freq:])
                # self.log_to_csv(task, epoch, batch_idx, avg_recent_batch_time, phase='train') # device was an error here
                # self.log_to_sqlite(task, epoch, batch_idx, avg_recent_batch_time, phase='train', device=str(device))
                print(f"Epoch: {epoch}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Input: {X_batch.shape}, Pred: {y_pred.shape}")

            del X_batch, y_batch, y_pred, loss
            if model_type == "LSTM": del current_h_s, current_h_c, h_s_out, h_c_out
            elif model_type == "GRU": del current_h_s, h_s_out
            torch.cuda.empty_cache() if device.type == 'cuda' else None


        avg_epoch_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
        return avg_epoch_batch_time, sum(total_train_loss) / len(total_train_loss) if total_train_loss else float('nan')


    def validate_epoch(self, model, model_type, val_loader, h_s_initial, h_c_initial, epoch, device, stop_requested, task):
        """Validate the model for a single epoch, adapting to model type."""
        model.eval()
        total_val_loss = 0
        log_freq = task.get('log_frequency', 100)
        
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

                if model_type == "LSTM":
                    current_h_s = h_s_initial.detach().clone() if h_s_initial is not None else None
                    current_h_c = h_c_initial.detach().clone() if h_c_initial is not None else None
                    y_pred, (_, _) = model(X_batch, current_h_s, current_h_c) # Don't need to carry states in validation per batch typically
                elif model_type == "GRU":
                    current_h_s = h_s_initial.detach().clone() if h_s_initial is not None else None
                    y_pred, _ = model(X_batch, current_h_s)
                elif model_type == "FNN":
                    y_pred = model(X_batch)
                else:
                    raise ValueError(f"Unsupported model_type in validate_epoch: {model_type}")

                if y_pred.ndim > y_batch.ndim and y_pred.shape[-1] == 1 and y_batch.ndim == 1 :
                     y_pred = y_pred.squeeze(-1)
                elif y_pred.ndim > y_batch.ndim and y_pred.shape[0] == y_batch.shape[0] and y_pred.shape[-1] == y_batch.shape[-1]:
                     pass # Assume model output is appropriate

                if y_batch.ndim == 1 and y_pred.ndim == 2 and y_pred.shape[1] == 1:
                    y_batch = y_batch.unsqueeze(1)

                loss = self.criterion(y_pred, y_batch)
                total_val_loss += loss.item() * X_batch.size(0) # Accumulate total loss correctly

                if batch_idx % log_freq == 0:
                     print(f"Validation Epoch: {epoch}, Batch: {batch_idx}/{len(val_loader)}, Loss: {loss.item():.4f}")

                del X_batch, y_batch, y_pred, loss
                if model_type == "LSTM": del current_h_s, current_h_c
                elif model_type == "GRU": del current_h_s
                torch.cuda.empty_cache() if device.type == 'cuda' else None
        
        return total_val_loss / len(val_loader.dataset) if len(val_loader.dataset) > 0 else float('nan')

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
