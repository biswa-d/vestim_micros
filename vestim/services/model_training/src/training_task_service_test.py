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

        
    def train_epoch(self, model, train_loader, optimizer, h_s, h_c, epoch, device, stop_requested, task):
        """Train the model for a single epoch."""
        model.train()
        total_train_loss = []
        batch_times = []  # Store time per batch
        log_freq = 100  # Define how often to log batches
        device_str = str(device)  # Convert torch.device to string

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            if stop_requested:  # Check if a stop has been requested
                print("Stop requested during training")
                break  # Exit the loop if stop is requested

            start_batch_time = time.time()  # Start timing for this batch

            h_s, h_c = torch.zeros_like(h_s), torch.zeros_like(h_c)
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred, (h_s, h_c) = model(X_batch, h_s, h_c)
            # y_pred = y_pred.squeeze(-1)

            loss = self.criterion(y_pred[:, -1, :], y_batch)
            loss.backward()
            optimizer.step()

            total_train_loss.append(loss.item())
            
            end_batch_time = time.time()  # End timing for this batch
            batch_time = end_batch_time - start_batch_time
            batch_times.append(batch_time)

            # Log less frequently
            if batch_idx % log_freq == 0:
                batch_freq_time = sum(batch_times) / len(batch_times)
                # self.log_to_csv(task, epoch, batch_idx, batch_freq_time, phase='train', device=device)
                self.log_to_sqlite(task, epoch, batch_idx, batch_freq_time, phase='train', device=device_str)

            # Log progress every 150 batches
            if batch_idx % log_freq == 0:
                print(f"Task ID: {task['task_id']}, Epoch: {epoch}, Batch: {batch_idx}, Input shape: {X_batch.shape}")
                print(f"Task ID: {task['task_id']}, Epoch: {epoch}, Batch: {batch_idx}, Output shape after LSTM: {y_pred.shape}")
            
            # Clear unused memory
            del X_batch, y_batch, y_pred  # Explicitly clear tensors

        avg_batch_time = sum(batch_times) / len(batch_times)  # Average batch time
        return avg_batch_time, sum(total_train_loss) / len(total_train_loss)


    def validate_epoch(self, model, val_loader, h_s, h_c, epoch, device, stop_requested, task):
        """Validate the model for a single epoch."""
        model.eval()
        total_loss = 0
        total_samples = 0
        batch_times = []  # Track validation time for each batch
        log_freq = 100  # Define how often to log batches
        device_str = str(device)  # Convert torch.device to string

        with torch.no_grad():
            for batch_idx, (X_batch, y_batch) in enumerate(val_loader):
                if stop_requested:  # Check if a stop has been requested
                    print("Stop requested during validation")
                    break  # Exit the loop if stop is requested

                start_batch_time = time.time()  # Start timing for this batch
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred, (h_s, h_c) = model(X_batch, h_s, h_c)
                # y_pred = y_pred.squeeze(-1)

                # loss = self.criterion(y_pred, y_batch)
                loss = self.criterion(y_pred[:, -1, :], y_batch)
                total_loss += loss.item() * X_batch.size(0)
                total_samples += X_batch.size(0)

                end_batch_time = time.time()  # End timing for this batch
                batch_time = end_batch_time - start_batch_time
                batch_times.append(batch_time)
                
                # Log less frequently
                if batch_idx % log_freq == 0:
                    batch_freq_time = sum(batch_times) / len(batch_times)
                    # self.log_to_csv(task, epoch, batch_idx, batch_freq_time, phase='validate')
                    self.log_to_sqlite(task, epoch, batch_idx, batch_freq_time, phase='validate', device=device_str)

                # Log progress every 150 batches
                if batch_idx % log_freq == 0:
                    print(f"Task ID: {task['task_id']}, Epoch: {epoch}, Batch: {batch_idx}, Input shape: {X_batch.shape}")
                    print(f"Task ID: {task['task_id']}, Epoch: {epoch}, Batch: {batch_idx}, Output shape after LSTM: {y_pred.shape}")
                
                # Clear unused memory
                del X_batch, y_batch, y_pred  # Explicitly clear tensors

        return  total_loss / total_samples

    def save_model(self, model, model_path):
        """Save the model to disk."""
        torch.save(model.state_dict(), model_path)

        # Save hyperparameters as well
        with open(model_path + '_hyperparams.json', 'w') as f:
            json.dump(model.hyperparams, f, indent=4)

    def get_optimizer(self, model, lr, weight_decay=0):

        """Initialize the optimizer for the model."""
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def get_scheduler(self, optimizer, step_size, gamma=0.1):
        """Initialize the learning rate scheduler with step size and gamma."""
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

