import torch, numpy as np
import torch.nn as nn
import torch.optim as optim
import json, csv, sqlite3, os
import time

class TrainingTaskService:
    def __init__(self):
        self.criterion = nn.MSELoss()  # Assuming you're using Mean Squared Error Loss for regression tasks
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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

        
    def train_epoch(self, model, train_loader, optimizer, epoch, device, stop_requested, task):
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
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            start_batch_time = time.time()  # Start timing for this batch
            # print(f"Model on device: {next(model.parameters()).device}")

            # Initialize hidden states with the actual batch size in each loop
            if isinstance(model, torch.nn.DataParallel):
                h_s = torch.zeros(model.module.num_layers, X_batch.size(0), model.module.hidden_units).to(device)
                h_c = torch.zeros(model.module.num_layers, X_batch.size(0), model.module.hidden_units).to(device)
            else:
                h_s = torch.zeros(model.num_layers, X_batch.size(0), model.hidden_units).to(device)
                h_c = torch.zeros(model.num_layers, X_batch.size(0), model.hidden_units).to(device)

            # Debugging: check if X_batch and hidden states are on the correct device
            # print(f"X_batch on device: {X_batch.device}")
            # print(f"h_s on device: {h_s.device}")
            # print(f"h_c on device: {h_c.device}")
            optimizer.zero_grad()
            y_pred, (h_s, h_c) = model(X_batch, h_s, h_c)
            y_pred = y_pred.squeeze(-1)

            loss = self.criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            total_train_loss.append(loss.item())

            end_batch_time = time.time()  # End timing for this batch
            batch_time = end_batch_time - start_batch_time
            batch_times.append(batch_time)

            # Log less frequently
            if batch_idx % log_freq == 0:
                batch_freq_time = sum(batch_times) / len(batch_times)
                self.log_to_sqlite(task, epoch, batch_idx, batch_freq_time, phase='train', device=device_str)

            # Log progress every 100 batches
            if batch_idx % log_freq == 0:
                print(f"Task ID: {task['task_id']}, Epoch: {epoch}, Batch: {batch_idx}, Input shape: {X_batch.shape}")
                print(f"Task ID: {task['task_id']}, Epoch: {epoch}, Batch: {batch_idx}, Output shape after LSTM: {y_pred.shape}")

            # Clear unused memory
            del X_batch, y_batch, y_pred  # Explicitly clear tensors

        avg_batch_time = sum(batch_times) / len(batch_times)  # Average batch time
        return avg_batch_time, sum(total_train_loss) / len(total_train_loss)


    def validate_epoch(self, model, val_loader, padding_size, epoch, device, stop_requested, task):
        """Validate the model for a single epoch and return the validation loss."""
        model.eval()  # Set model to evaluation mode

        total_loss = 0
        total_samples = 0
        batch_times = []  # Track validation time for each batch
        log_freq = 100  # Define how often to log batches
        device_str = str(device)  # Convert torch.device to string

        all_predictions = []
        all_true_values = []

        with torch.no_grad():
            for batch_idx, (X_batch, y_batch) in enumerate(val_loader):
                if stop_requested:  # Check if a stop has been requested
                    print("Stop requested during validation")
                    break  # Exit the loop if stop is requested

                start_batch_time = time.time()  # Start timing for this batch
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                batch_size = X_batch.size(0)
                X_batch = X_batch.unsqueeze(1)  # Adds a sequence length of 1, as in testing logic

                # Initialize hidden and cell states based on batch size
                if isinstance(model, torch.nn.DataParallel):
                    h_s = torch.zeros(model.module.num_layers, batch_size, model.module.hidden_units).to(device)
                    h_c = torch.zeros(model.module.num_layers, batch_size, model.module.hidden_units).to(device)
                else:
                    h_s = torch.zeros(model.num_layers, batch_size, model.hidden_units).to(device)
                    h_c = torch.zeros(model.num_layers, batch_size, model.hidden_units).to(device)

                # Forward pass
                y_pred, (h_s, h_c) = model(X_batch, h_s, h_c)
                y_pred = y_pred.squeeze(-1)  # Squeeze the last dimension to match y_batch

                # Collect predictions and true values
                all_predictions.append(y_pred.cpu().numpy())
                all_true_values.append(y_batch.cpu().numpy())

                # Calculate loss
                loss = self.criterion(y_pred, y_batch)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                end_batch_time = time.time()  # End timing for this batch
                batch_time = end_batch_time - start_batch_time
                batch_times.append(batch_time)

                # Log progress less frequently
                if batch_idx % log_freq == 0:
                    batch_freq_time = sum(batch_times) / len(batch_times)
                    self.log_to_sqlite(task, epoch, batch_idx, batch_freq_time, phase='validate', device=device_str)

                    print(f"Task ID: {task['task_id']}, Epoch: {epoch}, Batch: {batch_idx}, Input shape: {X_batch.shape}")
                    print(f"Task ID: {task['task_id']}, Epoch: {epoch}, Batch: {batch_idx}, Output shape after LSTM: {y_pred.shape}")

                # Clear unused memory
                del X_batch, y_batch, y_pred  # Explicitly clear tensors

            # Convert to flat arrays for evaluation
            y_pred = np.concatenate(all_predictions, axis=0).flatten()
            y_true = np.concatenate(all_true_values, axis=0)

            print(f"y_pred shape before removing padding: {y_pred.shape}")
            print(f"y_true shape before removing padding: {y_true.shape}")

            # Remove the padded data from the results if padding was applied
            if padding_size > 0:
                y_pred = y_pred[:-padding_size]
                y_true = y_true[:-padding_size]

            print(f"y_pred shape after removing padding: {y_pred.shape}")
            print(f"y_true shape after removing padding: {y_true.shape}")

        # Return average loss for validation
        return total_loss / total_samples


    def save_model(self, model, model_path):
        """Save the model to disk."""
        torch.save(model.state_dict(), model_path)

        # Save hyperparameters as well
        with open(model_path + '_hyperparams.json', 'w') as f:
            json.dump(model.hyperparams, f, indent=4)

    def get_optimizer(self, model, lr):
        """Initialize the optimizer for the model."""
        return optim.Adam(model.parameters(), lr=lr)

    def get_scheduler(self, optimizer, step_size, gamma=0.1):
        """Initialize the learning rate scheduler with step size and gamma."""
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

