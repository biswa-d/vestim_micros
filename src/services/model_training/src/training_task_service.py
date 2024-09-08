import torch, numpy as np
import torch.nn as nn
import torch.optim as optim
import json
import logging

class TrainingTaskService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)  # Initialize logger
        self.criterion = nn.MSELoss()  # Assuming you're using Mean Squared Error Loss for regression tasks
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

    def train_epoch(self, model, train_loader, optimizer, h_s, h_c, epoch, device, stop_requested):
        """Train the model for a single epoch."""
        model.train()
        total_train_loss = []

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            if stop_requested:  # Check if a stop has been requested
                print("Stop requested during training")
                break  # Exit the loop if stop is requested

            h_s, h_c = torch.zeros_like(h_s), torch.zeros_like(h_c)
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred, (h_s, h_c) = model(X_batch, h_s, h_c)
            y_pred = y_pred.squeeze(-1)

            loss = self.criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            total_train_loss.append(loss.item())

            # Log the training progress for each batch
            if batch_idx % 150 == 0:  # For example, every 150 batches
                print(f"Epoch: {epoch}, Batch: {batch_idx}, Input shape: {X_batch.shape}")
                print(f"Epoch: {epoch}, Batch: {batch_idx}, Output shape after LSTM: {y_pred.shape}")
            # Clear unused memory
            del X_batch, y_batch, y_pred  # Explicitly clear tensors
        
        self.logger.info(f"Completed epoch {epoch} | Avg Loss: {sum(total_train_loss) / len(total_train_loss)}")
        return sum(total_train_loss) / len(total_train_loss)
        

    def validate_epoch(self, model, val_loader, h_s, h_c, epoch, device, stop_requested):
        """Validate the model for a single epoch."""
        model.eval()
        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, (X_batch, y_batch) in enumerate(val_loader):
                if stop_requested:  # Check if a stop has been requested
                    print("Stop requested during validation")
                    break  # Exit the loop if stop is requested

                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred, (h_s, h_c) = model(X_batch, h_s, h_c)
                y_pred = y_pred.squeeze(-1)

                loss = self.criterion(y_pred, y_batch)
                total_loss += loss.item() * X_batch.size(0)
                total_samples += X_batch.size(0)

                # Log the validation progress for each batch
                if batch_idx % 150 == 0:  # For example, every 150 batches
                    self.logger.info(f"Epoch {epoch}, Batch {batch_idx} | Loss: {loss.item()}")
                    print(f"Epoch: {epoch}, Batch: {batch_idx}, Input shape: {X_batch.shape}")
                    print(f"Epoch: {epoch}, Batch: {batch_idx}, Output shape after LSTM: {y_pred.shape}")
                # Clear unused memory
                del X_batch, y_batch, y_pred  # Explicitly clear tensors

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

    def get_scheduler(self, optimizer, lr_drop_period):
        """Initialize the learning rate scheduler."""
        # Create a learning rate scheduler that reduces the LR by 10% every lr_drop_period epochs
        return optim.lr_scheduler.StepLR(optimizer, step_size=lr_drop_period, gamma=0.1)


    #Refactored codes
    
    # def train_epoch(self, model, train_loader, optimizer):
    #     """Train the model for a single epoch."""
    #     model.train()  # Set the model to training mode
    #     running_loss = 0.0

    #     for inputs, targets in train_loader:
    #         optimizer.zero_grad()  # Clear previous gradients

    #         outputs, _ = model(inputs)  # Forward pass
    #         loss = self.criterion(outputs, targets)  # Calculate loss
    #         loss.backward()  # Backward pass
    #         optimizer.step()  # Update weights

    #         running_loss += loss.item() * inputs.size(0)  # Accumulate loss

    #     epoch_loss = running_loss / len(train_loader.dataset)  # Average loss per sample
    #     return epoch_loss

    # def validate_epoch(self, model, val_loader):
    #     """Validate the model for a single epoch."""
    #     model.eval()  # Set the model to evaluation mode
    #     running_loss = 0.0

    #     with torch.no_grad():  # Disable gradient calculation
    #         for inputs, targets in val_loader:
    #             outputs, _ = model(inputs)  # Forward pass
    #             loss = self.criterion(outputs, targets)  # Calculate loss
    #             running_loss += loss.item() * inputs.size(0)  # Accumulate loss

    #     epoch_loss = running_loss / len(val_loader.dataset)  # Average loss per sample
    #     return epoch_loss
