import torch
import torch.nn as nn
import torch.optim as optim
import json

class TrainingService:
    def __init__(self):
        self.criterion = nn.MSELoss()  # Assuming you're using Mean Squared Error Loss for regression tasks

    def train_epoch(self, model, train_loader, optimizer):
        """Train the model for a single epoch."""
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for inputs, targets in train_loader:
            optimizer.zero_grad()  # Clear previous gradients

            outputs = model(inputs)  # Forward pass
            loss = self.criterion(outputs, targets)  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            running_loss += loss.item() * inputs.size(0)  # Accumulate loss

        epoch_loss = running_loss / len(train_loader.dataset)  # Average loss per sample
        return epoch_loss

    def validate_epoch(self, model, val_loader):
        """Validate the model for a single epoch."""
        model.eval()  # Set the model to evaluation mode
        running_loss = 0.0

        with torch.no_grad():  # Disable gradient calculation
            for inputs, targets in val_loader:
                outputs = model(inputs)  # Forward pass
                loss = self.criterion(outputs, targets)  # Calculate loss
                running_loss += loss.item() * inputs.size(0)  # Accumulate loss

        epoch_loss = running_loss / len(val_loader.dataset)  # Average loss per sample
        return epoch_loss

    def save_model(self, model, model_path):
        """Save the model to disk."""
        torch.save(model.state_dict(), model_path)

        # Save hyperparameters as well
        with open(model_path + '_hyperparams.json', 'w') as f:
            json.dump(model.hyperparams, f, indent=4)

    def _get_optimizer(self, model, lr):
        """Initialize the optimizer for the model."""
        return optim.Adam(model.parameters(), lr=lr)

    def _get_scheduler(self, optimizer, lr_drop_period):
        """Initialize the learning rate scheduler."""
        # Create a learning rate scheduler that reduces the LR by 10% every lr_drop_period epochs
        return optim.lr_scheduler.StepLR(optimizer, step_size=lr_drop_period, gamma=0.1)
