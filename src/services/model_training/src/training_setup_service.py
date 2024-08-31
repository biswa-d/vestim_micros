import torch
import torch.nn as nn
import torch.optim as optim
import time

class TrainingSetupService:
    def __init__(self):
        self.start_time = None
        self.progress_data = {}

    def start_training(self, model, train_loader, val_loader, hyper_params, progress_callback, model_dir):
        """
        Start the training process for a given model, with provided training and validation loaders.
        :param model: The model to train.
        :param train_loader: DataLoader for training data.
        :param val_loader: DataLoader for validation data.
        :param hyper_params: Dictionary of hyperparameters for training.
        :param progress_callback: Callback function to update progress.
        :param model_dir: Directory to save the trained model.
        """
        self.start_time = time.time()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        model = model.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=hyper_params['INITIAL_LR'])

        for epoch in range(1, hyper_params['MAX_EPOCHS'] + 1):
            model.train()
            total_train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            train_loss_avg = total_train_loss / len(train_loader)

            if epoch % hyper_params['ValidFrequency'] == 0 or epoch == hyper_params['MAX_EPOCHS']:
                validation_error = self.validate_model(model, val_loader, criterion, device)
                self.progress_data = {
                    'epoch': epoch,
                    'train_loss': train_loss_avg,
                    'validation_error': validation_error,
                    'elapsed_time': self.get_elapsed_time()
                }
                # Update progress using the callback function
                progress_callback(task=hyper_params, repetition=hyper_params['REPETITIONS'], epoch=epoch, train_loss=[train_loss_avg], validation_error=[validation_error])

        # Save the model after training
        self.save_trained_model(model, model_dir)

    def validate_model(self, model, val_loader, criterion, device):
        """
        Validate the model using the validation DataLoader.
        :param model: The model to validate.
        :param val_loader: DataLoader for validation data.
        :param criterion: Loss function.
        :param device: Device to use for validation (CPU/GPU).
        :return: Average validation loss.
        """
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                total_loss += loss.item()
        return total_loss / len(val_loader)

    def save_trained_model(self, model, model_dir):
        """
        Save the trained model to the specified directory.
        :param model: The trained model.
        :param model_dir: Directory where the model will be saved.
        """
        model_path = os.path.join(model_dir, 'final_model.pth')
        torch.save(model.state_dict(), model_path)

    def get_elapsed_time(self):
        """
        Calculate the elapsed time since the training started.
        :return: Formatted elapsed time as a string.
        """
        elapsed_time = time.time() - self.start_time
        return self._format_time(elapsed_time)

    def _format_time(self, elapsed_time):
        """
        Format time in hours, minutes, and seconds.
        :param elapsed_time: Time in seconds.
        :return: Formatted time as a string.
        """
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02}h:{int(minutes):02}m:{int(seconds):02}s"
