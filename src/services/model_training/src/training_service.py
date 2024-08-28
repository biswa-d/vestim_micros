import torch
import torch.nn as nn
import torch.optim as optim

class TrainingService:
    def start_training(self, model, data_loader, hyper_params, update_progress):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=hyper_params['INITIAL_LR'])

        for epoch in range(1, hyper_params['MAX_EPOCHS'] + 1):
            model.train()
            total_train_loss = 0.0

            for X_batch, y_batch in data_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            train_loss_avg = total_train_loss / len(data_loader)
            
            if epoch % hyper_params['ValidFrequency'] == 0 or epoch == hyper_params['MAX_EPOCHS']:
                validation_error = self.validate_model(model, data_loader, criterion, device)
                update_progress(epoch, train_loss_avg, validation_error)

        # Save the model after training
        self.save_trained_model(model, hyper_params)

    def validate_model(self, model, data_loader, criterion, device):
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                total_loss += loss.item()
        return total_loss / len(data_loader)

    def save_trained_model(self, model, hyper_params):
        # Logic to save the trained model
        pass
