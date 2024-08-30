from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.optim as optim
import time
from threading import Thread

app = Flask(__name__)

class TrainingService:
    def __init__(self):
        self.start_time = None
        self.progress_data = {}

    def start_training(self, model, data_loader, hyper_params):
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
                self.progress_data = {
                    'epoch': epoch,
                    'train_loss': train_loss_avg,
                    'validation_error': validation_error,
                    'elapsed_time': self.get_elapsed_time()
                }

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
        # Save logic here
        pass

    def get_elapsed_time(self):
        elapsed_time = time.time() - self.start_time
        return self._format_time(elapsed_time)

    def _format_time(self, elapsed_time):
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02}h:{int(minutes):02}m:{int(seconds):02}s"

training_service = TrainingService()

@app.route('/start_training', methods=['POST'])
def start_training():
    data = request.get_json()
    model = data['model']
    data_loader = data['data_loader']
    hyper_params = data['hyper_params']
    thread = Thread(target=training_service.start_training, args=(model, data_loader, hyper_params))
    thread.start()
    return jsonify({"message": "Training started"}), 200

@app.route('/get_progress', methods=['GET'])
def get_progress():
    return jsonify(training_service.progress_data), 200

if __name__ == '__main__':
    app.run(port=5004, debug=True)
