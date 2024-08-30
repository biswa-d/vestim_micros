import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_units, num_layers):
        super(LSTMModel, self).__init__()
        self.input_size = input_size  # Store input_size
        self.hidden_units = hidden_units  # Store hidden_units
        self.num_layers = num_layers  # Store num_layers
        self.lstm = nn.LSTM(input_size, hidden_units, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_units, 1)  # Assuming regression, adjust for your use case

    def forward(self, x, h_s=None, h_c=None):
        # If h_s and h_c are not provided, initialize them
        if h_s is None or h_c is None:
            h_s = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
            h_c = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        
        out, (h_s, h_c) = self.lstm(x, (h_s, h_c))
        out = self.fc(out[:, -1, :])
        return out, (h_s, h_c)

class LSTMModelService:
    def __init__(self):
        pass

    def build_lstm_model(self, params):
        """
        Build the LSTM model using the provided parameters.
        
        :param params: Dictionary containing model parameters.
        :return: An instance of LSTMModel.
        """
        input_size = params.get("INPUT_SIZE", 3)  # Default input size set to 3, change if needed
        hidden_units = int(params["HIDDEN_UNITS"])  # Ensure hidden_units is an integer
        num_layers = int(params["LAYERS"])
        
        print(f"Building LSTM model with input_size={input_size}, hidden_units={hidden_units}, num_layers={num_layers}")
        return LSTMModel(input_size, hidden_units, num_layers)

    def save_model(self, model, model_path):
        """
        Save the model to the specified path.
        
        :param model: The PyTorch model to save.
        :param model_path: The file path where the model will be saved.
        """
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    def create_and_save_lstm_model(self, params, model_path):
        """
        Build and save an LSTM model using the provided parameters.
        
        :param params: Dictionary containing model parameters.
        :param model_path: The file path where the model will be saved.
        :return: The built LSTM model.
        """
        model = self.build_lstm_model(params)
        self.save_model(model, model_path)
        return model
