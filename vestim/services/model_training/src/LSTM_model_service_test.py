import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_units, num_layers, device):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.device = device  # Store the device in the model
        self.lstm = nn.LSTM(input_size, hidden_units, num_layers, batch_first=True).to(self.device)
        self.fc = nn.Linear(hidden_units, 1).to(self.device)  # Assuming regression, adjust if needed

    def forward(self, x, h_s=None, h_c=None):
        # Ensure the input is on the correct device
        x = x.to(self.device)
        
        # Debugging: Print input shape
        # print(f"Input shape: {x.shape}")
        
        # Initialize hidden and cell states if not provided
        if h_s is None or h_c is None:
            h_s = torch.zeros(self.lstm.num_layers, x.size(0), self.hidden_units).to(self.device)
            h_c = torch.zeros(self.lstm.num_layers, x.size(0), self.hidden_units).to(self.device)
        
        # Pass input through LSTM
        out, (h_s, h_c) = self.lstm(x, (h_s, h_c))
        
        # Debugging: Print output shape after LSTM
        # print(f"Output shape after LSTM: {out.shape}")
        
        # # Check if output shape is as expected before slicing
        # if len(out.shape) == 3:
        #     out = out[:, -1, :]  # Get the last time step's output
        # else:
        #     raise ValueError(f"Unexpected output shape from LSTM: {out.shape}")
        
        # Pass the LSTM output through the fully connected layer
        out = self.fc(out)
        return out, (h_s, h_c)


class LSTMModelService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        return LSTMModel(input_size, hidden_units, num_layers, self.device)

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
