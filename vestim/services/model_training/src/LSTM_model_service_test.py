#inconsistency with original model noted and being corrected with this test script

import torch
import torch.nn as nn

class VEstimLSTM(nn.Module):
    def __init__(self, hidden_size, input_size, layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=layers, batch_first=True) #we have out data from the datacreate method arranged in (batches,  sequence, features) 
        self.linear = nn.Linear(hidden_size, 1)
        self.ReLU = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.1)
        # self.h_s = None
        # self.h_c = None

    def forward(self, x, h_s, h_c):
        # The h_s, h_c is defaulted to 0 every time, so only remember last 500-second data
        y, (h_s, h_c) = self.lstm(x, (h_s, h_c))
        y = self.linear(y)
        # y = torch.clamp(y, 0, 1)    # Clipped ReLU layer
        # y = self.LeakyReLU(y)
        return y, (h_s, h_c)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_units, num_layers, device):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_units,
                            num_layers=num_layers, batch_first=True)  # Match definition with VEstimLSTM
        self.linear = nn.Linear(hidden_units, 1)  # Renamed from 'fc' to 'linear' to match VEstimLSTM
        self.ReLU = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x, h_s, h_c):
        # Ensure input is on the correct device
        x = x.to(self.linear.weight.device)  

        # Pass through LSTM
        y, (h_s, h_c) = self.lstm(x, (h_s, h_c))

        # Pass through Linear layer (FC)
        y = self.linear(y)

        # Activation functions (comment/uncomment based on need)
        # y = torch.clamp(y, 0, 1)    # Clipped ReLU layer
        # y = self.LeakyReLU(y)

        return y, (h_s, h_c)


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
