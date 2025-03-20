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



class LSTMModelBN(nn.Module):
    def __init__(self, input_size, hidden_units, num_layers, device):
        super().__init__()
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.device = device  

        # BatchNorm for input features
        self.batch_norm_input = nn.BatchNorm1d(input_size)  

        # LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_units,
                            num_layers=num_layers, batch_first=True)  

        # BatchNorm for LSTM outputs
        self.batch_norm_lstm = nn.BatchNorm1d(hidden_units)  

        # Linear layer
        self.linear = nn.Linear(hidden_units, 1)  

        # Activation functions
        self.ReLU = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x, h_s, h_c):
        x = x.to(self.device)  

        # **Apply BatchNorm to input features** (requires permute)
        x = x.permute(0, 2, 1)  # Move sequence length to last
        x = self.batch_norm_input(x)
        x = x.permute(0, 2, 1)  # Move back sequence length to second dim

        # LSTM forward pass
        y, (h_s, h_c) = self.lstm(x, (h_s, h_c))

        # **Apply BatchNorm to LSTM outputs** (normalize along hidden units)
        y = y.permute(0, 2, 1)  # Move sequence length to last
        y = self.batch_norm_lstm(y)
        y = y.permute(0, 2, 1)  # Move back sequence length to second dim

        # Pass through linear layer
        y = self.linear(y)

        return y, (h_s, h_c)


class LSTMModelLN(nn.Module):
    def __init__(self, input_size, hidden_units, num_layers, device):
        super().__init__()
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.device = device

        # Layernorm on input features
        #self.layer_norm_input = nn.LayerNorm(input_size)  

        # LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_units,
                            num_layers=num_layers, batch_first=True, dropout=0.2)  

        # LayerNorm on hidden states
        self.layer_norm = nn.LayerNorm(hidden_units)

        # Linear layer
        self.linear = nn.Linear(hidden_units, 1)  

    def forward(self, x, h_s, h_c):
        x = x.to(self.device)  

        # Apply LayerNorm to input features
        #x = self.layer_norm_input(x)

        # LSTM forward pass
        y, (h_s, h_c) = self.lstm(x, (h_s, h_c))

        # **Apply LayerNorm on LSTM hidden states**
        y = self.layer_norm(y)

        # Pass through linear layer
        y = self.linear(y)

        return y, (h_s, h_c)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_units, num_layers, device):
        super().__init__()
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.device = device  # Store the device in the model
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_units,
                            num_layers=num_layers, batch_first=True)  # Match definition with VEstimLSTM
        self.linear = nn.Linear(hidden_units, 1)  # Renamed from 'fc' to 'linear' to match VEstimLSTM
        self.ReLU = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x, h_s, h_c):
        # Ensure input is on the correct device
        x = x.to(self.device)  

        # Pass through LSTM
        y, (h_s, h_c) = self.lstm(x, (h_s, h_c))

        # **Apply LayerNorm to stabilize hidden states**
        #y = self.layer_norm(y)

        # Pass through Linear layer (FC)
        y = self.linear(y)

        # Activation functions (comment/uncomment based on need)
        # y = torch.clamp(y, 0, 1)    # Clipped ReLU layer
        # y = self.LeakyReLU(y)

        return y, (h_s, h_c)

class CustomLSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_units):
        super().__init__()
        self.input_size = input_size
        self.hidden_units = hidden_units

        # LSTM parameters
        self.weight_ih = nn.Parameter(torch.Tensor(4 * hidden_units, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(4 * hidden_units, hidden_units))
        self.bias = nn.Parameter(torch.Tensor(4 * hidden_units))

        # LayerNorm for hidden states
        self.layer_norm = nn.LayerNorm(hidden_units)

        # Initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.xavier_uniform_(self.weight_hh)
        nn.init.zeros_(self.bias)

    def forward(self, x, h_s, h_c):
        # LSTM equations
        gates = (x @ self.weight_ih.t()) + (h_s @ self.weight_hh.t()) + self.bias
        i, f, g, o = gates.chunk(4, 1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        # Update cell state
        h_c = f * h_c + i * g

        # Apply LayerNorm to cell state
        h_c = self.layer_norm(h_c)

        # Update hidden state
        h_s = o * torch.tanh(h_c)

        return h_s, h_c


class LSTMModel_CustomLN(nn.Module):
    def __init__(self, input_size, hidden_units, num_layers, device):
        super().__init__()
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.device = device

        # Custom LSTM layers
        self.lstm_layers = nn.ModuleList([
            CustomLSTMLayer(input_size if i == 0 else hidden_units, hidden_units)
            for i in range(num_layers)
        ])

        # Linear layer
        self.linear = nn.Linear(hidden_units, 1)

    def forward(self, x, h_s, h_c):
        x = x.to(self.device)

        # Forward pass through each LSTM layer
        for i, lstm_layer in enumerate(self.lstm_layers):
            x, (h_s[i], h_c[i]) = lstm_layer(x, h_s[i], h_c[i])

        # Pass through linear layer
        y = self.linear(x)

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
    
    def build_lstm_model_LN(self, params):
        """
        Build the LSTM model using the provided parameters.
        
        :param params: Dictionary containing model parameters.
        :return: An instance of LSTMModel.
        """
        input_size = params.get("INPUT_SIZE", 3)  # Default input size set to 3, change if needed
        hidden_units = int(params["HIDDEN_UNITS"])  # Ensure hidden_units is an integer
        num_layers = int(params["LAYERS"])
        
        print(f"Building LSTM model with input_size={input_size}, hidden_units={hidden_units}, num_layers={num_layers}")
        return LSTMModelLN(input_size, hidden_units, num_layers, self.device)
    
    def build_lstm_model_CustomLN(self, params):
        """
        Build the LSTM model using the provided parameters.
        
        :param params: Dictionary containing model parameters.
        :return: An instance of LSTMModel.
        """
        input_size = params.get("INPUT_SIZE", 3)  # Default input size set to 3, change if needed
        hidden_units = int(params["HIDDEN_UNITS"])  # Ensure hidden_units is an integer
        num_layers = int(params["LAYERS"])
        
        print(f"Building LSTM model with input_size={input_size}, hidden_units={hidden_units}, num_layers={num_layers}")
        return LSTMModel_CustomLN(input_size, hidden_units, num_layers, self.device)

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

    def create_and_save_lstm_model_with_LN(self, params, model_path):
        """
        Build and save an LSTM model using the provided parameters.
        
        :param params: Dictionary containing model parameters.
        :param model_path: The file path where the model will be saved.
        :return: The built LSTM model.
        """
        model = self.build_lstm_model_LN(params)
        self.save_model(model, model_path)
        return model
    