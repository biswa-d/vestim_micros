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
        # x is expected to be on the correct device already by the caller (TrainingTaskService)
        # x = x.to(self.device) # REMOVED: This was causing the device mismatch.

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

        # LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_units,
                            num_layers=num_layers, batch_first=True)  

        # LayerNorm on hidden states
        self.layer_norm = nn.LayerNorm(hidden_units)

        # Linear layer
        self.linear = nn.Linear(hidden_units, 1)  

    def forward(self, x, h_s, h_c):
        # x is expected to be on the correct device already by the caller (TrainingTaskService)
        # x = x.to(self.device) # REMOVED: This was causing the device mismatch.

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
        # Ensure input is on the correct device - This is now handled by the caller (TrainingTaskService)
        # x = x.to(self.device) # REMOVED: This was causing the device mismatch.

        # Pass through LSTM
        y, (h_s, h_c) = self.lstm(x, (h_s, h_c))

        # **Apply LayerNorm to stabilize hidden states**
        #y = self.layer_norm(y)

        # Pass through Linear layer (FC)
        # We only want the output of the last time step for sequence-to-one prediction
        y = self.linear(y[:, -1, :])

        # Activation functions (comment/uncomment based on need)
        # y = torch.clamp(y, 0, 1)    # Clipped ReLU layer
        # y = self.LeakyReLU(y)

        return y, (h_s, h_c)


class LSTMModelService:
    def __init__(self):
        # self.device can be a fallback if no device is specified during model creation,
        # but ideally, the device should always be passed in.
        self.default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_lstm_model(self, params, target_device=None):
        """
        Build the LSTM model using the provided parameters and target device.
        
        :param params: Dictionary containing model parameters.
        :param target_device: The torch.device to build the model on.
        :return: An instance of LSTMModel.
        """
        input_size = params.get("INPUT_SIZE", 3)
        hidden_units = int(params["HIDDEN_UNITS"])
        num_layers = int(params["LAYERS"])
        device_to_use = target_device if target_device is not None else self.default_device
        
        print(f"Building LSTM model with input_size={input_size}, hidden_units={hidden_units}, num_layers={num_layers} on device: {device_to_use}")
        return LSTMModel(input_size, hidden_units, num_layers, device_to_use)
    
    def build_lstm_model_LN(self, params, target_device=None):
        """
        Build the LSTM model with LayerNorm using the provided parameters and target device.
        
        :param params: Dictionary containing model parameters.
        :param target_device: The torch.device to build the model on.
        :return: An instance of LSTMModelLN.
        """
        input_size = params.get("INPUT_SIZE", 3)
        hidden_units = int(params["HIDDEN_UNITS"])
        num_layers = int(params["LAYERS"])
        device_to_use = target_device if target_device is not None else self.default_device
        
        print(f"Building LSTMModelLN with input_size={input_size}, hidden_units={hidden_units}, num_layers={num_layers} on device: {device_to_use}")
        return LSTMModelLN(input_size, hidden_units, num_layers, device_to_use)

    def save_model(self, model, model_path):
        """
        Save the model to the specified path.
        
        :param model: The PyTorch model to save.
        :param model_path: The file path where the model will be saved.
        """
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    def create_and_save_lstm_model(self, params, model_path, target_device=None):
        """
        Build and save an LSTM model using the provided parameters and target device.
        
        :param params: Dictionary containing model parameters.
        :param model_path: The file path where the model will be saved.
        :param target_device: The torch.device to build the model on.
        :return: The built LSTM model.
        """
        model = self.build_lstm_model(params, target_device)
        self.save_model(model, model_path)
        return model

    def create_and_save_lstm_model_with_LN(self, params, model_path, target_device=None):
        """
        Build and save an LSTM model with LayerNorm using the provided parameters and target device.
        
        :param params: Dictionary containing model parameters.
        :param model_path: The file path where the model will be saved.
        :param target_device: The torch.device to build the model on.
        :return: The built LSTM model.
        """
        model = self.build_lstm_model_LN(params, target_device)
        self.save_model(model, model_path)
        return model