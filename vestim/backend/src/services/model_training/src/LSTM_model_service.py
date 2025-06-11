import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_units, num_layers, device, dropout_prob=0.0):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.device = device  # Store the device in the model
        self.dropout_prob = dropout_prob

        # Define the LSTM layer with dropout between layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_units,
            num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0  # Dropout only applied if num_layers > 1
        ).to(self.device)

        # Define a dropout layer for the outputs
        self.dropout = nn.Dropout(p=dropout_prob)

        # Define the fully connected layer
        self.fc = nn.Linear(hidden_units, 1).to(self.device)  

    def forward(self, x, h_s=None, h_c=None):
        # Ensure the input is on the correct device
        x = x.to(self.device)

        # Pass input through LSTM
        out, (h_s, h_c) = self.lstm(x, (h_s, h_c))

        # Apply dropout to the outputs of the LSTM
        out = self.dropout(out)

        # Pass the output through the fully connected layer
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
        dropout_prob = params.get("DROPOUT_PROB", 0.5)  # Default dropout probability

        print(f"Building LSTM model with input_size={input_size}, hidden_units={hidden_units}, "
              f"num_layers={num_layers}, dropout_prob={dropout_prob}")

        # Create an instance of the refactored LSTMModel
        model = LSTMModel(
            input_size=input_size,
            hidden_units=hidden_units,
            num_layers=num_layers,
            device=self.device,
            dropout_prob=dropout_prob
        )

        return model

    def save_model(self, model, model_path):
        """
        Save the model to the specified path after removing pruning reparameterizations.

        :param model: The PyTorch model to save.
        :param model_path: The file path where the model will be saved.
        """
        # Remove pruning reparameterizations before saving
        # model.remove_pruning()
        # Save the model's state dictionary
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