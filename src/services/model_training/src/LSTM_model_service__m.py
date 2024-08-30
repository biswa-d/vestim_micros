import torch
import torch.nn as nn
import os

class LSTMModelService:
    def __init__(self):
        pass

    def create_lstm_model(self, input_size, output_size, layers, hidden_units):
        """
        Creates an LSTM model with the given configuration.

        :param input_size: The number of features in the input data.
        :param output_size: The number of output features.
        :param layers: The number of LSTM layers.
        :param hidden_units: A list containing the number of hidden units for each layer.
        :return: A PyTorch LSTM model.
        """
        assert len(hidden_units) == layers, "Length of hidden_units list should match the number of layers."

        class LSTMModel(nn.Module):
            def __init__(self, input_size, output_size, layers, hidden_units):
                super(LSTMModel, self).__init__()
                self.lstm_layers = nn.ModuleList()
                self.input_size = input_size
                self.output_size = output_size
                self.layers = layers

                for i in range(layers):
                    layer_input_size = input_size if i == 0 else hidden_units[i - 1]
                    self.lstm_layers.append(nn.LSTM(layer_input_size, hidden_units[i], batch_first=True))

                self.fc = nn.Linear(hidden_units[-1], output_size)

            def forward(self, x):
                h = x
                for i, lstm_layer in enumerate(self.lstm_layers):
                    h, _ = lstm_layer(h)

                # Pass the output of the last LSTM layer through the fully connected layer
                h = self.fc(h[:, -1, :])  # We are interested in the output of the last timestep
                return h

        return LSTMModel(input_size, output_size, layers, hidden_units)

    def save_model(self, model, model_path):
        """
        Saves the PyTorch model to a specified file path.

        :param model: The PyTorch model to save.
        :param model_path: The path where the model will be saved.
        """
        torch.save(model.state_dict(), model_path)

    def create_and_return_lstm_model(self, input_size, output_size, layers, hidden_units):
        """
        Creates an LSTM model with the specified configuration and returns it.

        :param input_size: The number of features in the input data.
        :param output_size: The number of output features.
        :param layers: The number of LSTM layers.
        :param hidden_units: A list containing the number of hidden units for each layer.
        :return: The created PyTorch LSTM model.
        """
        model = self.create_lstm_model(input_size, output_size, layers, hidden_units)
        return model
    