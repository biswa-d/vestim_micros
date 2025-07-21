import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from vestim.services.model_training.src.LSTM_model import LSTMModel

class LSTMModelService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_lstm_model(self, params, device=None):
        """
        Build the LSTM model using the provided parameters.

        :param params: Dictionary containing model parameters.
        :param device: The target device for the model.
        :return: An instance of LSTMModel.
        """
        target_device = device if device is not None else self.device
        input_size = params.get("INPUT_SIZE", 3)
        hidden_units = int(params["HIDDEN_UNITS"])
        num_layers = int(params["LAYERS"])
        dropout_prob = params.get("DROPOUT_PROB", 0.5)

        apply_clipped_relu = params.get("normalization_applied", False)
        print(f"Building LSTM model with input_size={input_size}, hidden_units={hidden_units}, "
              f"num_layers={num_layers}, dropout_prob={dropout_prob}, device={target_device}, "
              f"apply_clipped_relu={apply_clipped_relu}")

        # Create an instance of the refactored LSTMModel
        model = LSTMModel(
            input_size=input_size,
            hidden_units=hidden_units,
            num_layers=num_layers,
            device=target_device,
            dropout_prob=dropout_prob,
            apply_clipped_relu=apply_clipped_relu
        )

        return model

    def create_model(self, params, device=None):
        """
        Create an LSTM model in-memory without saving it.

        :param params: Dictionary containing model parameters.
        :param device: The target device for the model.
        :return: An instance of LSTMModel.
        """
        return self.build_lstm_model(params, device=device)
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

    def create_and_save_lstm_model(self, params, model_path, device=None):
        """
        Build and save an LSTM model using the provided parameters.

        :param params: Dictionary containing model parameters.
        :param model_path: The file path where the model will be saved.
        :param device: The target device for the model.
        :return: The built LSTM model.
        """
        model = self.build_lstm_model(params, device=device)
        self.save_model(model, model_path)
        return model