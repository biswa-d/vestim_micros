import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from vestim.services.model_training.src.LSTM_model import LSTMModel
from vestim.services.model_training.src.LSTM_model_filterable import LSTM_EMA, LSTM_LPF

class LSTMModelService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_lstm_model(self, params, model_type="LSTM", device=None):
        """
        Build the LSTM model using the provided parameters.

        :param params: Dictionary containing model parameters.
        :param model_type: The type of model to build (e.g., "LSTM", "LSTM_EMA", "LSTM_LPF").
        :param device: The target device for the model.
        :return: An instance of the specified model.
        """
        target_device = device if device is not None else self.device
        input_size = params.get("INPUT_SIZE", 3)
        hidden_units = int(params["HIDDEN_UNITS"])
        num_layers = int(params["LAYERS"])
        dropout_prob = params.get("DROPOUT_PROB", 0.5)

        apply_clipped_relu = params.get("normalization_applied", False)
        use_layer_norm = params.get("LSTM_USE_LAYERNORM", False)
        print(f"Building {model_type} model with input_size={input_size}, hidden_units={hidden_units}, "
              f"num_layers={num_layers}, dropout_prob={dropout_prob}, device={target_device}, "
              f"apply_clipped_relu={apply_clipped_relu}, use_layer_norm={use_layer_norm}")

        model_class = {
            "LSTM": LSTMModel,
            "LSTM_EMA": LSTM_EMA,
            "LSTM_LPF": LSTM_LPF,
        }.get(model_type, LSTMModel)

        model = model_class(
            input_size=input_size,
            hidden_units=hidden_units,
            num_layers=num_layers,
            device=target_device,
            dropout_prob=dropout_prob,
            apply_clipped_relu=apply_clipped_relu,
            use_layer_norm=use_layer_norm
        )

        return model

    def create_model(self, params, model_type="LSTM", device=None):
        """
        Create an LSTM model in-memory without saving it.

        :param params: Dictionary containing model parameters.
        :param model_type: The type of model to create.
        :param device: The target device for the model.
        :return: An instance of the specified model.
        """
        return self.build_lstm_model(params, model_type=model_type, device=device)
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
        model_type = params.get("MODEL_TYPE", "LSTM")
        model = self.build_lstm_model(params, model_type=model_type, device=device)
        self.save_model(model, model_path)
        return model
