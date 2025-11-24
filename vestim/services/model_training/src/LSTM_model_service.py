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
        
        # INPUT_SIZE must be provided - it should equal the number of input features
        input_size = params.get("INPUT_SIZE")
        if input_size is None:
            raise ValueError(
                "INPUT_SIZE is missing from params. It must equal the number of input features in your data. "
                "Ensure FEATURE_COLUMNS is properly set in your hyperparameters."
            )
        
        # Parse RNN_LAYER_SIZES parameter (supports comma-separated like "64,32")
        rnn_layer_sizes = params.get("RNN_LAYER_SIZES") or params.get("LSTM_UNITS")
        if rnn_layer_sizes:
            # Parse the layer sizes string (e.g., "64,32" or "128,64,32")
            if isinstance(rnn_layer_sizes, str):
                layer_sizes = [int(x.strip()) for x in rnn_layer_sizes.split(',')]
            elif isinstance(rnn_layer_sizes, list):
                layer_sizes = [int(x) for x in rnn_layer_sizes]
            else:
                layer_sizes = [int(rnn_layer_sizes)]
            
            # For now, use only the first layer size (uniform layers)
            # Future: Support variable layers with custom stacked implementation
            hidden_units = layer_sizes[0]
            num_layers = len(layer_sizes)
        else:
            # Legacy: Fall back to HIDDEN_UNITS + LAYERS
            hidden_units = int(params.get("HIDDEN_UNITS", 10))
            num_layers = int(params.get("LAYERS", 1))
        
        # Use model-specific dropout parameter (LSTM_DROPOUT_PROB) with sensible default
        dropout_prob = float(params.get("LSTM_DROPOUT_PROB", params.get("DROPOUT_PROB", 0.2)))
        apply_clipped_relu = params.get("normalization_applied", False)
        use_layer_norm = params.get("LSTM_USE_LAYERNORM", False)
        
        print(f"Building {model_type} model with input_size={input_size}, hidden_units={hidden_units}, "
              f"num_layers={num_layers}, dropout_prob={dropout_prob}, device={target_device}, "
              f"apply_clipped_relu={apply_clipped_relu}, use_layer_norm={use_layer_norm}")

        # Choose model implementation based on type
        if model_type == "LSTM":
            model = LSTMModel(
                input_size=input_size,
                hidden_units=hidden_units,
                num_layers=num_layers,
                device=target_device,
                dropout_prob=dropout_prob,
                apply_clipped_relu=apply_clipped_relu,
                use_layer_norm=use_layer_norm
            )
        elif model_type == "LSTM_EMA":
            model = LSTM_EMA(
                input_size=input_size,
                hidden_units=hidden_units,
                num_layers=num_layers,
                device=target_device,
                dropout_prob=dropout_prob,
                apply_clipped_relu=apply_clipped_relu
            )
        elif model_type == "LSTM_LPF":
            model = LSTM_LPF(
                input_size=input_size,
                hidden_units=hidden_units,
                num_layers=num_layers,
                device=target_device,
                dropout_prob=dropout_prob,
                apply_clipped_relu=apply_clipped_relu
            )
        else:
            model = LSTMModel(
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
