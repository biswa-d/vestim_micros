import torch
import logging
from vestim.services.model_training.src.GRU_model import GRUModel

class GRUModelService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_gru_model(self, params: dict, device=None):
        """
        Build a GRU model using the provided parameters.

        :param params: Dictionary containing model parameters. Expected keys:
                       "INPUT_SIZE": int,
                       "RNN_LAYER_SIZES" or "GRU_UNITS": str (e.g., "64,32") or list,
                       OR legacy "HIDDEN_UNITS" + "LAYERS",
                       "OUTPUT_SIZE": int (optional, default 1),
                       "DROPOUT_PROB": float (optional, default 0.0)
        :param device: The target device for the model.
        :return: An instance of the specified GRU model.
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
        rnn_layer_sizes = params.get("RNN_LAYER_SIZES") or params.get("GRU_UNITS")
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
            # Legacy: Fall back to GRU_HIDDEN_UNITS + GRU_LAYERS or HIDDEN_UNITS + LAYERS
            hidden_units = int(params.get("GRU_HIDDEN_UNITS") or params.get("HIDDEN_UNITS", 10))
            num_layers = int(params.get("GRU_LAYERS") or params.get("LAYERS", 1))
        
        output_size = params.get("OUTPUT_SIZE", 1)
        dropout_prob = params.get("DROPOUT_PROB", 0.0)
        apply_clipped_relu = params.get("normalization_applied", False)
        use_layer_norm = params.get("GRU_USE_LAYERNORM", False)
        
        self.logger.info(
            f"Building GRU model with input_size={input_size}, hidden_units={hidden_units}, "
            f"num_layers={num_layers}, output_size={output_size}, dropout_prob={dropout_prob}, device={target_device}, "
            f"apply_clipped_relu={apply_clipped_relu}, use_layer_norm={use_layer_norm}"
        )

        model = GRUModel(
            input_size=input_size,
            hidden_units=hidden_units,
            num_layers=num_layers,
            output_size=output_size,
            dropout_prob=dropout_prob,
            device=target_device,
            apply_clipped_relu=apply_clipped_relu,
            use_layer_norm=use_layer_norm
        ).to(target_device)
        
        return model

    def create_model(self, params: dict, device=None):
        """
        Create a GRU model in-memory without saving it.

        :param params: Dictionary containing model parameters.
        :param device: The target device for the model.
        :return: An instance of GRUModel.
        """
        return self.build_gru_model(params, device=device)

    def save_model(self, model: GRUModel, model_path: str):
        """
        Save the GRU model to the specified path.

        :param model: The PyTorch GRUModel to save.
        :param model_path: The file path where the model will be saved.
        """
        torch.save(model.state_dict(), model_path)
        self.logger.info(f"GRU Model saved to {model_path}")

    def create_and_save_gru_model(self, params: dict, model_path: str, target_device=None):
        """
        Build and save a GRU model using the provided parameters.

        :param params: Dictionary containing model parameters.
        :param model_path: The file path where the model will be saved.
        :param target_device: Target device for the model (optional, will override self.device if provided).
        :return: The built GRUModel.
        """
        # Update device if target_device is specified
        if target_device is not None:
            self.device = target_device
            self.logger.info(f"GRU model service device updated to: {target_device}")
        
        model = self.build_gru_model(params)
        self.save_model(model, model_path)
        return model