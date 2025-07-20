import torch
import logging
from vestim.services.model_training.src.FNN_model import FNNModel

class FNNModelService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_fnn_model(self, params: dict):
        """
        Build an FNN model using the provided parameters.

        :param params: Dictionary containing model parameters. Expected keys:
                       "INPUT_SIZE": int,
                       "OUTPUT_SIZE": int (typically 1),
                       "HIDDEN_LAYER_SIZES": list of int (e.g., [128, 64]),
                       "DROPOUT_PROB": float (optional, default 0.0)
        :return: An instance of FNNModel.
        """
        input_size = params.get("INPUT_SIZE")
        if input_size is None and "FEATURE_COLUMNS" in params:
            input_size = len(params["FEATURE_COLUMNS"])
            params["INPUT_SIZE"] = input_size
        output_size = params.get("OUTPUT_SIZE", 1) # Default to 1 output neuron
        hidden_layer_sizes = params.get("HIDDEN_LAYER_SIZES")
        dropout_prob = params.get("DROPOUT_PROB", 0.0)

        if input_size is None:
            self.logger.error("INPUT_SIZE is required to build FNNModel.")
            raise ValueError("INPUT_SIZE is required for FNNModel.")
        if hidden_layer_sizes is None or not isinstance(hidden_layer_sizes, list):
            self.logger.error("HIDDEN_LAYER_SIZES (list of ints) is required for FNNModel.")
            raise ValueError("HIDDEN_LAYER_SIZES (list of ints) is required for FNNModel.")

        self.logger.info(
            f"Building FNN model with input_size={input_size}, output_size={output_size}, "
            f"hidden_layers={hidden_layer_sizes}, dropout_prob={dropout_prob}, device={self.device}"
        )

        model = FNNModel(
            input_size=input_size,
            output_size=output_size,
            hidden_layer_sizes=hidden_layer_sizes,
            dropout_prob=dropout_prob
        ).to(self.device)
        
        return model

    def create_model_for_optuna(self, params: dict, device=None):
        """
        Create an FNN model in-memory for Optuna trials.
        Calculates INPUT_SIZE from FEATURE_COLUMNS.
        """
        if "INPUT_SIZE" not in params and "FEATURE_COLUMNS" in params:
            params["INPUT_SIZE"] = len(params["FEATURE_COLUMNS"])
        
        if device is not None:
            self.device = device
        return self.build_fnn_model(params)

    def create_model(self, params: dict, device=None):
        """
        Create an FNN model in-memory without saving it.

        :param params: Dictionary containing model parameters.
        :param device: The target device for the model.
        :return: An instance of FNNModel.
        """
        if device is not None:
            self.device = device
        return self.build_fnn_model(params)

    def save_model(self, model: FNNModel, model_path: str):
        """
        Save the FNN model to the specified path.

        :param model: The PyTorch FNNModel to save.
        :param model_path: The file path where the model will be saved.
        """
        torch.save(model.state_dict(), model_path)
        self.logger.info(f"FNN Model saved to {model_path}")

    def create_and_save_fnn_model(self, params: dict, model_path: str, target_device=None):
        """
        Build and save an FNN model using the provided parameters.

        :param params: Dictionary containing model parameters.
        :param model_path: The file path where the model will be saved.
        :param target_device: Target device (for consistency with other model services).
        :return: The built FNNModel.
        """
        if target_device is not None:
            self.device = target_device
        model = self.build_fnn_model(params)
        self.save_model(model, model_path)
        return model