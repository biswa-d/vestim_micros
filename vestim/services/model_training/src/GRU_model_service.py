import torch
import logging
from vestim.services.model_training.src.GRU_model import GRUModel

class GRUModelService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_gru_model(self, params: dict):
        """
        Build a GRU model using the provided parameters.

        :param params: Dictionary containing model parameters. Expected keys:
                       "INPUT_SIZE": int,
                       "HIDDEN_UNITS": int,
                       "NUM_LAYERS": int,
                       "OUTPUT_SIZE": int (optional, default 1),
                       "DROPOUT_PROB": float (optional, default 0.0)
        :return: An instance of GRUModel.
        """
        input_size = params.get("INPUT_SIZE")
        hidden_units = params.get("HIDDEN_UNITS")
        num_layers = params.get("NUM_LAYERS")
        output_size = params.get("OUTPUT_SIZE", 1) # Default to 1 output neuron
        dropout_prob = params.get("DROPOUT_PROB", 0.0)

        if input_size is None:
            self.logger.error("INPUT_SIZE is required to build GRUModel.")
            raise ValueError("INPUT_SIZE is required for GRUModel.")
        if hidden_units is None:
            self.logger.error("HIDDEN_UNITS is required for GRUModel.")
            raise ValueError("HIDDEN_UNITS is required for GRUModel.")
        if num_layers is None:
            self.logger.error("NUM_LAYERS is required for GRUModel.")
            raise ValueError("NUM_LAYERS is required for GRUModel.")

        self.logger.info(
            f"Building GRU model with input_size={input_size}, hidden_units={hidden_units}, "
            f"num_layers={num_layers}, output_size={output_size}, dropout_prob={dropout_prob}, device={self.device}"
        )

        model = GRUModel(
            input_size=input_size,
            hidden_units=hidden_units,
            num_layers=num_layers,
            output_size=output_size,
            dropout_prob=dropout_prob,
            device=self.device # Pass device to model constructor
        ).to(self.device) # Ensure model is on the correct device
        
        return model
    def create_model(self, params: dict, device=None):
        """
        Create a GRU model in-memory without saving it.

        :param params: Dictionary containing model parameters.
        :param device: The target device for the model.
        :return: An instance of GRUModel.
        """
        # Update device if target_device is specified
        if device is not None:
            self.device = device
            self.logger.info(f"GRU model service device updated to: {device}")
        return self.build_gru_model(params)

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