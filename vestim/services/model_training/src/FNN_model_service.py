import torch
import json
import logging
from vestim.services.model_training.src.FNN_model import FNNModel

class FNNModelService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_fnn_model(self, params: dict, trial=None, device=None):
        """
        Build an FNN model using the provided, fully-resolved parameters.
        The parameters are expected to be concrete values, not search ranges.
        
        :param params: Dictionary containing resolved model parameters.
        :param trial: An Optuna trial object (optional, for context).
        :param device: The target device for the model.
        :return: An instance of FNNModel.
        """
        target_device = device if device is not None else self.device
        self.logger.debug(f"Building FNN model with received params: {params}")

        input_size = params.get("INPUT_SIZE", 3)
        output_size = params.get("OUTPUT_SIZE", 1)

        # Directly use the resolved hyperparameters from the params dictionary.
        # The GUI is now responsible for suggesting the values during an Optuna trial.
        hidden_layer_sizes = params.get("FNN_UNITS")
        if not hidden_layer_sizes:
            # Fallback for backward compatibility or standard runs
            hidden_layer_sizes = params.get("HIDDEN_LAYER_SIZES")

        dropout_prob = params.get("FNN_DROPOUT_PROB", params.get("DROPOUT_PROB", 0.0))

        if not hidden_layer_sizes:
            self.logger.error("FNN hidden layer configuration is missing or invalid.")
            raise ValueError("Cannot build FNN model without hidden layer sizes.")

        self.logger.info(
            f"Building FNN model with input_size={input_size}, output_size={output_size}, "
            f"hidden_layers={hidden_layer_sizes}, dropout_prob={dropout_prob}, device={target_device}"
        )

        apply_clipped_relu = params.get("normalization_applied", False)
        
        model = FNNModel(
            input_size=input_size,
            output_size=output_size,
            hidden_layer_sizes=hidden_layer_sizes,
            dropout_prob=dropout_prob,
            apply_clipped_relu=apply_clipped_relu
        ).to(target_device)
        
        return model
    def create_model(self, params: dict, trial=None, device=None):
        """
        Create an FNN model in-memory, suitable for Optuna trials.

        :param params: Dictionary containing model parameters.
        :param trial: An Optuna trial object.
        :param device: The target device for the model.
        :return: An instance of FNNModel.
        """
        return self.build_fnn_model(params, trial=trial, device=device)

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