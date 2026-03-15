import torch
import logging
from vestim.services.model_training.src.NARX_model import NARXModel


class NARXModelService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_narx_model(self, params: dict, device=None):
        """
        Build a NARX model using the provided parameters.

        Expected keys:
            INPUT_SIZE: int
            OUTPUT_SIZE: int (default 1)
            HIDDEN_LAYER_SIZES or NARX_HIDDEN_LAYERS: list or comma-separated string
            OUTPUT_DELAY: int (default 1)
            DROPOUT_PROB: float (default 0.0)
            activation: str (default "ReLU")
            normalization_applied: bool (default False)
        """
        target_device = device if device is not None else self.device

        input_size = params.get("INPUT_SIZE")
        if input_size is None:
            raise ValueError(
                "INPUT_SIZE is missing from params. It must equal the number of input features in your data."
            )

        output_size = int(params.get("OUTPUT_SIZE", 1))

        hidden_layer_sizes = params.get("HIDDEN_LAYER_SIZES", params.get("NARX_HIDDEN_LAYERS", [128, 64]))
        if isinstance(hidden_layer_sizes, str):
            hidden_layer_sizes = [int(x.strip()) for x in hidden_layer_sizes.split(',') if x.strip()]
        elif isinstance(hidden_layer_sizes, int):
            hidden_layer_sizes = [hidden_layer_sizes]
        else:
            hidden_layer_sizes = [int(x) for x in hidden_layer_sizes]

        output_delay = int(params.get("OUTPUT_DELAY", 1))
        dropout_prob = float(params.get("DROPOUT_PROB", 0.0))
        activation_function = params.get("activation", "ReLU")
        apply_clipped_relu = bool(params.get("NARX_APPLY_CLIPPED_RELU", False))

        self.logger.info(
            f"Building NARX model with input_size={input_size}, output_size={output_size}, "
            f"hidden_layer_sizes={hidden_layer_sizes}, output_delay={output_delay}, "
            f"dropout_prob={dropout_prob}, activation={activation_function}, "
            f"device={target_device}, apply_clipped_relu={apply_clipped_relu}"
        )

        model = NARXModel(
            input_size=input_size,
            output_size=output_size,
            hidden_layer_sizes=hidden_layer_sizes,
            output_delay=output_delay,
            dropout_prob=dropout_prob,
            apply_clipped_relu=apply_clipped_relu,
            activation_function=activation_function,
            device=target_device,
        ).to(target_device)

        return model

    def create_model(self, params: dict, trial=None, device=None):
        return self.build_narx_model(params, device=device)

    def save_model(self, model: NARXModel, model_path: str):
        torch.save(model.state_dict(), model_path)
        self.logger.info(f"NARX model saved to {model_path}")

    def create_and_save_narx_model(self, params: dict, model_path: str, target_device=None):
        if target_device is not None:
            self.device = target_device
        model = self.build_narx_model(params)
        self.save_model(model, model_path)
        return model
