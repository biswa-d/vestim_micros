import torch
import logging
from vestim.services.model_training.src.FNN_model import FNNModel

class FNNModelService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_fnn_model(self, params: dict, trial=None):
        """
        Build an FNN model using the provided parameters.
        This method is designed to be flexible for both standard model creation and Optuna trials.

        :param params: Dictionary containing model parameters.
        :param trial: An Optuna trial object, if in an optimization context.
        :return: An instance of FNNModel.
        """
        input_size = params.get("INPUT_SIZE", 3)
        output_size = params.get("OUTPUT_SIZE", 1)

        if trial:
            # Optuna-specific hyperparameter suggestion
            n_layers = trial.suggest_int('FNN_N_LAYERS', params['FNN_N_LAYERS'][0], params['FNN_N_LAYERS'][1])
            hidden_layer_sizes = []
            for i in range(n_layers):
                min_units, max_units = params['FNN_UNITS'][i]
                units = trial.suggest_int(f'FNN_UNITS_L{i}', min_units, max_units)
                hidden_layer_sizes.append(units)
            
            dropout_prob = trial.suggest_float('FNN_DROPOUT_PROB', params['FNN_DROPOUT_PROB'][0], params['FNN_DROPOUT_PROB'][1])

        else:
            # Standard model creation from fixed hyperparameters
            if 'FNN_N_LAYERS' in params and 'FNN_UNITS' in params:
                n_layers = params['FNN_N_LAYERS']
                fnn_units = params['FNN_UNITS']
                if isinstance(fnn_units, str):
                    hidden_layer_sizes = [int(u.strip()) for u in fnn_units.split(',')]
                else:
                    hidden_layer_sizes = fnn_units

                if len(hidden_layer_sizes) != n_layers:
                    raise ValueError(f"Mismatch between FNN_N_LAYERS ({n_layers}) and the number of units provided in FNN_UNITS ({len(hidden_layer_sizes)}).")
            else:
                # Fallback to the original HIDDEN_LAYER_SIZES for backward compatibility
                hidden_layer_sizes = params.get("HIDDEN_LAYER_SIZES")

            dropout_prob = params.get("DROPOUT_PROB", 0.0)

        if not hidden_layer_sizes:
            self.logger.error("FNN hidden layer configuration is missing or invalid.")
            raise ValueError("Cannot build FNN model without hidden layer sizes.")

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
    def create_model(self, params: dict, trial=None, device=None):
        """
        Create an FNN model in-memory, suitable for Optuna trials.

        :param params: Dictionary containing model parameters.
        :param trial: An Optuna trial object.
        :param device: The target device for the model.
        :return: An instance of FNNModel.
        """
        if device:
            self.device = device
        return self.build_fnn_model(params, trial=trial)

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