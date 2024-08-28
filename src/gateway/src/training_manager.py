import time
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.gateway.src.hyper_param_manager import VEstimHyperParamManager
from src.services.model_training.src.training_service import TrainingService
from src.services.model_training.src.LSTM_model_service import LSTMModelService
from src.services.model_training.src.data_loader_service import DataLoaderService
from src.gateway.src.job_manager import JobManager

class VEstimTrainingManager:
    def __init__(self):
        self.params = None
        self.current_hyper_params = None  # Initialize as None initially
        self.hyper_param_manager = VEstimHyperParamManager()  # Retrieve hyperparameters from the manager
        self.training_service = TrainingService()
        self.lstm_model_service = LSTMModelService()
        self.job_manager = JobManager()  # Manages job-related directories and files
        self.data_loader_service = DataLoaderService()  # Initialize the data loader service
        self.start_time = None
        self.models = []  # Store models in memory
        self.data_loaders = []  # Store data loaders in memory

    def build_models(self):
        # Retrieve current parameters from the hyperparameter manager
        self.params = self.hyper_param_manager.get_current_params()
        self.current_hyper_params = self.params  # Ensure current_hyper_params is set here

        # Set fixed input and output sizes
        input_size = 3  # Fixed input size
        output_size = 1  # Fixed output size
        layers = int(self.params['LAYERS'])
        hidden_units_list = [int(h) for h in self.params['HIDDEN_UNITS'].split(',')]
        lr_drop_factors = [float(lr) for lr in self.params['LR_DROP_FACTOR'].split(',')]

        for hidden_units in hidden_units_list:
            for lr_drop_factor in lr_drop_factors:
                model_dir = os.path.join(self.job_manager.get_job_folder(), 'models', f'model_lstm_hu_{hidden_units}_lrd_{lr_drop_factor}')
                os.makedirs(model_dir, exist_ok=True)
                
                model_name = f"model_lstm_hu_{hidden_units}_lrd_{lr_drop_factor}.pth"
                model_path = os.path.join(model_dir, model_name)

                # Call to the LSTMModelService to create the model
                model = self.lstm_model_service.create_and_save_lstm_model(input_size, output_size, layers, hidden_units, model_path)

                # Log the model details
                with open(os.path.join(model_dir, 'model_config.json'), 'w') as config_file:
                    json.dump({
                        'hidden_units': hidden_units,
                        'lr_drop_factor': lr_drop_factor,
                        'model_path': model_path
                    }, config_file, indent=4)

                # Store the model in the list
                self.models.append(model)

    def create_data_loaders(self):
        lookbacks = [int(lb) for lb in self.current_hyper_params['LOOKBACK'].split(',')]
        batch_sizes = [int(bs) for bs in self.current_hyper_params['BATCH_SIZE'].split(',')]

        for lookback in lookbacks:
            for batch_size in batch_sizes:
                data_loader = self.data_loader_service.create_data_loader(
                    train_folder=os.path.join(self.job_manager.get_job_folder(), 'train', 'processed_data'),
                    lookback=lookback,
                    batch_size=batch_size
                )
                self.data_loaders.append(data_loader)


    def start_training(self):
    # Build models first
        self.build_models()

        # Create data loaders
        self.create_data_loaders()

        # Start the training process for each model with each data loader in parallel
        with ThreadPoolExecutor(max_workers=len(self.models) * len(self.data_loaders)) as executor:
            futures = []
            for model in self.models:
                for data_loader in self.data_loaders:
                    futures.append(
                        executor.submit(
                            self.training_service.start_training,
                            model,
                            data_loader,
                            self.current_hyper_params,
                            self.update_progress
                        )
                    )

            for future in as_completed(futures):
                try:
                    future.result()  # Wait for the model training to complete
                except Exception as exc:
                    print(f"Model training generated an exception: {exc}")
                else:
                    print(f"Model training completed successfully.")


    def update_progress(self, repetition, epoch, train_loss, validation_error):
        # Calculate elapsed time
        elapsed_time = time.time() - self.start_time
        hours, minutes, seconds = self._format_time(elapsed_time)

        # Update GUI with the progress
        self._update_gui_with_progress(repetition, epoch, train_loss, validation_error, hours, minutes, seconds)

    def _update_gui_with_progress(self, repetition, epoch, train_loss, validation_error, hours, minutes, seconds):
        # Interface with the GUI to update the training progress
        pass  # This would be implemented based on your specific GUI framework

    def _format_time(self, elapsed_time):
        # Format the elapsed time into hours, minutes, and seconds
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        return int(hours), int(minutes), int(seconds)

