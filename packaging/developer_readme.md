# Vestirn Developer Guide

This document provides a developer-oriented guide to the `vestim` repository. It aims to give a quick start on the overall workflow and outline which methods in which files and folders are responsible for specific tasks.

## Project Overview

(Vestirn appears to be a desktop application, likely built with PyQt, for managing and executing machine learning workflows, specifically for time-series data, possibly related to battery or sensor data given file names like Arbin, Digatron. The workflow involves data import, processing (conversion, resampling, augmentation, normalization), hyperparameter configuration, model training (FNN, GRU, LSTM), and model testing.)

## Overall Workflow

1.  **Data Import (`DataImportGUI`)**: User selects raw training and testing data files (e.g., MAT, CSV, Excel from sources like Arbin, Digatron, STLA). These files are organized into a new job directory by a `FileOrganizer` worker, which uses a source-specific `DataProcessor` (e.g., `DataProcessorArbin`) to copy and potentially perform initial conversion/resampling to CSV.
2.  **Data Augmentation (`DataAugmentGUI`)**: User configures and applies augmentation steps to the processed data within the job folder. This includes:
    *   Creating new features using custom formulas (`FormulaInputDialog`).
    *   Resampling data to a consistent frequency.
    *   Applying normalization (scaling) to selected features.
    *   These operations are managed by `DataAugmentManager` which uses `DataAugmentService` and `NormalizationService`. An `AugmentationWorker` handles this in a separate thread.
3.  **Hyperparameter Configuration (`VEstimHyperParamGUI`)**: User configures hyperparameters for model training. This includes:
    *   Selecting feature and target columns.
    *   Choosing training methodology (e.g., whole sequence, single/multi-step prediction).
    *   Selecting model type (FNN, GRU, LSTM) and its specific architecture (layers, units, dropout).
    *   Configuring learning rate schedulers and validation criteria.
    *   Parameters are managed by `VEstimHyperParamManager` and can be loaded/saved.
4.  **Training Setup (`VEstimTrainSetupGUI`)**: Based on selected hyperparameters, the `VEstimTrainingSetupManager` (via a `SetupWorker` thread):
    *   Validates parameters.
    *   Builds model architectures (e.g., `LSTMModelService.build_lstm_model()`).
    *   Creates a list of training tasks, each representing a specific model configuration to train.
5.  **Model Training (`VEstimTrainingTaskGUI`)**: For each training task:
    *   The `TrainingTaskManager` (via a `TrainingThread`) orchestrates the training.
    *   `DataLoaderService` prepares `DataLoader` instances using appropriate data handlers (`SequenceRNNDataHandler` for RNNs, `WholeSequenceFNNDataHandler` for FNNs).
    *   The `TrainingTaskService` executes the training loop (epoch training and validation), logs progress (CSV, SQLite), and saves the trained model using the respective model service (e.g., `LSTMModelService.save_model()`).
6.  **Model Testing (`VEstimTestingGUI`)**:
    *   The `VEstimTestingManager` (via a `TestingThread`) runs testing tasks.
    *   For each model, `TestDataService` creates a test data loader.
    *   `VEstimTestingService` loads the trained model, performs predictions on the test data, calculates metrics, and returns results.
    *   The GUI displays results, including plots.

## Directory Structure

- **`vestim/`**: Root directory for the application.
  - **[`config.py`](./config.py)**: Configuration settings for the project.
  - **[`logger_config.py`](./logger_config.py)**: Setup for application-wide logging.
  - **`gateway/`**: Manages interactions between the GUI and the backend services. Contains manager modules that orchestrate calls to services based on GUI events.
    - `src/`: Source code for gateway components.
  - **`gui/`**: Contains all Qt-based graphical user interface components. These are the front-end windows and widgets the user interacts with.
    - `src/`: Source code for GUI windows and widgets.
  - **`services/`**: Houses the core business logic, data processing, model training, and other backend functionalities. These modules perform the actual computations and data manipulations.
    - `data_conversion/`: Service for converting data formats.
    - `data_import/`: Service for importing data.
    - `data_processor/`: Services for data processing (e.g., specific to Arbin, Digatron, STLA) and data augmentation.
    - `hyper_param_selection/`: Service related to hyperparameter selection processes. (Note: No files listed under this, might be planned or integrated elsewhere).
    - `model_testing/`: Services for evaluating trained models.
    - `model_training/`: Services for training various machine learning models (e.g., FNN, GRU, LSTM) and handling data loading for training.
  - **[`__init__.py`](./__init__.py)**: Makes `vestim` a Python package.
  - **`__pycache__/`**: Directory for Python's cached bytecode (should be ignored by developers for understanding the source).


## Module Breakdown

### `vestim/` (Root Files)

#### [`config.py`](./config.py)
Defines project-wide configuration variables, primarily paths.

- **Variables:**
  - `ROOT_DIR`: Stores the absolute path to the project's root directory. Used for constructing other paths within the project.
  - `OUTPUT_DIR`: Defines the path to the directory where output files (e.g., trained models, logs, processed data) are stored, relative to `ROOT_DIR`.

#### [`logger_config.py`](./logger_config.py)
Configures the logging mechanism for the application, ensuring consistent logging behavior.

- **Functions:**
  - `setup_logger(log_file='default.log')`: Initializes and configures a global logger instance. It sets up a console handler for immediate feedback and a rotating file handler for persistent logs. Prevents duplicate handlers if called multiple times. Returns the configured logger.

### `vestim/gateway/src/`
This directory contains manager classes that act as intermediaries or controllers between the GUI layer and the backend services. They typically handle user actions from the GUI, invoke appropriate service methods, and relay results back to the GUI.

#### [`data_augment_manager_qt.py`](./gateway/src/data_augment_manager_qt.py)
Manages data augmentation operations, coordinating between the GUI and the data augmentation services. It handles tasks like applying augmentations, resampling, validating formulas, and retrieving column information.

- **Class: `DataAugmentManager(QObject)`**
  - **`__init__(self)`**: Initializes the manager, likely setting up connections to services and logging.
  - **`_set_job_context(self, job_folder: str)`**: Sets the context for a specific job, typically involving defining paths based on the `job_folder`.
  - **`apply_augmentations(self, ...)`**: Orchestrates the application of various data augmentation techniques (e.g., new column creation, normalization, resampling) to data files within a specified job folder. It interacts with `DataAugmentService` and `NormalizationService`.
  - **`resample_data(self, df: pd.DataFrame, frequency: str) -> pd.DataFrame`**: Resamples the given DataFrame to a specified frequency. Likely calls a corresponding method in `DataAugmentService`.
  - **`validate_formula(self, formula: str, df: pd.DataFrame) -> bool`**: Validates a formula for creating new columns against a sample DataFrame. Delegates to `DataAugmentService`.
  - **`get_column_info(self, job_folder: str) -> Dict[str, Dict[str, Any]]`**: Retrieves information about columns (e.g., names, types, statistics) from data files within the job folder.
  - **`get_sample_train_dataframe(self, job_folder: str) -> Optional[pd.DataFrame]`**: Fetches a sample DataFrame from the training data in the specified job folder, often used for previews or validations in the GUI.

#### [`hyper_param_manager_qt.py`](./gateway/src/hyper_param_manager_qt.py)
Manages hyperparameter configurations for model training. It follows a singleton pattern to ensure a single source of truth for hyperparameters. It handles loading, validating, saving, and providing access to these parameters.

- **Class: `VEstimHyperParamManager`** (Singleton)
  - **`__new__(cls, *args, **kwargs)`**: Implements the singleton pattern, ensuring only one instance of the manager exists.
  - **`__init__(self)`**: Initializes the manager, setting up default paths and loading initial parameters if available.
  - **`load_params(self, filepath)`**: Loads and validates hyperparameter configurations from a specified JSON file.
  - **`validate_and_normalize_params(self, params)`**: Validates the structure and values of the loaded parameters. It also normalizes certain parameter formats (e.g., string lists).
  - **`save_params(self)`**: Saves the current validated hyperparameters to a JSON file within the active job's folder.
  - **`save_params_to_file(self, new_params, filepath)`**: Saves a given set of parameters to a specified file path after validation.
  - **`update_params(self, new_params)`**: Updates the current hyperparameters with a new set of parameters, typically after user modification in the GUI.
  - **`get_current_params(self)`**: Retrieves the currently loaded and validated hyperparameters.
  - **`get_hyper_params(self)`**: Returns the current hyperparameter dictionary.

#### [`job_manager_qt.py`](./gateway/src/job_manager_qt.py)
Manages the creation and retrieval of job-specific directories. It uses a singleton pattern to ensure consistent job path management throughout the application. A "job" likely represents a single run or experiment, encompassing data, models, and results.

- **Class: `JobManager`** (Singleton)
  - **`__new__(cls, *args, **kwargs)`**: Implements the singleton pattern.
  - **`__init__(self)`**: Initializes the manager, setting the base output directory and the current job folder to `None`.
  - **`create_new_job(self)`**: Creates a new unique job folder within the main output directory. This folder will house all artifacts for a new job (e.g., training data, test data, model outputs).
  - **`get_job_folder(self)`**: Returns the path to the currently active job folder.
  - **`get_train_folder(self)`**: Returns the path to the 'train' subfolder within the current job folder, used for storing training datasets.
  - **`get_test_folder(self)`**: Returns the path to the 'test' subfolder within the current job folder, used for storing testing datasets.
  - **`get_test_results_folder(self)`**: Returns the path to the 'test_results' subfolder within the current job folder, used for storing model evaluation results.

#### [`testing_manager_qt.py`](./gateway/src/testing_manager_qt.py)
Manages the model testing process. It orchestrates the execution of testing tasks, logs results, and interacts with the `TestingService`.

- **Class: `VEstimTestingManager`**
  - **`__init__(self)`**: Initializes the testing manager, setting up necessary services (like `JobManager`, `TestingService`), configurations (max workers for threading), and task queues.
  - **`start_testing(self, queue)`**: Initiates the testing process by receiving a queue of testing tasks and starting a new thread to run `_run_testing_tasks`.
  - **`_run_testing_tasks(self)`**: The core method that processes testing tasks from the queue. It uses a `ThreadPoolExecutor` to run individual model tests concurrently.
  - **`_test_single_model(self, task, idx, test_folder)`**: Handles the testing of a single model. It loads the model, prepares test data using `TestDataService`, runs predictions via `TestingService`, calculates metrics, and logs the results.
  - **`generate_shorthand_name(task)` (static method)**: Creates a concise, descriptive name for a testing task based on its parameters, used for file naming and logging.
  - **`log_test_to_sqlite(self, task, results, db_log_file)`**: Logs the results of a test task (including metrics and task parameters) to an SQLite database.
  - **`log_test_to_csv(self, task, results, csv_log_file)`**: Logs the results of a test task to a CSV file.

#### [`training_setup_manager_qt.py`](./gateway/src/training_setup_manager_qt.py)
Manages the setup phase for model training. This includes building model architectures based on hyperparameters and creating a list of training tasks. It follows a singleton pattern.

- **Class: `VEstimTrainingSetupManager`** (Singleton)
  - **`__new__(cls, *args, **kwargs)`**: Implements the singleton pattern.
  - **`__init__(self, progress_signal=None, job_manager=None)`**: Initializes the manager, connecting to `JobManager` and `VEstimHyperParamManager`, and setting up a progress signal for GUI updates.
  - **`setup_training(self)`**: Orchestrates the training setup process: validates parameters, builds models, and creates training tasks.
  - **`create_selected_model(self, model_type, model_params, model_path)`**: Creates an instance of a specified model type (e.g., LSTM, GRU, FNN) with given parameters and saves its definition/architecture.
  - **`build_models(self)`**: Iterates through hyperparameter combinations to define and save different model architectures (e.g., LSTM models with varying layers/units) to be trained.
  - **`create_training_tasks(self)`**: Generates a list of training tasks based on the built models and hyperparameter settings. Each task represents a specific model configuration to be trained.
  - **`_create_task_info(self, model_task, hyperparams, repetition)`**: Constructs a detailed dictionary for a single training task, including model paths, hyperparameters, and unique identifiers.
  - **`calculate_learnable_parameters(self, layers, input_size, hidden_units)`**: Calculates the number of learnable parameters for a given RNN model configuration.
  - **`update_task(self, task_id, db_log_file=None, csv_log_file=None)`**: Updates the information for a specific task, potentially after it has been processed or if its logging paths change.
  - **`validate_parameters(self, params)`**: Validates and converts hyperparameter values to their appropriate data types before use.

#### [`training_task_manager_qt.py`](./gateway/src/training_task_manager_qt.py)
Manages the execution of individual training tasks. This involves setting up logging, creating data loaders, running the training loop for a model, and saving the trained model.

- **Functions (Module-level):**
  - **`format_time(seconds)`**: Utility function to format a duration in seconds into a human-readable string (HH:MM:SS).

- **Class: `TrainingTaskManager`**
  - **`__init__(self)`**: Initializes the manager, setting up logging, services (`JobManager`, `VEstimHyperParamManager`), and device configuration (CPU/GPU).
  - **`log_to_csv(self, task, epoch, train_loss, val_loss, elapsed_time, current_lr, best_val_loss, delta_t_epoch)`**: Logs epoch-level training progress (losses, time, learning rate) to a CSV file specific to the task.
  - **`log_to_sqlite(self, task, epoch, train_loss, val_loss, best_val_loss, elapsed_time, avg_batch_time, early_stopping, model_memory_usage, current_lr)`**: Logs detailed epoch-level training progress to an SQLite database specific to the task.
  - **`process_task(self, task, update_progress_callback)`**: Orchestrates the training for a single task. It sets up logging, creates data loaders, runs the training loop, and saves the model.
  - **`setup_job_logging(self, task)`**: Configures CSV and SQLite logging for the given training task, creating necessary files and tables.
  - **`create_sql_tables(self, db_log_file)`**: Creates the schema (tables) in the SQLite database for storing training logs if they don't already exist.
  - **`create_data_loaders(self, task)`**: Prepares and returns PyTorch `DataLoader` instances for training and validation datasets based on the task's specifications. It uses `DataLoaderService`.
  - **`run_training(self, task, update_progress_callback, train_loader, val_loader, device)`**: Executes the main training loop for the specified task. This includes iterating through epochs, performing forward and backward passes, calculating losses, updating model weights, handling learning rate scheduling, early stopping, and logging progress. It interacts with the specific model service (e.g., `LSTMModelService`).
  - **`convert_hyperparams(self, hyperparams)`**: Converts hyperparameter values from strings (as loaded from config) to their appropriate Python types (e.g., int, float, list of ints).
  - **`save_model(self, task)`**: Saves the trained model's state dictionary and potentially the full model object to disk. It might save in multiple formats (e.g., PyTorch internal, ONNX).
  - **`stop_task(self)`**: Sets a flag to gracefully stop the current training task, typically triggered by user intervention.

### `vestim/gui/src/`
This directory contains the Qt-based GUI components of the application. These classes define the visual elements and user interactions.

#### [`data_augment_gui_qt.py`](./gui/src/data_augment_gui_qt.py)
Provides the graphical user interface for configuring and applying data augmentation steps. It allows users to select a job, define formulas for new columns, set resampling frequencies, and trigger the augmentation process.

- **Class: `FormulaInputDialog(QDialog)`**
  - **`__init__(self, available_columns, session_created_column_names, parent=None)`**: Initializes the dialog for users to input custom formulas. It takes available column names and names of columns already created in the current session to aid the user.
  - **`initUI(self)`**: Sets up the UI elements of the formula input dialog (input field, column list, buttons).
  - **`add_column_to_formula(self, item)`**: Adds a selected column name from the list to the formula input field.
  - **`accept_formula(self)`**: Validates and accepts the entered formula, making it available to the main augmentation GUI.

- **Class: `AugmentationWorker(QObject)`**
  - **`__init__(self, data_augment_manager, job_folder, padding_length, resampling_frequency, column_formulas, normalize_data=False)`**: Initializes the worker thread for data augmentation. It takes the `DataAugmentManager`, job details, and augmentation parameters.
  - **`run(self)`**: Executes the data augmentation process by calling the `apply_augmentations` method of the `DataAugmentManager`. This runs in a separate thread to keep the GUI responsive. Emits signals upon completion or error.

- **Class: `DataAugmentGUI(QMainWindow)`**
  - **`__init__(self, job_folder=None)`**: Initializes the main window for data augmentation. Sets up the `DataAugmentManager` and loads initial data if a `job_folder` is provided.
  - **`initUI(self)`**: Creates and arranges all the UI widgets for the data augmentation screen (job selection, formula list, resampling options, normalization checkbox, apply button, navigation buttons).
  - **`select_job_folder(self)`**: Opens a dialog for the user to select the root folder for the current job. Loads sample data upon selection.
  - **`show_formula_dialog(self)`**: Displays the `FormulaInputDialog` to allow the user to define a new column based on a formula.
  - **`remove_formula(self)`**: Removes a selected formula from the list of formulas to be applied.
  - **`apply_changes(self)`**: Gathers all augmentation settings from the GUI, creates an `AugmentationWorker`, and starts it in a new thread to perform the augmentation.
  - **`handle_augmentation_finished(self, job_folder, processed_files_metadata)`**: Slot to handle the `finished` signal from `AugmentationWorker`. Updates the GUI, logs completion, and potentially enables navigation to the next step.
  - **`handle_critical_error(self, error_msg)`**: Slot to display critical error messages that occur during augmentation.
  - **`handle_formula_error(self, error_msg)`**: Slot to display errors related to formula validation.
  - **`go_to_hyperparameter_gui(self)`**: Navigates the user to the hyperparameter configuration GUI.

- **Functions (Module-level):**
  - **`main()`**: Entry point to run the `DataAugmentGUI` as a standalone application (likely for testing/development).

#### [`data_import_gui_qt.py`](./gui/src/data_import_gui_qt.py)
Provides the GUI for users to select and organize their training and testing datasets. It allows specifying folders for train/test data, lists the files, and then processes/copies them into a structured job directory.

- **Class: `DataImportGUI(QMainWindow)`**
  - **`__init__(self)`**: Initializes the main window for data import. Sets up `JobManager` and `DataProcessorService`.
  - **`initUI(self)`**: Sets up the UI elements: buttons to select train/test folders, lists to display selected files, a dropdown for data source type (e.g., Arbin, Digatron), a button to start organizing files, and navigation buttons.
  - **`select_train_folder(self)`**: Opens a dialog for the user to select the folder containing training data files. Populates the training file list.
  - **`select_test_folder(self)`**: Opens a dialog for the user to select the folder containing testing data files. Populates the testing file list.
  - **`populate_file_list(self, folder_path, list_widget)`**: Lists all files (or files of a specific type based on data source) from the given `folder_path` into the specified `list_widget`.
  - **`check_folders_selected(self)`**: Checks if both train and test folders have been selected by the user.
  - **`organize_files(self)`**: Initiates the file organization process. It creates a new job via `JobManager`, gathers selected file paths and the data processor type, then starts a `FileOrganizer` worker thread.
  - **`on_files_processed(self, job_folder)`**: Slot called when the `FileOrganizer` worker finishes. Enables navigation to the next screen (Data Augmentation GUI).
  - **`move_to_next_screen(self, job_folder)`**: Transitions the application to the `DataAugmentGUI`, passing the `job_folder`.

- **Class: `FileOrganizer(QObject)`**
  - **`__init__(self, train_files, test_files, data_processor, sampling_frequency=None)`**: Initializes the worker thread for organizing and preprocessing data files. Takes lists of train/test files and the selected `DataProcessorService`.
  - **`run(self)`**: Executes the file organization and initial processing. It iterates through the selected train and test files, copies them to the appropriate subdirectories within the newly created job folder, and potentially performs initial processing (like resampling or format conversion) using the `DataProcessorService`. Emits progress signals.

- **Functions (Module-level):**
  - **`main()`**: Entry point to run the `DataImportGUI` as a standalone application.

#### [`hyper_param_gui_qt.py`](./gui/src/hyper_param_gui_qt.py)
Provides the GUI for configuring all hyperparameters related to model training. This includes selecting features/targets, training methods, model architectures (LSTM, GRU, FNN) and their specific parameters, learning rate schedulers, and validation criteria.

- **Class: `VEstimHyperParamGUI(QWidget)`**
  - **`__init__(self)`**: Initializes the hyperparameter GUI window. Sets up `JobManager` and `VEstimHyperParamManager`.
  - **`setup_window(self)`**: Basic window setup (title, icon, size).
  - **`build_gui(self)`**: Constructs the entire UI by calling various `add_*_selection` methods to create sections for different hyperparameter categories. Also loads column names for feature/target selection.
  - **`add_feature_target_selection(self, layout)`**: Adds UI elements for selecting feature columns and target columns from the available data columns.
  - **`add_training_method_selection(self, layout)`**: Adds UI elements for selecting the overall training approach (e.g., 'whole_sequence', 'single_step', 'multi_step') and related parameters like batch size and epochs.
  - **`update_training_method(self)`**: Updates UI elements based on the selected training method (e.g., shows/hides sequence length input).
  - **`update_batch_size_visibility(self)`**: Shows or hides the batch size input based on whether the selected training method uses batching.
  - **`add_model_selection(self, layout)`**: Adds UI elements for selecting the model type (LSTM, GRU, FNN) and configuring its architecture (layers, units, dropout, etc.).
  - **`update_model_params(self)`**: Dynamically updates the model parameter input fields based on the selected model type.
  - **`add_scheduler_selection(self, layout)`**: Adds UI elements for selecting and configuring a learning rate scheduler (e.g., ReduceLROnPlateau, StepLR).
  - **`update_scheduler_settings(self)`**: Updates the scheduler-specific input fields based on the chosen scheduler type.
  - **`add_validation_criteria(self, layout)`**: Adds UI elements for setting validation split, early stopping criteria, and other validation-related parameters.
  - **`collect_hyperparameters(self)`**: Gathers all hyperparameter values from the various input fields in the GUI and returns them as a dictionary.
  - **`proceed_to_training(self)`**: Collects hyperparameters, validates them, saves them using `VEstimHyperParamManager`, and then navigates to the `TrainingSetupGUI`.
  - **`show_training_setup_gui(self, params_to_pass)`**: Transitions to the `TrainingSetupGUI`, passing the collected hyperparameters.
  - **`load_column_names(self)`**: Loads available column names from a sample processed data file (from the current job) to populate feature/target selection dropdowns.
  - **`load_params_from_json(self)`**: Allows the user to load hyperparameter settings from a previously saved JSON file. Updates the GUI with these loaded parameters.
  - **`update_params(self, new_params)`**: Updates the internal state with new parameters and refreshes the GUI to reflect them.
  - **`update_gui_with_loaded_params(self)`**: Populates all GUI input fields with values from a loaded hyperparameter dictionary.
  - **`open_guide(self)`**: Opens a PDF document that serves as a guide for understanding and setting the various hyperparameters.

#### [`testing_gui_qt.py`](./gui/src/testing_gui_qt.py)
Provides the GUI for initiating and monitoring the model testing phase. It displays a list of models to be tested (derived from training tasks), shows their results (metrics, plots) as they become available, and allows users to view detailed plots.

- **Class: `TestingThread(QThread)`**
  - **`__init__(self, testing_manager, queue)`**: Initializes a worker thread for running testing tasks. Takes the `VEstimTestingManager` and a queue of tasks.
  - **`run(self)`**: Starts the testing process by calling the `start_testing` method of the `VEstimTestingManager`. It processes tasks from the queue and emits signals for status updates.

- **Class: `VEstimTestingGUI(QMainWindow)`**
  - **`__init__(self)`**: Initializes the main testing GUI window. Sets up `JobManager`, `VEstimTestingManager`, and a queue for results.
  - **`initUI(self)`**: Sets up the UI: a table to display testing results, a button to start testing, status labels, and an elapsed time display.
  - **`display_hyperparameters(self, params)`**: (Likely intended to show hyperparameters for context, but might be partially implemented or used differently).
  - **`add_result_row(self, result)`**: Adds a new row to the results table with information from a completed testing task (model name, metrics, links to plots).
  - **`plot_model_result(self, predictions_file, save_dir, target_column_name)`**: Generates and saves a plot comparing actual vs. predicted values for a specific model test.
  - **`save_plot(self, fig, test_file_path, save_dir)`**: Saves a Matplotlib figure to a PNG file in the specified directory.
  - **`show_training_history_plot(self, plot_path, task_id)`**: Opens a new window to display a saved training history plot for a given task.
  - **`start_testing(self)`**: Initiates the testing process. It retrieves the list of training tasks (which imply models to be tested), creates a `TestingThread`, and starts it.
  - **`update_elapsed_time(self)`**: Updates the elapsed time display on the GUI.
  - **`process_queue(self)`**: Periodically checks the results queue for completed test tasks and calls `add_result_row` to update the GUI.
  - **`all_tests_completed(self)`**: Called when all testing tasks are finished. Stops timers and updates status.
  - **`open_job_folder(self)`**: Opens the current job's main folder in the system's file explorer.

#### [`training_setup_gui_qt.py`](./gui/src/training_setup_gui_qt.py)
Provides a GUI to display the status of the training setup process (model building and task creation) and allows the user to proceed to the main training GUI once setup is complete.

- **Class: `SetupWorker(QThread)`**
  - **`__init__(self, job_manager)`**: Initializes a worker thread for the training setup process. Takes an instance of `VEstimTrainingSetupManager`.
  - **`run(self)`**: Executes the training setup by calling the `setup_training` method of the `VEstimTrainingSetupManager`. Emits progress signals to update the GUI.

- **Class: `VEstimTrainSetupGUI(QWidget)`**
  - **`__init__(self, params)`**: Initializes the training setup GUI window. Receives hyperparameters (`params`) from the previous screen. Sets up `JobManager` and `VEstimTrainingSetupManager`.
  - **`build_gui(self)`**: Constructs the UI: displays selected hyperparameters, a status area for setup progress, an elapsed time display, and a "Start Setup" button.
  - **`display_hyperparameters(self, layout)`**: Creates a display area to show the key hyperparameters that were configured in the previous step.
  - **`start_setup(self)`**: Initiates the training setup process by creating and starting a `SetupWorker` thread.
  - **`update_status(self, message, path="", task_count=None)`**: Slot to receive progress updates from the `SetupWorker`. Updates status labels and progress bars in the GUI.
  - **`show_proceed_button(self)`**: Makes the "Proceed to Training" button visible once the setup process is successfully completed.
  - **`transition_to_training_gui(self)`**: Navigates the user to the `TrainingTaskGUI` to start and monitor the actual model training.
  - **`update_elapsed_time(self)`**: Updates the elapsed time display on the GUI during the setup process.

#### [`training_task_gui_qt.py`](./gui/src/training_task_gui_qt.py)
Provides the main GUI for managing and monitoring the model training process. It displays a list of training tasks, allows users to start/stop training, and shows real-time progress (loss plots, epoch information, logs).

- **Class: `TrainingThread(QThread)`**
  - **`__init__(self, task, training_task_manager)`**: Initializes a worker thread for executing a single training task. Takes the task details and an instance of `TrainingTaskManager`.
  - **`run(self)`**: Executes the training for the given task by calling `process_task` on the `TrainingTaskManager`. Emits signals for epoch updates and completion.

- **Class: `VEstimTrainingTaskGUI(QMainWindow)`**
  - **`__init__(self, task_list, params)`**: Initializes the training task GUI. Receives a list of training tasks and overall hyperparameters. Sets up `TrainingTaskManager`.
  - **`build_gui(self, task)`**: Dynamically builds or rebuilds the GUI for a specific task, including areas for hyperparameters, plots (loss vs. epoch), progress bars, status messages, and a log window.
  - **`display_hyperparameters(self, params)`**: Shows the hyperparameters relevant to the current training task in a dedicated section of the GUI.
  - **`setup_time_and_plot(self, task)`**: Initializes the plotting area for training/validation loss and sets up timers for elapsed time.
  - **`setup_log_window(self, task)`**: Creates and configures a text area to display real-time log messages for the current training task.
  - **`clear_layout(self)`**: Clears the central widget of the GUI, used when switching between tasks or views.
  - **`start_task_processing(self)`**: Initiates the processing of the current training task in the `task_list`. It creates and starts a `TrainingThread`.
  - **`update_elapsed_time(self)`**: Updates the display of total elapsed training time.
  - **`clear_plot(self)`**: Clears the Matplotlib plot canvas.
  - **`process_queue(self)`**: Checks a queue for epoch progress data emitted by the `TrainingThread` and calls `update_gui_after_epoch`.
  - **`handle_error(self, error_message)`**: Displays an error message if an issue occurs during training.
  - **`update_gui_after_epoch(self, progress_data)`**: Updates the GUI with new data after each epoch (e.g., refreshes loss plots, updates progress bars, logs messages).
  - **`stop_training(self)`**: Signals the `TrainingTaskManager` to stop the currently running training task and handles GUI updates.
  - **`check_if_stopped(self)`**: Checks if the stop flag has been acknowledged by the training thread.
  - **`task_completed(self)`**: Handles GUI updates and logic when a training task finishes (e.g., saves final plots, enables navigation).
  - **`wait_for_thread_to_stop(self)`**: Ensures the training thread has fully stopped before proceeding.
  - **`on_closing(self)`**: Handles cleanup actions when the GUI window is closed, such as stopping any active training.
  - **`transition_to_testing_gui(self)`**: Navigates the user to the `VEstimTestingGUI` after all training tasks are completed or if the user chooses to proceed.

### `vestim/services/`

#### [`normalization_service.py`](./services/normalization_service.py)
This module provides functions for data normalization and denormalization, primarily using scikit-learn scalers. It supports calculating global statistics for a dataset, creating scalers from these stats, saving/loading scalers, and transforming/inverse-transforming data.

- **Functions:**
  - **`calculate_global_dataset_stats(data_items: list, feature_columns: list, data_reading_func=pd.read_csv, **read_kwargs)`**: Computes global statistics (mean, std, min, max) for specified `feature_columns` across a list of data files (`data_items`). It reads data using `data_reading_func`.
  - **`create_scaler_from_stats(global_stats, feature_columns, scaler_type='min_max')`**: Creates a scikit-learn scaler object (e.g., `MinMaxScaler`, `StandardScaler`) initialized with pre-computed `global_stats` for the given `feature_columns`.
  - **`save_scaler(scaler, directory, filename="scaler.joblib")`**: Saves a trained scaler object to disk using `joblib`.
  - **`load_scaler(scaler_path)`**: Loads a previously saved scaler object from disk.
  - **`transform_data(data_df, scaler, feature_columns)`**: Applies the normalization (scaling) transformation to the specified `feature_columns` of a DataFrame using a pre-fitted `scaler`.
  - **`inverse_transform_data(data_df, scaler, feature_columns)`**: Reverts the normalization, applying the inverse transformation to scaled data to get it back to its original scale.

### `vestim/services/data_conversion/src/`

#### [`data_conversion_service.py`](./services/data_conversion/src/data_conversion_service.py)
This service provides functionality to convert data files between different formats, primarily targeting CSV as the output format. It includes a Flask API endpoint for triggering conversions.

- **Flask Routes:**
  - **`@app.route('/convert', methods=['POST'])` -> `convert_files()`**: API endpoint that accepts a list of file paths and their types (e.g., 'mat', 'csv', 'excel'). It then calls the appropriate conversion function for each file.
- **Conversion Functions:**
  - **`convert_mat_to_csv(mat_file, output_folder)`**: Converts a MATLAB `.mat` file to one or more CSV files (if the .mat file contains multiple 2D arrays). Each suitable array is saved as a separate CSV.
  - **`convert_csv_to_csv(csv_file, output_folder)`**: Essentially copies a CSV file to the output folder, potentially with some cleaning or standardization (though the current implementation seems to be a direct copy after reading and re-writing).
  - **`convert_excel_to_csv(excel_file, output_folder)`**: Converts each sheet in an Excel file (`.xls`, `.xlsx`) into a separate CSV file in the output folder.

### `vestim/services/data_import/src/`

#### [`data_import_service.py`](./services/data_import/src/data_import_service.py)
This service provides a Flask API endpoint for uploading and organizing data files into a structured job directory.

- **Flask Routes:**
  - **`@app.route('/upload', methods=['POST'])` -> `upload_files()`**: API endpoint that handles file uploads. It expects 'train_files' and 'test_files' in the request, along with an 'output_base_dir'. It creates a new unique job directory within the `output_base_dir` and copies the uploaded train and test files into respective 'train_data' and 'test_data' subdirectories within this job folder. Returns the path to the created job folder.

### `vestim/services/data_processor/src/`

#### [`data_augment_service.py`](./services/data_processor/src/data_augment_service.py)
This service class encapsulates various data augmentation and preprocessing operations that can be applied to datasets.

- **Class: `DataAugmentService`**
  - **`__init__(self)`**: Initializes the service, including a logger.
  - **`_set_job_context(self, job_folder: str)`**: Sets the current job folder context for path generations.
  - **`load_processed_data(self, train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]`**: Loads processed training and testing data from specified CSV file paths into pandas DataFrames.
  - **`resample_data(self, df: pd.DataFrame, frequency: str, progress_callback=None) -> pd.DataFrame`**: Resamples a time-series DataFrame to a specified `frequency` (e.g., '1S', '10T'). It handles both upsampling (filling with NaNs or forward fill) and downsampling (aggregating, typically taking the mean).
  - **`validate_formula(self, formula: str, df: pd.DataFrame) -> Tuple[bool, Optional[str]]`**: Validates a user-provided string `formula` for creating a new column by attempting to evaluate it against a sample DataFrame `df`. Returns a boolean indicating validity and an error message if invalid.
  - **`create_columns(self, df: pd.DataFrame, column_formulas: Dict[str, str]) -> pd.DataFrame`**: Creates new columns in the DataFrame `df` based on a dictionary of `column_formulas` (where keys are new column names and values are their calculation formulas).
  - **`pad_data(self, df: pd.DataFrame, padding_length: int, resample_freq_for_time_padding: Optional[str] = None) -> pd.DataFrame`**: Pads the DataFrame `df` to a specified `padding_length`. If `resample_freq_for_time_padding` is provided, it pads based on time index; otherwise, it pads with NaN rows.
  - **`apply_normalization(self, df: pd.DataFrame, scaler: object, columns_to_normalize: List[str]) -> pd.DataFrame`**: Applies a pre-fitted scikit-learn `scaler` object to normalize the specified `columns_to_normalize` in the DataFrame.
  - **`save_single_augmented_file(self, augmented_df: pd.DataFrame, output_filepath: str)`**: Saves an augmented DataFrame to a CSV file at the given `output_filepath`.
  - **`update_augmentation_metadata(self, job_folder: str, processed_files_info: List[Dict[str, Any]])`**: Updates or creates a metadata JSON file (`augmentation_metadata.json`) within the `job_folder` to store information about the augmented files (e.g., original names, new names, columns).
  - **`get_column_info(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]`**: Extracts information about each column in the DataFrame, such as data type, min, max, mean, and standard deviation.

#### [`data_processor_qt_arbin.py`](./services/data_processor/src/data_processor_qt_arbin.py)
This class is responsible for processing data specifically from Arbin battery testing systems. It handles organizing raw data files, converting them to a standard CSV format, and resampling them if required. (Note: STLA processor seems very similar, likely for MAT files from STLA systems).

- **Class: `DataProcessorArbin`** (and `DataProcessorSTLA` which is very similar)
  - **`__init__(self)`**: Initializes the processor, setting up logging and internal counters for progress tracking.
  - **`organize_and_convert_files(self, train_files, test_files, progress_callback=None, sampling_frequency=None)`**: Main method to orchestrate the processing of Arbin/STLA data. It copies original files, then converts (MAT, CSV, Excel) and optionally resamples them into 'train_processed' and 'test_processed' subdirectories within the job folder.
  - **`switch_log_file(self, log_file)`**: Changes the target file for logging.
  - **`_copy_files(self, files, destination_folder, progress_callback=None)`**: Copies a list of files to a specified destination folder, updating progress via a callback.
  - **`_convert_files(self, input_folder, output_folder, progress_callback=None, sampling_frequency=1)`**: Converts files from `input_folder` to `output_folder` (primarily for MAT to CSV, potentially without resampling or with fixed resampling).
  - **`_convert_and_resample_files(self, input_folder, output_folder, progress_callback=None, sampling_frequency='1S')`**: Iterates through files in `input_folder`, determines their type (MAT for Arbin/STLA, also handles CSV/Excel for Arbin), and calls the appropriate `_convert_*_to_csv_resampled` method.
  - **`_convert_mat_to_csv(self, mat_file, output_folder)`**: Converts a MAT file to CSV without resampling. (Arbin/STLA)
  - **`_convert_csv_to_csv(self, csv_file_path, output_folder)`**: Converts/copies a CSV file to the output folder without resampling. (Arbin only)
  - **`_convert_excel_to_csv(self, excel_file_path, output_folder)`**: Converts an Excel file to CSVs (one per sheet) without resampling. (Arbin only)
  - **`_convert_mat_to_csv_resampled(self, mat_file, output_folder, sampling_frequency=1)`**: Converts data from a MAT file to CSV, then resamples it to the specified `sampling_frequency`. (Arbin/STLA)
  - **`_convert_csv_to_csv_resampled(self, csv_file_path, output_folder, sampling_frequency='1S')`**: Reads a CSV file, resamples its data, and saves it as a new CSV in the `output_folder`. (Arbin only)
  - **`_convert_excel_to_csv_resampled(self, excel_file_path, output_folder, sampling_frequency='1S')`**: Reads data from each sheet of an Excel file, resamples it, and saves each sheet as a separate resampled CSV. (Arbin only)
  - **`extract_data_from_matfile(self, file_path)`**: Extracts relevant data arrays (assumed to be 2D numeric) from a `.mat` file. (Arbin/STLA)
  - **`_resample_data(self, df, sampling_frequency='1S')`**: Resamples a pandas DataFrame to the given `sampling_frequency`, typically using mean aggregation for downsampling and forward-fill for upsampling. Assumes a 'datetime' index. (Arbin/STLA)
  - **`_extract_data_from_matfile(file_path)` (static method, duplicate logic in Arbin)**: A static version that also extracts data from MAT files. (Note: This appears to be a redundant or misplaced static method in Arbin, STLA has a similar instance method).
  - **`_update_progress(self, progress_callback)`**: Helper method to emit progress updates via the `progress_callback`.

#### [`data_processor_qt_digatron.py`](./services/data_processor/src/data_processor_qt_digatron.py)
This class handles data processing specifically for Digatron battery testing system files, which are typically CSVs but may have custom header formats.

- **Class: `DataProcessorDigatron`**
  - **`__init__(self)`**: Initializes the processor, setting up logging and progress counters.
  - **`organize_and_convert_files(self, train_files, test_files, progress_callback=None, sampling_frequency=None)`**: Orchestrates the processing of Digatron CSV files. It copies raw files and then processes them (handling custom headers and resampling) into 'train_processed' and 'test_processed' subdirectories.
  - **`_process_csv_with_custom_header_skip(self, input_csv_path, output_csv_path, sampling_frequency=None)`**: Processes Digatron CSV files that have a specific, non-standard header section. It attempts to identify the actual data start, reads the CSV, renames columns to a standard format (e.g., 'Time', 'Voltage', 'Current'), optionally resamples, and saves the processed data.
  - **`_process_standard_csv(self, input_csv_path, output_csv_path, sampling_frequency=None)`**: Processes Digatron CSV files assumed to have a more standard header or one that can be inferred. It reads the CSV, optionally resamples, and saves it.
  - **`switch_log_file(self, job_log_file)`**: Changes the target file for logging.
  - **`_copy_file(self, file_path, destination_folder, progress_callback=None)`**: Copies a single file to a destination, updating progress.
  - **`_update_progress(self, progress_callback)`**: Helper to emit progress updates.
  - **`_resample_data(self, df, sampling_frequency='1S')`**: Resamples a pandas DataFrame, assuming a 'datetime' index, to the specified `sampling_frequency`.

#### [`data_processor_qt_stla.py`](./services/data_processor/src/data_processor_qt_stla.py)
This class is responsible for processing data specifically from STLA systems, which primarily involves MAT files. Its structure and functionality are very similar to `DataProcessorArbin`, focusing on MAT file conversion and resampling.

- **Class: `DataProcessorSTLA`**
  - **`__init__(self)`**: Initializes the processor, setting up logging and progress counters.
  - **`organize_and_convert_files(self, train_files, test_files, progress_callback=None, sampling_frequency=None)`**: Main method to orchestrate the processing of STLA MAT data. It copies original files, then converts and optionally resamples them into 'train_processed' and 'test_processed' subdirectories.
  - **`switch_log_file(self, log_file)`**: Changes the target file for logging.
  - **`_copy_files(self, files, destination_folder, progress_callback=None)`**: Copies a list of files to a destination folder, updating progress.
  - **`_convert_files(self, input_folder, output_folder, progress_callback=None, sampling_frequency=1)`**: Converts MAT files from `input_folder` to CSVs in `output_folder` (potentially without resampling or with fixed resampling).
  - **`_convert_and_resample_files(self, input_folder, output_folder, progress_callback=None, sampling_frequency='1S')`**: Iterates through MAT files in `input_folder` and calls `_convert_mat_to_csv_resampled`.
  - **`_convert_mat_to_csv(self, mat_file, output_folder)`**: Converts a MAT file to CSV without resampling.
  - **`_convert_mat_to_csv_resampled(self, mat_file, output_folder, sampling_frequency=1)`**: Converts data from a MAT file to CSV, then resamples it.
  - **`extract_data_from_matfile(self, file_path)`**: Extracts relevant data arrays from a `.mat` file. (Instance method)
  - **`_resample_data(self, df, sampling_frequency='1S')`**: Resamples a pandas DataFrame.
  - **`_extract_data_from_matfile(file_path)` (static method)**: Extracts data from MAT files. (Note: Similar to the instance method, potentially for different internal use or a slight variation).
  - **`_update_progress(self, progress_callback)`**: Helper method to emit progress updates.

#### [`data_processor_qt.py`](./services/data_processor/src/data_processor_qt.py)
This class appears to be a more generic or older version of a data processor, primarily focused on MAT file conversion to CSV. It lacks the resampling capabilities and specific handling for different data sources (Arbin, Digatron, STLA) seen in the other `_qt_` suffixed processors. It includes progress callback mechanisms, suggesting it's intended for use with a Qt GUI.

- **Class: `DataProcessor`** (in [`data_processor_qt.py`](vestim/services/data_processor/src/data_processor_qt.py:11))
  - **`__init__(self)`**: Initializes the processor, setting up logging and progress counters.
  - **`organize_and_convert_files(self, train_files, test_files, progress_callback=None)`**: Orchestrates copying raw files and then converting MAT files to CSV in 'train_processed' and 'test_processed' folders.
  - **`switch_log_file(self, log_file)`**: Changes the target file for logging.
  - **`_copy_files(self, files, destination_folder, progress_callback=None)`**: Copies a list of files to a destination, updating progress.
  - **`_convert_files(self, input_folder, output_folder, progress_callback=None)`**: Iterates through files in `input_folder` and calls `_convert_mat_to_csv` if they are MAT files.
  - **`_convert_mat_to_csv(self, mat_file, output_folder)`**: Converts a MAT file to CSV. Extracts all 2D numerical arrays from the MAT file and saves each as a separate CSV.
  - **`_update_progress(self, progress_callback)`**: Helper to emit progress updates.

#### [`data_processor.py`](./services/data_processor/src/data_processor.py)
This is another generic data processor, very similar to [`data_processor_qt.py`](vestim/services/data_processor/src/data_processor_qt.py:11) but without any Qt-specific elements like `progress_callback` or logging setup. It seems to be a base or non-GUI version for MAT file conversion.

- **Class: `DataProcessor`** (in [`data_processor.py`](vestim/services/data_processor/src/data_processor.py:9))
  - **`__init__(self)`**: Basic initializer (currently empty).
  - **`organize_and_convert_files(self, train_files, test_files)`**: Orchestrates copying raw files and then converting MAT files to CSV into 'train_processed' and 'test_processed' folders within a 'data/jobs/current_job' structure (path seems somewhat hardcoded or reliant on a specific setup).
  - **`_copy_files(self, files, destination_folder)`**: Copies a list of files.
  - **`_convert_files(self, input_folder, output_folder)`**: Iterates and converts MAT files to CSV.
  - **`_convert_mat_to_csv(self, mat_file, output_folder)`**: Converts a MAT file to multiple CSVs, saving each 2D numerical array.

### `vestim/services/model_testing/src/`

#### [`test_data_service.py`](./services/model_testing/src/test_data_service.py)
This service is responsible for preparing test data into a format suitable for model evaluation, specifically creating PyTorch `DataLoader` instances.

- **Class: `VEstimTestDataService`**
  - **`__init__(self)`**: Initializes the service, including a logger.
  - **`create_test_file_loader(self, test_file_path, lookback, batch_size, feature_cols, target_col)`**: Reads a test data CSV file from `test_file_path`. It then processes this data:
    - Selects the specified `feature_cols` and `target_col`.
    - Creates sequences of data based on the `lookback` window (i.e., for time-series prediction, each sample will contain `lookback` previous steps as features).
    - Converts these sequences into PyTorch tensors.
    - Creates and returns a PyTorch `DataLoader` for the test set, configured with the given `batch_size`. This loader can then be used to feed data to a model for evaluation.

#### [`testing_service.py`](./services/model_testing/src/testing_service.py)
This service handles the core logic of testing a trained model. It loads a model, runs it on test data, and computes evaluation metrics.

- **Class: `VEstimTestingService`**
  - **`__init__(self, device='cpu')`**: Initializes the service, setting the device (CPU/GPU) for model inference and a logger.
  - **`load_model(self, model_path)`**: Loads a pre-trained PyTorch model from the given `model_path`. It expects the model to be saved as a state dictionary.
  - **`test_model(self, model, test_loader, hidden_units, num_layers, target_column_name: str, scaler_target=None, scaler_features=None, feature_columns_to_scale=None)`**: Evaluates the loaded `model` using data from the `test_loader`.
    - It iterates through the test data, makes predictions, and inverse-transforms predictions and actual values if scalers are provided.
    - Calculates various regression metrics (MAE, MSE, RMSE, R-squared, MAPE).
    - Returns a dictionary containing these metrics, along with lists of actual and predicted values.
  - **`run_testing(self, task, model_path, test_loader, test_file_path)`**: A higher-level method that orchestrates the testing for a single task. It loads the model, calls `test_model` to get predictions and metrics, and returns these results. This method doesn't save results itself but provides them to the caller (likely `VEstimTestingManager`).

### `vestim/services/model_training/src/`

#### [`base_data_handler.py`](./services/model_training/src/base_data_handler.py)
Defines an abstract base class for data handling, ensuring that concrete data handlers implement specific methods for loading and processing data.

- **Class: `BaseDataHandler(ABC)`**
  - **`__init__(self, feature_cols, target_col)`**: Initializes the handler with lists of feature column names and the target column name. Also sets up a logger.
  - **`load_and_process_data(self, folder_path: str, **kwargs) -> tuple[np.ndarray, np.ndarray]` (abstract method)**: Abstract method that must be implemented by subclasses. It's responsible for loading data from the given `folder_path`, processing it (e.g., creating sequences, splitting into features and targets), and returning features (X) and targets (y) as NumPy arrays.
  - **`_read_and_select_columns(self, file_path: str) -> pd.DataFrame | None`**: A utility method to read a CSV file into a pandas DataFrame and select only the `feature_cols` and `target_col` specified during initialization. Handles potential errors during file reading or column selection.

#### [`data_loader_service.py`](./services/model_training/src/data_loader_service.py)
This service is responsible for creating PyTorch `DataLoader` instances for training and validation, using appropriate data handlers based on the specified training method.

- **Class: `DataLoaderService`**
  - **`__init__(self)`**: Initializes the service, including a logger.
  - **`create_data_loaders(self, folder_path: str, training_method: str, feature_cols: list, target_col: str, lookback: int, batch_size: int, validation_split: float = 0.2, num_workers: int = 0, pin_memory: bool = False, shuffle_train: bool = True, shuffle_val: bool = False, **kwargs) -> tuple[DataLoader, DataLoader]`**:
    - Determines the appropriate data handler (`SequenceRNNDataHandler` or `WholeSequenceFNNDataHandler`) based on the `training_method`.
    - Initializes the selected handler with `feature_cols`, `target_col`, and other relevant parameters like `lookback`.
    - Calls the handler's `load_and_process_data` method to get features (X) and targets (y).
    - Splits the data into training and validation sets based on `validation_split`.
    - Creates PyTorch `TensorDataset` objects for both training and validation sets.
    - Creates and returns PyTorch `DataLoader` instances for both training and validation, configured with `batch_size`, `num_workers`, `pin_memory`, and shuffle options.

#### [`FNN_model_service.py`](./services/model_training/src/FNN_model_service.py)
This service handles the creation and saving of Feedforward Neural Network (FNN) models.

- **Class: `FNNModelService`**
  - **`__init__(self)`**: Initializes the service, including a logger.
  - **`build_fnn_model(self, params: dict)`**: Constructs an `FNNModel` instance based on the provided `params` dictionary. This dictionary should contain keys like `input_dim`, `hidden_dims` (a list of hidden layer sizes), `output_dim`, and `dropout_rate`.
  - **`save_model(self, model: FNNModel, model_path: str)`**: Saves the state dictionary of the trained `FNNModel` to the specified `model_path`.
  - **`create_and_save_fnn_model(self, params: dict, model_path: str)`**: A convenience method that first calls `build_fnn_model` to create the model and then `save_model` to save its initial state (or architecture definition) to disk.

#### [`FNN_model.py`](./services/model_training/src/FNN_model.py)
Defines the PyTorch module for a Feedforward Neural Network (FNN).

- **Class: `FNNModel(nn.Module)`**
  - **`__init__(self, input_size, output_size, hidden_layer_sizes, dropout_prob=0.0)`**: Constructor for the FNN.
    - `input_size`: Number of input features.
    - `output_size`: Number of output features (typically 1 for regression).
    - `hidden_layer_sizes`: A list of integers, where each integer is the number of neurons in a hidden layer.
    - `dropout_prob`: Dropout probability to apply after each hidden layer.
    - It dynamically creates a sequence of linear layers, ReLU activation functions, and dropout layers based on `hidden_layer_sizes`.
  - **`forward(self, x)`**: Defines the forward pass of the network. The input `x` is passed through the sequence of defined layers.

#### [`GRU_model_service.py`](./services/model_training/src/GRU_model_service.py)
This service is responsible for building and saving Gated Recurrent Unit (GRU) models.

- **Class: `GRUModelService`**
  - **`__init__(self)`**: Initializes the service, including a logger.
  - **`build_gru_model(self, params: dict)`**: Constructs a `GRUModel` instance. It expects `params` to contain `input_dim` (number of input features), `hidden_dim` (size of GRU hidden state), `layer_dim` (number of GRU layers), `output_dim` (number of output features), and `dropout_prob`.
  - **`save_model(self, model: GRUModel, model_path: str)`**: Saves the state dictionary of the trained `GRUModel` to the specified `model_path`.
  - **`create_and_save_gru_model(self, params: dict, model_path: str)`**: A utility method that builds a GRU model using `params` and then saves its initial state/architecture to `model_path`.

#### [`GRU_model.py`](./services/model_training/src/GRU_model.py)
Defines the PyTorch module for a Gated Recurrent Unit (GRU) network.

- **Class: `GRUModel(nn.Module)`**
  - **`__init__(self, input_size, hidden_units, num_layers, output_size=1, dropout_prob=0.0, device='cpu')`**: Constructor for the GRU model.
    - `input_size`: Number of input features per time step.
    - `hidden_units`: Number of features in the hidden state `h`.
    - `num_layers`: Number of recurrent GRU layers.
    - `output_size`: Number of output features (typically 1 for regression).
    - `dropout_prob`: Dropout probability for GRU layers (if `num_layers` > 1).
    - `device`: The device (CPU/GPU) the model's tensors should be on.
    - It initializes a `nn.GRU` layer and a `nn.Linear` layer for the output.
  - **`forward(self, x, h_0=None)`**: Defines the forward pass.
    - `x`: Input tensor of shape (batch_size, sequence_length, input_size).
    - `h_0`: Optional initial hidden state. If not provided, it's initialized to zeros.
    - The input `x` and hidden state `h_0` are passed through the GRU layer.
    - The output of the last time step from the GRU layer is passed through the final linear layer to produce the prediction.

#### [`LSTM_model_service.py`](./services/model_training/src/LSTM_model_service.py)
This file defines both the `LSTMModel` (PyTorch `nn.Module`) and the `LSTMModelService` for building and saving these models.

- **Class: `LSTMModel(nn.Module)`** (defined in [`LSTM_model_service.py`](vestim/services/model_training/src/LSTM_model_service.py:9))
  - **`__init__(self, input_size, hidden_units, num_layers, device, dropout_prob=0.0)`**: Constructor for the LSTM model.
    - `input_size`: Number of input features per time step.
    - `hidden_units`: Number of features in the hidden state `h` and cell state `c`.
    - `num_layers`: Number of recurrent LSTM layers.
    - `device`: The device (CPU/GPU) for tensor allocation.
    - `dropout_prob`: Dropout probability for LSTM layers (if `num_layers` > 1).
    - Initializes an `nn.LSTM` layer and an `nn.Linear` output layer.
  - **`forward(self, x, h_s=None, h_c=None)`**: Defines the forward pass.
    - `x`: Input tensor of shape (batch_size, sequence_length, input_size).
    - `h_s`, `h_c`: Optional initial hidden and cell states. If not provided, they are initialized to zeros.
    - Passes input and states through the LSTM layer.
    - The output of the last time step from the LSTM layer is passed through the linear layer.

- **Class: `LSTMModelService`** (defined in [`LSTM_model_service.py`](vestim/services/model_training/src/LSTM_model_service.py:48))
  - **`__init__(self)`**: Initializes the service, including a logger.
  - **`build_lstm_model(self, params: dict)`**: Constructs an `LSTMModel` instance. Expects `params` to include `input_dim`, `hidden_dim`, `layer_dim`, `output_dim`, `dropout_prob`, and `device`.
  - **`save_model(self, model, model_path: str)`**: Saves the state dictionary of the trained `LSTMModel` to `model_path`.
  - **`create_and_save_lstm_model(self, params: dict, model_path: str)`**: Utility to build and save an initial LSTM model.

#### [`LSTM_model_service_lnorm.py`](./services/model_training/src/LSTM_model_service_lnorm.py)
This file provides alternative/experimental LSTM model definitions, potentially focusing on Layer Normalization. It includes `VEstimLSTM` and another `LSTMModel`, plus a corresponding `LSTMModelService`.

- **Class: `VEstimLSTM(nn.Module)`**
  - **`__init__(self, hidden_size, input_size, layers)`**: Constructor for an LSTM model.
  - **`forward(self, x, h_s, h_c)`**: Defines the forward pass.

- **Class: `LSTMModel(nn.Module)`** (defined in [`LSTM_model_service_lnorm.py`](vestim/services/model_training/src/LSTM_model_service_lnorm.py:25))
  - **`__init__(self, input_size, hidden_units, num_layers, device)`**: Constructor for an LSTM model (no dropout parameter here).
  - **`forward(self, x, h_s, h_c)`**: Defines the forward pass.

- **Class: `LSTMModelService`** (defined in [`LSTM_model_service_lnorm.py`](vestim/services/model_training/src/LSTM_model_service_lnorm.py:58))
  - **`__init__(self)`**: Initializes the service.
  - **`build_lstm_model(self, params)`**: Builds an LSTM model (likely the `LSTMModel` from this file).
  - **`save_model(self, model, model_path)`**: Saves the model.
  - **`create_and_save_lstm_model(self, params, model_path)`**: Creates and saves an LSTM model.

#### [`LSTM_model_service_test.py`](./services/model_training/src/LSTM_model_service_test.py)
This file contains several LSTM model variants (`VEstimLSTM`, `LSTMModelBN` with Batch Normalization, `LSTMModelLN` with Layer Normalization, and a standard `LSTMModel`) along with an `LSTMModelService` capable of building these different types. This suggests a testing ground for various LSTM architectures.

- **Class: `VEstimLSTM(nn.Module)`**: Similar to the one in `LSTM_model_service_lnorm.py`.
  - **`__init__(self, hidden_size, input_size, layers)`**: Constructor.
  - **`forward(self, x, h_s, h_c)`**: Forward pass.

- **Class: `LSTMModelBN(nn.Module)`**: LSTM with Batch Normalization.
  - **`__init__(self, input_size, hidden_units, num_layers, device)`**: Constructor, includes `nn.BatchNorm1d` layer.
  - **`forward(self, x, h_s, h_c)`**: Forward pass, applies batch norm after LSTM.

- **Class: `LSTMModelLN(nn.Module)`**: LSTM with Layer Normalization.
  - **`__init__(self, input_size, hidden_units, num_layers, device)`**: Constructor, includes `nn.LayerNorm`.
  - **`forward(self, x, h_s, h_c)`**: Forward pass, applies layer norm after LSTM.

- **Class: `LSTMModel(nn.Module)`** (defined in [`LSTM_model_service_test.py`](vestim/services/model_training/src/LSTM_model_service_test.py:106)): Standard LSTM.
  - **`__init__(self, input_size, hidden_units, num_layers, device)`**: Constructor.
  - **`forward(self, x, h_s, h_c)`**: Forward pass.

- **Class: `LSTMModelService`** (defined in [`LSTM_model_service_test.py`](vestim/services/model_training/src/LSTM_model_service_test.py:140))
  - **`__init__(self)`**: Initializes the service.
  - **`build_lstm_model(self, params)`**: Builds a standard or Batch Norm LSTM based on `params` (logic might depend on presence of `use_bn` in params).
  - **`build_lstm_model_LN(self, params)`**: Builds an LSTM model with Layer Normalization.
  - **`save_model(self, model, model_path)`**: Saves any of the built models.
  - **`create_and_save_lstm_model(self, params, model_path)`**: Utility to build and save standard/BN LSTM.
  - **`create_and_save_lstm_model_with_LN(self, params, model_path)`**: Utility to build and save LayerNorm LSTM.

#### [`sequence_rnn_data_handler.py`](./services/model_training/src/sequence_rnn_data_handler.py)
This class inherits from `BaseDataHandler` and is specifically designed to prepare data for sequence-based RNN models (like LSTM, GRU).

- **Class: `SequenceRNNDataHandler(BaseDataHandler)`**
  - **`__init__(self, feature_cols, target_col, lookback, concatenate_raw_data=False)`**:
    - Calls the parent `__init__`.
    - `lookback`: Defines the number of previous time steps to use as input features for predicting the next time step.
    - `concatenate_raw_data`: Boolean, if true, all data files in the folder are concatenated before sequence creation. Otherwise, sequences are created per file and then concatenated.
  - **`_create_sequences_from_array(self, X_data_arr: np.ndarray, Y_data_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]`**:
    - Takes NumPy arrays of features (`X_data_arr`) and targets (`Y_data_arr`).
    - Creates input sequences (X) of shape (num_samples, `lookback`, num_features) and corresponding target sequences (y) of shape (num_samples, num_targets). Each sample `X[i]` contains `lookback` time steps of features, and `y[i]` is the target value at the step following the sequence `X[i]`.
  - **`load_and_process_data(self, folder_path: str, **kwargs) -> tuple[np.ndarray, np.ndarray]`**:
    - Reads all CSV files from the specified `folder_path`.
    - For each file (or for the concatenated data if `concatenate_raw_data` is true):
      - Selects the required `feature_cols` and `target_col`.
      - Calls `_create_sequences_from_array` to generate sequences.
    - Concatenates sequences from all files (if not already concatenated).
    - Returns the final X (features) and y (targets) NumPy arrays ready for training an RNN.

#### [`whole_sequence_fnn_data_handler.py`](./services/model_training/src/whole_sequence_fnn_data_handler.py)
This class inherits from `BaseDataHandler` and prepares data for FNN models where the entire sequence is treated as a single input vector.

- **Class: `WholeSequenceFNNDataHandler(BaseDataHandler)`**
  - **`__init__(self, feature_cols, target_col, lookback, concatenate_raw_data=False)`**:
    - Calls parent `__init__`.
    - `lookback`: Defines the length of the input sequence.
    - `concatenate_raw_data`: If true, data from all files in the folder is concatenated before processing.
  - **`load_and_process_data(self, folder_path: str, **kwargs) -> tuple[np.ndarray, np.ndarray]`**:
    - Reads CSV files from `folder_path`.
    - For each file (or concatenated data):
      - Selects `feature_cols` and `target_col`.
      - If `lookback` is greater than 0, it reshapes/flattens sequences of `lookback` length into single feature vectors. The target is the value at the end of each `lookback` sequence.
      - If `lookback` is 0 or not applicable, it might treat each row as an independent sample (depending on FNN architecture).
    - Returns X (features) and y (targets) as NumPy arrays. The features array will have a shape where the second dimension is `lookback * num_feature_cols` if lookback is used for flattening sequences.

#### [`training_task_service.py`](./services/model_training/src/training_task_service.py)
This service contains the core logic for executing a single training epoch and a validation epoch for a given model. It also handles some logging and scheduler retrieval.

- **Class: `TrainingTaskService`**
  - **`__init__(self)`**: Initializes the service, including a logger.
  - **`log_to_csv(self, task, epoch, batch_idx, batch_time, phase)`**: Logs batch-level timing information (e.g., time per batch) to a CSV file during training or validation.
  - **`log_to_sqlite(self, task, epoch, batch_idx, batch_time, phase, device)`**: Logs batch-level timing and device memory usage to an SQLite database.
  - **`train_epoch(self, model, model_type, train_loader, optimizer, h_s_initial, h_c_initial, epoch, device, stop_requested, task)`**:
    - Performs one epoch of training.
    - Iterates through `train_loader`.
    - For each batch:
      - Moves data to the specified `device`.
      - Performs forward pass (handles different input requirements for FNN vs RNNs like LSTM/GRU which need hidden states `h_s_initial`, `h_c_initial`).
      - Calculates loss (MSELoss is used).
      - Performs backward pass and optimizer step.
      - Logs batch processing time.
      - Checks `stop_requested` flag.
    - Returns average training loss for the epoch and updated hidden/cell states for RNNs.
  - **`validate_epoch(self, model, model_type, val_loader, h_s_initial, h_c_initial, epoch, device, stop_requested, task)`**:
    - Performs one epoch of validation.
    - Sets model to evaluation mode (`model.eval()`).
    - Iterates through `val_loader` with `torch.no_grad()`.
    - For each batch:
      - Moves data to device.
      - Performs forward pass.
      - Calculates loss.
      - Logs batch processing time.
      - Checks `stop_requested` flag.
    - Returns average validation loss for the epoch and updated hidden/cell states for RNNs.
  - **`save_model(self, model, model_path)`**: Saves the model's state dictionary to the given `model_path`.
  - **`get_scheduler(self, optimizer, lr_drop_period)`**: Creates and returns a `StepLR` learning rate scheduler. This scheduler adjusts the learning rate of the `optimizer` by a factor (gamma, typically 0.1) every `lr_drop_period` epochs, as per `torch.optim.lr_scheduler.StepLR` behavior. This method centralizes scheduler creation for training tasks.

## Conclusion

This developer guide provides an overview of the `vestim` project, its architecture, workflow, directory structure, and a detailed breakdown of key modules and their functionalities. It is intended to help developers quickly understand the codebase, navigate through different components, and contribute effectively to the project. For the most specific details, developers should always refer to the source code itself, including inline comments and docstrings within each module.

## Running the Application

The `vestim` application is a desktop GUI application. The typical entry point for the workflow is the Data Import screen.

1.  **Setup Environment**:
    *   Ensure you have Python installed (e.g., Python 3.8+).
    *   It is recommended to use a virtual environment:
        ```bash
        python -m venv venv
        source venv/bin/activate  # On Windows: venv\Scripts\activate
        ```
    *   Install dependencies:
        ```bash
        pip install -r requirements.txt
        ```
        (Note: Check if [`requirements_f.txt`](./requirements_f.txt:0) or [`pyproject.toml`](./pyproject.toml:0) are more appropriate for full setup, as they exist in the project structure).

2.  **Run the Application**:
    The application can be started by running the main data import GUI component, as described in its module documentation:
    ```bash
    python vestim/gui/src/data_import_gui_qt.py
    ```
    This will launch the "Data Import" window, as per the workflow, from which the user can proceed through the data processing and model training stages.

    Individual GUI components or services might also be runnable as standalone scripts if they include a `if __name__ == "__main__":` block with a `main()` function (as noted in their respective documentation sections), primarily for testing or development purposes.

## Key Dependencies

The `vestim` project relies on several core Python libraries for its functionality:

-   **PyQt5** (or potentially a compatible Qt binding like PySide): Used for building the graphical user interface components. The `_qt.py` suffix in many filenames and classes like [`QMainWindow`](./gui/src/data_import_gui_qt.py:0) and [`QObject`](./gui/src/data_augment_gui_qt.py:0) indicate Qt usage.
-   **Pandas**: Extensively used for data manipulation, loading CSV/Excel files (e.g., in [`DataProcessorArbin`](./services/data_processor/src/data_processor_qt_arbin.py:0)), and handling time-series data.
-   **NumPy**: For numerical computations, especially array operations underlying Pandas data structures and PyTorch tensors.
-   **scikit-learn**: Utilized for data preprocessing tasks such as normalization (e.g., `MinMaxScaler`, `StandardScaler` via [`normalization_service.py`](./services/normalization_service.py:0)) and for model evaluation metrics.
-   **PyTorch**: The primary machine learning framework used for defining (`nn.Module` subclasses like [`LSTMModel`](./services/model_training/src/LSTM_model_service.py:9)), training (e.g., in [`TrainingTaskService`](./services/model_training/src/training_task_service.py:0)), and testing neural network models (FNN, GRU, LSTM).
-   **Matplotlib**: For generating plots, such as training history (loss curves) and actual vs. predicted values in testing, often managed by GUI components like [`VEstimTestingGUI`](./gui/src/testing_gui_qt.py:0).
-   **Joblib**: For efficiently saving and loading Python objects, particularly scikit-learn scalers as seen in [`normalization_service.py`](./services/normalization_service.py:0).
-   **SQLite3**: Used for structured logging of training progress and testing results into local database files, managed by classes like [`TrainingTaskManager`](./gateway/src/training_task_manager_qt.py:0).

For a comprehensive list of all dependencies and their specific versions, please refer to the [`requirements.txt`](./requirements.txt:0) file in the root of the repository. Additional dependencies or project metadata might be found in [`pyproject.toml`](./pyproject.toml:0).