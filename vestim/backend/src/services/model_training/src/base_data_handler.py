from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BaseDataHandler(ABC):
    """
    Abstract base class for data handlers.
    Each handler is responsible for loading and processing data 
    from a folder into a format suitable for creating PyTorch DataLoaders.
    """

    def __init__(self, feature_cols, target_col):
        """
        Initialize the handler with feature and target column information.

        :param feature_cols: List of strings, names of the feature columns.
        :param target_col: String, name of the target column.
        """
        if not isinstance(feature_cols, list) or not all(isinstance(col, str) for col in feature_cols):
            raise ValueError("feature_cols must be a list of strings.")
        if not isinstance(target_col, str):
            raise ValueError("target_col must be a string.")
            
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.logger = None # Placeholder, to be set by DataLoaderService or if handlers log independently

    @abstractmethod
    def load_and_process_data(self, folder_path: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """
        Load data from CSV files in the given folder path, process it
        according to the specific handler's logic, and return X and y numpy arrays.

        :param folder_path: Path to the folder containing CSV files.
        :param kwargs: Additional arguments specific to the handler (e.g., lookback for RNNs).
        :return: A tuple (X_data_processed, y_data_processed) as numpy arrays.
                 Shape of X depends on the handler (e.g., [N, lookback, features] or [N, features]).
                 Shape of y is typically [N, num_output_features] or [N,].
        """
        pass

    def _read_and_select_columns(self, file_path: str) -> pd.DataFrame | None:
        """
        Helper method to read a CSV and select specified feature and target columns.
        Handles potential errors during file reading or column selection.
        """
        try:
            df = pd.read_csv(file_path)
            # Ensure all specified feature columns and target column exist
            required_cols = self.feature_cols + [self.target_col]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                if self.logger:
                    self.logger.error(f"Missing required columns in {file_path}: {missing_cols}")
                else:
                    print(f"ERROR: Missing required columns in {file_path}: {missing_cols}")
                return None
            
            # Select only the required columns to reduce memory early
            df_selected = df[required_cols].copy() # Use .copy() to avoid SettingWithCopyWarning later
            return df_selected
        except FileNotFoundError:
            if self.logger:
                self.logger.error(f"File not found: {file_path}")
            else:
                print(f"ERROR: File not found: {file_path}")
            return None
        except pd.errors.EmptyDataError:
            if self.logger:
                self.logger.warning(f"Empty data file: {file_path}")
            else:
                print(f"WARNING: Empty data file: {file_path}")
            return None
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error reading CSV {file_path}: {e}")
            else:
                print(f"ERROR: Error reading CSV {file_path}: {e}")
            return None