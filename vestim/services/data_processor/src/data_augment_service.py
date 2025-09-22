# ---------------------------------------------------------------------------------
# Author: Biswanath Dehury
# Date: `2025-04-14`
# Version: 1.0.0
# Description: 
# Service for data augmentation operations - provides core functionality to:
# 1. Load and save data from job folders
# 2. Resample data to specific frequencies
# 3. Create new columns based on formulas
# 4. Evaluate and validate formulas safely
# 
# This service handles the actual data processing operations
# ---------------------------------------------------------------------------------

import os
import io # Import io for string buffer
import re
import glob
import shutil
import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Dict, Optional, Union, Any
from scipy.signal import butter, filtfilt, lfilter

from vestim.logger_config import setup_logger
from vestim.gateway.src.job_manager_qt import JobManager # Import JobManager
from vestim.services import normalization_service as norm_svc # For normalization

# Set up logging
logger = setup_logger(log_file='data_augment_service.log')

class DataAugmentService:
    """Service class for data augmentation operations"""
    
    def __init__(self):
        """Initialize the DataAugmentService"""
        self.logger = logging.getLogger(__name__)
        self.job_manager = JobManager() # Instantiate JobManager

    def _set_job_context(self, job_folder: str):
        """Sets the JobManager's context to the given job_folder."""
        if not job_folder or not os.path.isdir(job_folder):
            self.logger.error(f"Invalid job_folder provided to _set_job_context: {job_folder}")
            return # Exit if job_folder is invalid

        job_id = os.path.basename(job_folder)
        if not job_id.startswith("job_"): # Basic validation
             self.logger.warning(f"Job folder '{job_id}' might not be a valid job ID format.")

        if self.job_manager.get_job_id() != job_id:
            self.logger.info(f"Setting JobManager's current job_id to: {job_id} (from path: {job_folder})")
            self.job_manager.job_id = job_id
        elif self.job_manager.get_job_folder() != job_folder :
            self.logger.info(f"JobManager's job_id '{job_id}' matches, but ensuring folder context is updated using path: {job_folder}")
            self.job_manager.job_id = job_id


    def load_processed_data(self, train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load processed training and testing data
        
        Args:
            train_path: Path to processed training data
            test_path: Path to processed testing data
            
        Returns:
            Tuple of (train_df, test_df)
        """
        self.logger.info(f"Loading processed data from {train_path} and {test_path}")
        
        train_files = glob.glob(os.path.join(train_path, "*.csv"))
        test_files = glob.glob(os.path.join(test_path, "*.csv"))
        
        if not train_files or not test_files:
            raise FileNotFoundError(f"No CSV files found in {train_path} or {test_path}")
        
        train_dfs = []
        for file in train_files:
            try:
                df = pd.read_csv(file)
                train_dfs.append(df)
                self.logger.info(f"Loaded train file: {file}, shape: {df.shape}")
            except Exception as e:
                self.logger.error(f"Error loading {file}: {e}")
        
        test_dfs = []
        for file in test_files:
            try:
                df = pd.read_csv(file)
                test_dfs.append(df)
                self.logger.info(f"Loaded test file: {file}, shape: {df.shape}")
            except Exception as e:
                self.logger.error(f"Error loading {file}: {e}")
        
        if train_dfs:
            train_df = pd.concat(train_dfs, ignore_index=True)
        else:
            raise ValueError("No valid training data found")
            
        if test_dfs:
            test_df = pd.concat(test_dfs, ignore_index=True)
        else:
            raise ValueError("No valid testing data found")
        
        self.logger.info(f"Successfully loaded data. Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        return train_df, test_df
    
    def resample_data(self, df: pd.DataFrame, frequency: str, progress_callback=None) -> pd.DataFrame:
        """
        Resample data to the specified frequency
        
        Args:
            df: DataFrame to resample
            frequency: Target frequency (e.g., "1Hz", "0.5Hz", etc.)
            progress_callback: Optional callback function to report progress
            
        Returns:
            Resampled DataFrame
        """
        self.logger.info(f"--- Entering resample_data for frequency: {frequency} ---")
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        self.logger.info(f"Initial DataFrame to resample_data:\n{info_str}\n{df.head().to_string()}")
        
        if progress_callback:
            progress_callback(10)
        
        if df.empty:
            self.logger.warning("Input DataFrame to resample_data is empty. Returning empty DataFrame.")
            if progress_callback: progress_callback(100)
            return pd.DataFrame()

        time_column = None
        # Prioritize common, exact names first, and ensure they have data
        preferred_candidates = ['Time', 'Timestamp', 'time', 'timestamp']
        for col_name in preferred_candidates:
            if col_name in df.columns and not df[col_name].isnull().all():
                time_column = col_name
                break
        
        # If no preferred candidate found, search more broadly for a time-like column with data
        if not time_column:
            for col in df.columns:
                col_lower = col.lower().replace(" ", "")
                if ('timestamp' in col_lower or 'time' in col_lower or 'date' in col_lower) and not df[col].isnull().all():
                    time_column = col
                    break
        
        if not time_column:
            self.logger.error("No time column found in DataFrame")
            if progress_callback: progress_callback(0)
            raise ValueError("DataFrame must contain a time-related column for resampling")
        
        self.logger.info(f"Using '{time_column}' as time column for resampling.")
        
        if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
            try:
                self.logger.info(f"Converting '{time_column}' to datetime. Assuming seconds since epoch.")
                df[time_column] = pd.to_datetime(df[time_column], unit='s', errors='coerce')
                if df[time_column].isna().all():
                    self.logger.error(f"All values in '{time_column}' became NaT. Cannot resample.")
                    if progress_callback: progress_callback(0)
                    raise ValueError(f"All values in time column '{time_column}' are invalid for datetime conversion.")
                df = df.dropna(subset=[time_column])
                if df.empty:
                    self.logger.error(f"DataFrame became empty after dropping NaT values from '{time_column}'.")
                    if progress_callback: progress_callback(0)
                    raise ValueError(f"Time column '{time_column}' resulted in empty DataFrame after NaT handling.")
            except Exception as e:
                self.logger.error(f"Error converting '{time_column}' to datetime: {e}", exc_info=True)
                if progress_callback: progress_callback(0)
                raise ValueError(f"Could not convert '{time_column}' to datetime: {e}")
        
        if df.empty:
            self.logger.warning("DataFrame is empty after time column processing. Returning empty DataFrame.")
            if progress_callback: progress_callback(100)
            return pd.DataFrame()

        if progress_callback: progress_callback(30)
        
        df_for_resampling = df.set_index(time_column, drop=True)
        
        if df_for_resampling.empty:
            self.logger.warning("df_for_resampling is empty before applying resample method.")
            if progress_callback: progress_callback(100)
            return pd.DataFrame()
            
        match = re.match(r'(\d*\.?\d*)Hz', frequency)
        if not match:
            self.logger.error(f"Invalid frequency format: {frequency}")
            if progress_callback: progress_callback(0)
            raise ValueError(f"Invalid frequency format: {frequency}. Expected format like '1Hz'")
        
        if progress_callback: progress_callback(50)
        
        freq_value = float(match.group(1))
        period_seconds = 1.0 / freq_value
        pandas_freq = f"{int(period_seconds)}S" if period_seconds >= 1 else f"{int(period_seconds * 1000)}L"
        self.logger.info(f"Calculated pandas resampling frequency: {pandas_freq}")
        
        try:
            if not df_for_resampling.index.is_monotonic_increasing:
                self.logger.warning(f"Time index for resampling is not monotonically increasing. Sorting index.")
                df_for_resampling = df_for_resampling.sort_index()
            
            # Apply ffill only. Subsequent padding must handle potential leading NaNs.
            resampled_df = df_for_resampling.resample(pandas_freq).ffill()
            self.logger.info(f"DataFrame after .resample().ffill(). Shape: {resampled_df.shape}. Head:\n{resampled_df.head().to_string()}")

            # Drop any leading rows that are all NaN after ffill (except for the time index)
            # This can happen if the resampling starts before the first actual data point.
            if not resampled_df.empty:
                # Check if all data columns (excluding the index if it's already reset) are NaN for the first row
                # If index is not yet reset, columns() gives data columns.
                # If index is already reset, time_column needs to be excluded.
                # At this stage, index is still the time column.
                if resampled_df.iloc[0].isnull().all():
                    self.logger.info(f"First row of resampled_df is all NaN. Attempting to drop leading NaN rows.")
                    # Find first valid (non-all-NaN) row
                    first_valid_index = resampled_df.dropna(how='all').index.min()
                    if pd.notna(first_valid_index):
                        resampled_df = resampled_df.loc[first_valid_index:]
                        self.logger.info(f"Shape after dropping leading NaN rows: {resampled_df.shape}. New Head:\n{resampled_df.head().to_string()}")
                    else:
                        self.logger.warning("Resampled DataFrame became all NaN after ffill. Returning empty.")
                        resampled_df = pd.DataFrame(columns=df_for_resampling.columns) # Keep columns for consistency

            if progress_callback: progress_callback(80)
            resampled_df = resampled_df.reset_index()
            self.logger.info(f"Resampling successful. Final shape: {resampled_df.shape}")
            if progress_callback: progress_callback(100)
            return resampled_df
            
        except Exception as e:
            self.logger.error(f"Error during resampling: {e}", exc_info=True)
            if progress_callback: progress_callback(0)
            raise ValueError(f"Resampling failed: {e}")
    
    @staticmethod
    def _generate_noise(mean, std, size):
        return np.random.normal(mean, std, size)

    @staticmethod
    def _moving_average(data_series, window):
        if not isinstance(data_series, pd.Series):
            data_series = pd.Series(data_series)
        return data_series.rolling(window=int(window), min_periods=1).mean()

    @staticmethod
    def _rolling_max(data_series, window):
        return data_series.rolling(window=int(window), min_periods=1).max()

    @staticmethod
    def _shift_series(data_series, periods):
        shifted_series = data_series.shift(periods=int(periods))
        
        # For delta calculations (like delta_t), we want to handle the first value properly
        # Instead of forward fill which corrupts the shift, fill first NaN with 0
        # This makes delta_t start from 0 (no change from previous) which is logical
        if periods > 0:  # Positive shift (looking backward)
            # Fill the first NaN values with the first actual value to make delta = 0
            first_valid_idx = shifted_series.first_valid_index()
            if first_valid_idx is not None:
                fill_value = data_series.iloc[0]  # Use first actual value
                shifted_series.iloc[:periods] = fill_value
        elif periods < 0:  # Negative shift (looking forward)
            # Fill the last NaN values with the last actual value to make delta = 0
            last_valid_idx = shifted_series.last_valid_index()
            if last_valid_idx is not None:
                fill_value = data_series.iloc[-1]  # Use last actual value
                shifted_series.iloc[periods:] = fill_value
        
        return shifted_series

    def validate_formula(self, formula: str, df: pd.DataFrame, log_details: bool = True) -> Tuple[bool, Optional[str]]:
        if log_details:
            self.logger.info(f"Validating formula: {formula}")
        forbidden_patterns = [r'__.*__', r'eval\s*\(', r'exec\s*\(', r'import\s+', r'open\s*\(', r'os\.', r'sys\.', r'subprocess\.', r'shutil\.']
        for pattern in forbidden_patterns:
            if re.search(pattern, formula):
                error_msg = f"Formula contains forbidden pattern: {pattern}"
                self.logger.warning(error_msg)
                return False, error_msg
        
        potential_cols = re.findall(r'[a-zA-Z_]\w*', formula)
        python_keywords = ['and', 'or', 'not', 'if', 'else', 'for', 'in', 'True', 'False', 'None']
        numpy_funcs = ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs', 'floor', 'ceil']
        allowed_refs = python_keywords + numpy_funcs + ['np', 'noise', 'moving_average', 'rolling_max', 'shift', 'butterworth_filter']
        potential_cols = [col for col in potential_cols if col not in allowed_refs]
        
        df_columns = df.columns.tolist()
        for col in potential_cols:
            if col not in df_columns:
                error_msg = f"Column '{col}' not found in the data."
                self.logger.warning(f"Formula references non-existent column: {col}")
                return False, error_msg
        
        try:
            sample_df = df.iloc[:1].copy()
            safe_globals = {
                "np": np,
                "noise": lambda mean, std: self._generate_noise(mean, std, len(sample_df)),
                "moving_average": lambda data, window: self._moving_average(data, window),
                "rolling_max": lambda data, window: self._rolling_max(data, window),
                "shift": lambda data, periods: self._shift_series(data, periods),
                "butterworth_filter": lambda data, corner_freq, sampling_rate, order=4: self.apply_butterworth_filter(sample_df, data, corner_freq, sampling_rate, order)
            }
            safe_locals = {col: sample_df[col] for col in sample_df.columns}
            result = eval(formula, safe_globals, safe_locals)
            if not hasattr(result, '__len__'):
                result = np.full(len(sample_df), result)
            if log_details:
                self.logger.info(f"Formula validated successfully: {formula}")
            return True, None
        except Exception as e:
            error_msg = f"Error evaluating formula: {e}"
            if isinstance(e, SyntaxError): error_msg = f"Invalid syntax in formula: {e}"
            elif isinstance(e, NameError): error_msg = f"Invalid name or function used in formula: {e}"
            elif isinstance(e, TypeError): error_msg = f"Type error during formula evaluation: {e}"
            self.logger.error(f"Formula validation failed during evaluation: {e}", exc_info=True)
            return False, error_msg
    
    def create_columns(self, df: pd.DataFrame,
                      column_formulas: List[Tuple[str, str]],
                      progress_callback=None, log_details: bool = True) -> pd.DataFrame:
        if log_details:
            self.logger.info(f"Creating {len(column_formulas)} new columns")
        if not column_formulas: return df
        if progress_callback: progress_callback(10)
        
        result_df = df.copy()
        progress_increment = 80 / len(column_formulas) if len(column_formulas) > 0 else 0
        
        for i, (column_name, formula) in enumerate(column_formulas):
            if log_details:
                self.logger.info(f"Creating column '{column_name}' with formula: {formula}")
            is_valid, error_detail = self.validate_formula(formula, result_df, log_details=log_details)
            if not is_valid:
                error_message = f"Invalid formula for column '{column_name}': {formula}. Reason: {error_detail}"
                self.logger.error(error_message)
                if progress_callback: progress_callback(0)
                raise ValueError(error_message)
            if progress_callback: progress_callback(10 + int((i + 0.5) * progress_increment))
            
            try:
                safe_globals = {
                    "np": np,
                    "noise": lambda mean, std: self._generate_noise(mean, std, len(result_df)),
                    "moving_average": lambda data, window: self._moving_average(result_df[data], window) if isinstance(data, str) else self._moving_average(data, window),
                    "rolling_max": lambda data, window: self._rolling_max(result_df[data], window) if isinstance(data, str) else self._rolling_max(data, window),
                    "shift": lambda data, periods: self._shift_series(result_df[data], periods) if isinstance(data, str) else self._shift_series(data, periods),
                    "butterworth_filter": lambda data, corner_freq, sampling_rate, order=4: self.apply_butterworth_filter(result_df, data, corner_freq, sampling_rate, order)
                }
                safe_locals = {col: result_df[col] for col in result_df.columns}
                result_df[column_name] = eval(formula, safe_globals, safe_locals)
                #self.logger.info(f"Successfully created column '{column_name}'")
                if progress_callback: progress_callback(10 + int((i + 1) * progress_increment))
            except Exception as e:
                self.logger.error(f"Error creating column '{column_name}': {e}", exc_info=True)
                if progress_callback: progress_callback(0)
                raise ValueError(f"Error creating column '{column_name}': {e}")
        
        if progress_callback: progress_callback(90)
        return result_df

    def pad_data(self, df: pd.DataFrame, padding_length: int, resample_freq_for_time_padding: Optional[str] = None) -> pd.DataFrame:
        """
        Prepends padding rows to the beginning of the DataFrame.

        Args:
            df: The input DataFrame.
            padding_length: The number of rows to prepend.
            resample_freq_for_time_padding: Optional resampling frequency string (e.g., "1Hz") 
                                             to calculate time decrements.
        Returns:
            The DataFrame with padding prepended.
        """
        if padding_length <= 0:
            self.logger.info("Padding length is zero or negative, skipping padding.")
            return df

        if df.empty:
            self.logger.warning("Input DataFrame is empty, cannot apply padding.")
            return df

        self.logger.info(f"Applying padding of {padding_length} rows with resample_freq_for_time_padding: {resample_freq_for_time_padding}")

        cols_to_pad_zero = ['Current', 'Power'] # Columns to pad with 0.0
        time_col_name = None
        
        padding_data = {}
        
        # Determine the first actual data row to use for non-zero/non-time padding values
        # This ensures that if the input df (after resampling) has leading NaNs, we pick the first *valid* data.
        first_data_row_for_padding = df.bfill().iloc[0] if not df.empty else None
        
        # The actual first row of the incoming df (could have NaNs) is used for time calculations.
        original_first_row = df.iloc[0] if not df.empty else None

        if original_first_row is None or first_data_row_for_padding is None:
            self.logger.error("DataFrame is effectively empty; cannot determine values for padding.")
            return df # Or raise an error

        # Identify time column
        for col_candidate in ['time', 'Time', 'timestamp', 'Timestamp']:
            if col_candidate in df.columns and pd.api.types.is_datetime64_any_dtype(df[col_candidate]):
                time_col_name = col_candidate
                break
        if not time_col_name:
             for col in df.columns:
                 if pd.api.types.is_datetime64_any_dtype(df[col]):
                     time_col_name = col
                     self.logger.info(f"Using inferred datetime column '{time_col_name}' for time padding.")
                     break
        
        time_delta_for_padding = pd.Timedelta(seconds=1)
        if time_col_name and time_col_name in df.columns: # Check if time_col_name was actually found
            first_time_val_anchor = original_first_row[time_col_name] # Anchor for time decrement
            if pd.notna(first_time_val_anchor):
                if resample_freq_for_time_padding:
                    match = re.match(r'(\d*\.?\d*)Hz', resample_freq_for_time_padding)
                    if match:
                        try:
                            freq_hz = float(match.group(1))
                            if freq_hz > 0: time_delta_for_padding = pd.Timedelta(seconds=1.0 / freq_hz)
                        except ValueError: self.logger.warning(f"Could not parse freq {resample_freq_for_time_padding} for time delta.")
                    else: self.logger.warning(f"Freq format {resample_freq_for_time_padding} not recognized for time delta.")
                elif len(df) >= 2:
                    second_time_val = df[time_col_name].iloc[1]
                    if pd.notna(second_time_val) and first_time_val_anchor != second_time_val:
                        inferred_delta = abs(first_time_val_anchor - second_time_val)
                        if inferred_delta > pd.Timedelta(0): time_delta_for_padding = inferred_delta
            else:
                self.logger.warning(f"First value in time column '{time_col_name}' is NaT. Using default 1s decrement or NaT for padding.")

        for col in df.columns:
            if col in cols_to_pad_zero:
                padding_data[col] = [0.0] * padding_length
            elif col == time_col_name:
                first_time_val_anchor = original_first_row[col]
                if pd.notna(first_time_val_anchor):
                    try:
                        padding_times = [first_time_val_anchor - (i * time_delta_for_padding) for i in range(padding_length, 0, -1)]
                        padding_data[col] = padding_times
                    except Exception as e_time_pad:
                        self.logger.warning(f"Could not create time sequence for padding column '{col}': {e_time_pad}. Using NaT.")
                        padding_data[col] = [pd.NaT] * padding_length
                else:
                    self.logger.warning(f"First value in time column '{col}' is NaT. Padding time with NaT.")
                    padding_data[col] = [pd.NaT] * padding_length
            else:
                padding_value = first_data_row_for_padding[col]
                padding_data[col] = [padding_value] * padding_length
                if pd.isna(padding_value):
                    self.logger.debug(f"Padding column '{col}' with NaN/NaT based on its first available non-NaN value.")


        padding_df = pd.DataFrame(padding_data, columns=df.columns) # Ensure column order

        padded_df = pd.concat([padding_df, df], ignore_index=True)
        self.logger.info(f"Padding applied. Original shape: {df.shape}, New shape: {padded_df.shape}")
        
        self.logger.info(f"Padded DataFrame head (first 5 rows):\n{padded_df.head(5).to_string()}")

        return padded_df

    def apply_normalization(self, df: pd.DataFrame, scaler: object, columns_to_normalize: List[str], normalize_all_numeric: bool = True) -> pd.DataFrame:
        """
        Applies normalization to the specified columns of the DataFrame using a pre-fitted scaler.

        Args:
            df (pd.DataFrame): The input DataFrame.
            scaler (object): The pre-fitted scaler object.
            columns_to_normalize (List[str]): List of column names that the scaler was fitted on.
            normalize_all_numeric (bool): If True, normalizes all numeric columns found in the scaler's features.
                                          If False, normalizes only `columns_to_normalize`.

        Returns:
            pd.DataFrame: The DataFrame with specified columns normalized.
        """
        # self.logger.info(f"Applying normalization with normalize_all_numeric={normalize_all_numeric}.") # Too verbose
        if df.empty:
            self.logger.warning("Input DataFrame to apply_normalization is empty. Returning empty DataFrame.")
            return df
        if not scaler:
            self.logger.error("No scaler object provided for normalization. Returning original DataFrame.")
            return df
        
        all_numeric_columns_in_df = df.select_dtypes(include=np.number).columns.tolist()
        
        # The scaler was fitted on `columns_to_normalize`. We pass this as `feature_columns`.
        # The columns we want to *apply* the transform to is determined by `normalize_all_numeric`.
        
        cols_to_apply_transform = all_numeric_columns_in_df if normalize_all_numeric else columns_to_normalize

        try:
            transformed_df = norm_svc.transform_data(
                data_df=df,
                scaler=scaler,
                feature_columns=columns_to_normalize, # The columns the scaler was FIT on
                all_numeric_columns=cols_to_apply_transform # The columns to APPLY the transform to
            )
            # self.logger.info("Normalization function executed.") # Too verbose
            return transformed_df
        except Exception as e:
            self.logger.error(f"Error during data normalization in service: {e}", exc_info=True)
            return df
    
    def save_single_augmented_file(self, augmented_df: pd.DataFrame, output_filepath: str):
        # self.logger.info(f"Saving augmented DataFrame to: {output_filepath}") # Too verbose
        try:
            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
            augmented_df.to_csv(output_filepath, index=False)
            # self.logger.info(f"Successfully saved augmented file: {output_filepath}") # Too verbose
        except Exception as e:
            self.logger.error(f"Failed to save augmented file {output_filepath}: {e}", exc_info=True)
            raise 

    def update_augmentation_metadata(self, job_folder: str, processed_files_info: List[Dict[str, Any]], 
                                   filter_configs: Optional[List[Dict[str, Any]]] = None, 
                                   normalization_info: Optional[Dict[str, Any]] = None, 
                                   column_formulas: Optional[List[Tuple[str, str]]] = None,
                                   resampling_info: Optional[Dict[str, Any]] = None,
                                   padding_info: Optional[Dict[str, Any]] = None):
        self.logger.info(f"Updating augmentation metadata for job: {job_folder}")
        self._set_job_context(job_folder)

        json_metadata_path = os.path.join(job_folder, 'augmentation_metadata.json')

        # Convert column formulas to structured format
        created_columns = []
        if column_formulas:
            for column_name, formula in column_formulas:
                created_columns.append({
                    "column_name": column_name,
                    "formula": formula,
                    "type": "calculated"
                })

        # Create comprehensive structured metadata
        try:
            structured_metadata = {
                "job_id": self.job_manager.get_job_id(),
                "augmentation_run_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "applied_filters": filter_configs if filter_configs else [],
                "created_columns": created_columns,
                "resampling": resampling_info if resampling_info else {"applied": False},
                "padding": padding_info if padding_info else {"applied": False},
                "normalization": normalization_info if normalization_info else {"applied": False},
                "processed_files": processed_files_info,
                "total_files_processed": len(processed_files_info),
                "metadata_version": "2.0"  # Version for future compatibility
            }
            
            import json
            with open(json_metadata_path, 'w') as f:
                json.dump(structured_metadata, f, indent=2, default=str)
            
            self.logger.info(f"Successfully saved comprehensive augmentation metadata at {json_metadata_path}")
        except Exception as e:
            self.logger.error(f"Could not save structured metadata file at {json_metadata_path}: {e}", exc_info=True)
    
    def get_column_info(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        self.logger.info(f"Getting column info for DataFrame with shape {df.shape}")
        column_info = {}
        for column in df.columns:
            column_data = df[column]
            column_type = str(column_data.dtype)
            info = {'name': column, 'type': column_type, 'count': len(column_data),
                    'non_null_count': column_data.count(), 'null_count': column_data.isna().sum()}
            if pd.api.types.is_numeric_dtype(column_data):
                info.update({'min': column_data.min(), 'max': column_data.max(), 'mean': column_data.mean(),
                             'std': column_data.std(), 'median': column_data.median()})
            if pd.api.types.is_object_dtype(column_data) or pd.api.types.is_categorical_dtype(column_data):
                value_counts = column_data.value_counts()
                if len(value_counts) <= 10: info['unique_values'] = value_counts.to_dict()
                info['unique_count'] = len(value_counts)
            column_info[column] = info
        self.logger.info(f"Collected info for {len(column_info)} columns")
        return column_info

    def apply_butterworth_filter_non_causal(self, df: pd.DataFrame, column_name: str, corner_frequency: float, sampling_rate: float, filter_order: int = 4, output_column_name: str = None) -> pd.DataFrame:
        """
        Apply a NON-CAUSAL Butterworth filter to a specific column using filtfilt (zero-phase filtering).
        
        WARNING: This method uses future data points and is NOT suitable for real-time applications.
        Use apply_butterworth_filter() for causal filtering in real-time systems.
        
        If output_column_name is provided, a new column is created. Otherwise, the original column is overwritten.
        
        Args:
            df: The input DataFrame.
            column_name: The name of the column to filter.
            corner_frequency: The corner frequency for the filter.
            sampling_rate: The sampling rate of the data in Hz.
            filter_order: The order of the Butterworth filter.
            output_column_name: The name of the new column for the filtered data.
            
        Returns:
            The DataFrame with the filtered column.
        """
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame.")
            
        data = df[column_name].values
        
        nyquist = 0.5 * sampling_rate
        if corner_frequency >= nyquist:
            raise ValueError(f"Corner frequency ({corner_frequency}Hz) must be less than the Nyquist frequency ({nyquist}Hz).")
        
        normal_corner = corner_frequency / nyquist
        b, a = butter(filter_order, normal_corner, btype='low', analog=False)
        
        # NON-CAUSAL: Uses future data points (zero-phase filtering)
        filtered_data = filtfilt(b, a, data)
        
        target_column = output_column_name if output_column_name else column_name
        df[target_column] = filtered_data
        
        return df

    def apply_butterworth_filter(self, df: pd.DataFrame, column_name: str, corner_frequency: float, sampling_rate: float, filter_order: int = 4, output_column_name: str = None) -> pd.DataFrame:
        """
        Apply a CAUSAL Butterworth filter to a specific column using lfilter.
        
        This method is suitable for real-time applications as it only uses past and current data points.
        The filtering matches what would happen during real-time inference.
        
        If output_column_name is provided, a new column is created. Otherwise, the original column is overwritten.
        
        Args:
            df: The input DataFrame.
            column_name: The name of the column to filter.
            corner_frequency: The corner frequency for the filter.
            sampling_rate: The sampling rate of the data in Hz.
            filter_order: The order of the Butterworth filter.
            output_column_name: The name of the new column for the filtered data.
            
        Returns:
            The DataFrame with the filtered column.
        """
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame.")
            
        data = df[column_name].values
        
        nyquist = 0.5 * sampling_rate
        if corner_frequency >= nyquist:
            raise ValueError(f"Corner frequency ({corner_frequency}Hz) must be less than the Nyquist frequency ({nyquist}Hz).")
        
        normal_corner = corner_frequency / nyquist
        b, a = butter(filter_order, normal_corner, btype='low', analog=False)
        
        # CAUSAL: Only uses past and current data points (suitable for real-time)
        filtered_data = lfilter(b, a, data)
        
        target_column = output_column_name if output_column_name else column_name
        df[target_column] = filtered_data
        
        # Filter applied successfully
        
        return df

    def apply_noise_injection(self, df: pd.DataFrame, column_name: str, noise_type: str, noise_level_percent: float) -> pd.DataFrame:
        """
        Apply noise injection to a specific column for training robustness.
        
        Args:
            df: The input DataFrame.
            column_name: The name of the column to add noise to.
            noise_type: Type of noise ('gaussian' or 'uniform').
            noise_level_percent: Percentage of column's standard deviation to use as noise level.
            
        Returns:
            pd.DataFrame: DataFrame with noise added to the specified column.
        """
        if column_name not in df.columns:
            self.logger.warning(f"Column '{column_name}' not found in DataFrame. Skipping noise injection.")
            return df
            
        if df[column_name].dtype not in ['float64', 'float32', 'int64', 'int32']:
            self.logger.warning(f"Column '{column_name}' is not numeric. Skipping noise injection.")
            return df
            
        try:
            column_data = df[column_name].values
            
            # Calculate noise scale based on column's standard deviation
            column_std = np.std(column_data)
            if column_std == 0:
                self.logger.warning(f"Column '{column_name}' has zero variance. Skipping noise injection.")
                return df
                
            # Convert percentage to actual noise level
            noise_scale = column_std * (noise_level_percent / 100.0)
            
            if noise_type == 'gaussian':
                # Add Gaussian noise with specified standard deviation
                noise = np.random.normal(0, noise_scale, len(column_data))
            elif noise_type == 'uniform':
                # Add uniform noise in [-noise_scale, +noise_scale] range
                noise = np.random.uniform(-noise_scale, noise_scale, len(column_data))
            else:
                self.logger.error(f"Unknown noise type '{noise_type}'. Supported types: 'gaussian', 'uniform'")
                return df
                
            # Add noise to the column
            df[column_name] = column_data + noise
            
            self.logger.info(f"Applied {noise_type} noise ({noise_level_percent}% of std dev = {noise_scale:.6f}) to column '{column_name}'")
            return df
            
        except Exception as e:
            self.logger.error(f"Error applying noise injection to column '{column_name}': {e}", exc_info=True)
            return df
