# ---------------------------------------------------------------------------------
# Author: Biswanath Dehury
# Date: `{{date:2025-03-02}}`
# Version: 1.0.0
# Description: This is exactly same as DataProcessorSTLA now and changes in the future may be brought for this according to requirements.
# ---------------------------------------------------------------------------------


import os
import shutil
import h5py
import scipy.io as sio # Re-add for fallback
import numpy as np
import gc  # Explicit garbage collector
from vestim.gateway.src.job_manager_qt import JobManager
from tqdm import tqdm
import pandas as pd

import logging

class DataProcessorArbin:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.job_manager = JobManager()
        self.total_files = 0  # Total number of files to process (copy + convert)
        self.processed_files = 0  # Keep track of total processed files

    def organize_and_convert_files(self, train_files, test_files, progress_callback=None, sampling_frequency=None):
        # Ensure `total_files` is calculated upfront and is non-zero
        self.logger.info("Starting file organization and conversion.")
        # Count initial files for copying
        self.total_files = len(train_files) + len(test_files)

        if self.total_files == 0:
            self.logger.error("No files to process.")
            raise ValueError("No files to process.")

        job_id, job_folder = self.job_manager.create_new_job()
        self.logger.info(f"Job created with ID: {job_id}, Folder: {job_folder}")

        # Switch logger to job-specific log file
        job_log_file = os.path.join(job_folder, 'job.log')
        self.switch_log_file(job_log_file)

        # Create directories for raw and processed data
        train_raw_folder = os.path.join(job_folder, 'train_data', 'raw_data')
        train_processed_folder = os.path.join(job_folder, 'train_data', 'processed_data')
        test_raw_folder = os.path.join(job_folder, 'test_data', 'raw_data')
        test_processed_folder = os.path.join(job_folder, 'test_data', 'processed_data')

        os.makedirs(train_raw_folder, exist_ok=True)
        os.makedirs(train_processed_folder, exist_ok=True)
        os.makedirs(test_raw_folder, exist_ok=True)
        os.makedirs(test_processed_folder, exist_ok=True)
        self.logger.info(f"Created folders: {train_raw_folder}, {train_processed_folder}, {test_raw_folder}, {test_processed_folder}")

        # Reset processed files counter before processing starts
        self.processed_files = 0

        # Process copying and converting files
        self._copy_files(train_files, train_raw_folder, progress_callback)
        self._copy_files(test_files, test_raw_folder, progress_callback)

        # Increment total file count for files that need conversion
        supported_extensions = ('.mat', '.csv', '.xlsx', '.xls')
        self.total_files += len([f for f in os.listdir(train_raw_folder) if f.lower().endswith(supported_extensions)])
        self.total_files += len([f for f in os.listdir(test_raw_folder) if f.lower().endswith(supported_extensions)])

        self.logger.info(f"Starting file conversion for relevant files.")

        # **Check if resampling is needed**
        if sampling_frequency is None:
            self.logger.info("No resampling selected. Performing standard conversion.")
            self._convert_files(train_raw_folder, train_processed_folder, progress_callback)
            self._convert_files(test_raw_folder, test_processed_folder, progress_callback)
        else:
            self.logger.info(f"Resampling enabled. Resampling frequency: {sampling_frequency} Hz")
            self._convert_and_resample_files(train_raw_folder, train_processed_folder, progress_callback, sampling_frequency)
            self._convert_and_resample_files(test_raw_folder, test_processed_folder, progress_callback, sampling_frequency)

        return job_folder

    
    def switch_log_file(self, log_file):
        # Remove the current file handler(s)
        for handler in self.logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                self.logger.removeHandler(handler)

        # Add a new file handler to the logger for the job-specific log file
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)

        self.logger.info(f"Switched logging to {log_file}")

    def _copy_files(self, files, destination_folder, progress_callback=None):
        """ Copy the files to a destination folder and update progress. """
        processed_files = 0  # Track the number of processed files

        self.logger.info(f"Copying {len(files)} files to {destination_folder}")
        for file_path in files:
            dest_path = os.path.join(destination_folder, os.path.basename(file_path))
            shutil.copy(file_path, dest_path)
            self.logger.info(f"Copied {file_path} to {dest_path}")

            # Update progress based on the number of files processed
            processed_files += 1
            self.processed_files += 1  # Update the overall processed files count
            self._update_progress(progress_callback)

    def _convert_files(self, input_folder, output_folder, progress_callback=None, sampling_frequency=1):
        """ Convert files from .mat, .csv, .xlsx to .csv and update progress. """
        for root, _, files in os.walk(input_folder):
            
            self.logger.info(f"Converting files in folder: {input_folder} to .csv")
            for file in tqdm(files, desc="Converting files"):
                file_path = os.path.join(root, file)
                if file.lower().endswith('.mat'):
                    self._convert_mat_to_csv(file_path, output_folder)
                    self.logger.info(f"Converted {file_path} to CSV")
                    self.processed_files += 1
                    self._update_progress(progress_callback)
                elif file.lower().endswith('.csv'):
                    self._convert_csv_to_csv(file_path, output_folder)
                    self.logger.info(f"Converted {file_path} to CSV")
                    self.processed_files += 1
                    self._update_progress(progress_callback)
                elif file.lower().endswith(('.xlsx', '.xls')):
                    self._convert_excel_to_csv(file_path, output_folder)
                    self.logger.info(f"Converted {file_path} to CSV")
                    self.processed_files += 1
                    self._update_progress(progress_callback)

                # Explicitly clear memory after each conversion
                gc.collect()  # Explicit garbage collection after processing each file

    def _convert_and_resample_files(self, input_folder, output_folder, progress_callback=None, sampling_frequency='1S'):
        """ Convert files from .mat, .csv, .xlsx to .csv, resample, and update progress. """
        for root, _, files in os.walk(input_folder):
            
            self.logger.info(f"Converting and resampling files in folder: {input_folder} to .csv")
            for file in tqdm(files, desc="Converting and resampling files"):
                file_path = os.path.join(root, file)
                if file.lower().endswith('.mat'):
                    self._convert_mat_to_csv_resampled(file_path, output_folder, sampling_frequency)
                    self.logger.info(f"Converted and resampled {file_path} to CSV")
                    self.processed_files += 1
                    self._update_progress(progress_callback)
                elif file.lower().endswith('.csv'):
                    # Assuming CSVs might also need resampling
                    self._convert_csv_to_csv_resampled(file_path, output_folder, sampling_frequency)
                    self.logger.info(f"Converted and resampled {file_path} to CSV")
                    self.processed_files += 1
                    self._update_progress(progress_callback)
                elif file.lower().endswith(('.xlsx', '.xls')):
                    # Assuming Excels might also need resampling
                    self._convert_excel_to_csv_resampled(file_path, output_folder, sampling_frequency)
                    self.logger.info(f"Converted and resampled {file_path} to CSV")
                    self.processed_files += 1
                    self._update_progress(progress_callback)

                # Explicitly clear memory after each conversion
                gc.collect()  # Explicit garbage collection after processing each file
    
    def _convert_mat_to_csv(self, mat_file, output_folder):
        """ Convert .mat file to .csv and delete large arrays after processing. """
        # Extract data
        df = self.extract_data_from_matfile(mat_file)
        if df is None:
            print("Failed to extract data.")
            return
        csv_file_name = os.path.join(output_folder, os.path.splitext(os.path.basename(mat_file))[0] + '.csv')

        # Save the CSV
        df.to_csv(csv_file_name, index=False)
        print(f'Data successfully written to {csv_file_name}')

        # Delete large arrays to free memory
        del df
        gc.collect()  # Force garbage collection
        self.logger.info(f"Converted mat file to CSV and saved in processed folder.")

    def _convert_csv_to_csv(self, csv_file_path, output_folder):
        """Convert CSV file to a standardized CSV format."""
        try:
            df = pd.read_csv(csv_file_path)
            
            # No column filtering or renaming, save as is.
            df_processed = df
            
            csv_file_name = os.path.join(output_folder, os.path.splitext(os.path.basename(csv_file_path))[0] + '.csv')
            df_processed.to_csv(csv_file_name, index=False)
            self.logger.info(f"Successfully converted {csv_file_path} to {csv_file_name}")

        except Exception as e:
            self.logger.error(f"Error converting CSV file {csv_file_path}: {e}")
        finally:
            del df
            gc.collect()

    def _convert_excel_to_csv(self, excel_file_path, output_folder):
        """Convert Excel file to a standardized CSV format."""
        try:
            # Reading the first sheet by default
            df = pd.read_excel(excel_file_path, sheet_name=0)
            
            # No column filtering or renaming, save as is.
            df_processed = df

            csv_file_name = os.path.join(output_folder, os.path.splitext(os.path.basename(excel_file_path))[0] + '.csv')
            df_processed.to_csv(csv_file_name, index=False)
            self.logger.info(f"Successfully converted {excel_file_path} to {csv_file_name}")

        except Exception as e:
            self.logger.error(f"Error converting Excel file {excel_file_path}: {e}")
        finally:
            del df
            gc.collect()

    def _convert_mat_to_csv_resampled(self, mat_file, output_folder, sampling_frequency=1):
        """
        Full workflow to convert a .mat file to a resampled CSV file.

        Parameters:
        mat_file (str): Path to the .mat file.
        output_csv (str): Output CSV file path.
        target_freq (str): Frequency for resampling (default: '1S' for 1Hz).
        """
        print(f"Processing file: {mat_file}")

        # Step 1: Extract data
        df = self.extract_data_from_matfile(mat_file)
        if df is None:
            print("Failed to extract data.")
            return

        # Step 2: Resample data
        df_resampled = self._resample_data(df, sampling_frequency)
        if df_resampled is None:
            print("Failed to resample data.")
            return
        
        csv_file_name = os.path.join(output_folder, os.path.splitext(os.path.basename(mat_file))[0] + '.csv')

        # Save the CSV
        df_resampled.to_csv(csv_file_name, index=False)
        print(f'Data successfully written to {csv_file_name}')

        # Delete large arrays to free memory
        del df, df_resampled
        gc.collect()  # Force garbage collection
        self.logger.info(f"Converted mat file to CSV and saved in processed folder.")
        
    def _convert_csv_to_csv_resampled(self, csv_file_path, output_folder, sampling_frequency='1S'):
        """Converts a CSV file to a standardized, resampled CSV file."""
        try:
            df = pd.read_csv(csv_file_path)

            # Minimal column mapping for Timestamp if necessary, otherwise keep original names
            # Prefer 'Timestamp' if available, else 'Time'
            if 'Timestamp' not in df.columns and 'Time' in df.columns:
                df.rename(columns={'Time': 'Timestamp'}, inplace=True)
            elif 'timestamp' in df.columns: # common alternative
                 df.rename(columns={'timestamp': 'Timestamp'}, inplace=True)
            # Add other common time column names if needed

            # Ensure 'Timestamp' column exists for resampling
            if 'Timestamp' not in df.columns:
                self.logger.error(f"A recognizable time column ('Timestamp', 'Time', 'timestamp') not found in {csv_file_path}. Skipping resampling.")
                # Save as is (already handled by _convert_csv_to_csv to not filter)
                self._convert_csv_to_csv(csv_file_path, output_folder)
                return
            
            # No explicit column filtering here, pass all columns to _resample_data
            df_processed = df

            df_resampled = self._resample_data(df_processed.copy(), sampling_frequency) # Pass a copy
            if df_resampled is None:
                self.logger.error(f"Failed to resample data from {csv_file_path}. Saving non-resampled.")
                # Fallback: save the processed but non-resampled data
                csv_file_name = os.path.join(output_folder, os.path.splitext(os.path.basename(csv_file_path))[0] + '_processed.csv')
                df_processed.to_csv(csv_file_name, index=False)
                return

            # If 'Time' was used for resampling and needs to be 'Timestamp' in output:
            # df_resampled.rename(columns={'Time': 'Timestamp'}, inplace=True)

            csv_file_name = os.path.join(output_folder, os.path.splitext(os.path.basename(csv_file_path))[0] + '.csv')
            df_resampled.to_csv(csv_file_name, index=False)
            self.logger.info(f"Successfully converted and resampled {csv_file_path} to {csv_file_name}")

        except Exception as e:
            self.logger.error(f"Error converting and resampling CSV file {csv_file_path}: {e}")
        finally:
            if 'df' in locals(): del df
            if 'df_processed' in locals(): del df_processed
            if 'df_resampled' in locals(): del df_resampled
            gc.collect()

    def _convert_excel_to_csv_resampled(self, excel_file_path, output_folder, sampling_frequency='1S'):
        """Converts an Excel file to a standardized, resampled CSV file."""
        try:
            df = pd.read_excel(excel_file_path, sheet_name=0)

            # Minimal column mapping for Timestamp if necessary
            if 'Timestamp' not in df.columns and 'Time' in df.columns:
                df.rename(columns={'Time': 'Timestamp'}, inplace=True)
            elif 'timestamp' in df.columns:
                 df.rename(columns={'timestamp': 'Timestamp'}, inplace=True)

            if 'Timestamp' not in df.columns:
                self.logger.error(f"A recognizable time column ('Timestamp', 'Time', 'timestamp') not found in {excel_file_path}. Skipping resampling.")
                self._convert_excel_to_csv(excel_file_path, output_folder) # Fallback
                return

            # No explicit column filtering here
            df_processed = df

            df_resampled = self._resample_data(df_processed.copy(), sampling_frequency)
            if df_resampled is None:
                self.logger.error(f"Failed to resample data from {excel_file_path}. Saving non-resampled.")
                csv_file_name = os.path.join(output_folder, os.path.splitext(os.path.basename(excel_file_path))[0] + '_processed.csv')
                df_processed.to_csv(csv_file_name, index=False)
                return
            
            # df_resampled.rename(columns={'Time': 'Timestamp'}, inplace=True) # If 'Timestamp' is preferred output

            csv_file_name = os.path.join(output_folder, os.path.splitext(os.path.basename(excel_file_path))[0] + '.csv')
            df_resampled.to_csv(csv_file_name, index=False)
            self.logger.info(f"Successfully converted and resampled {excel_file_path} to {csv_file_name}")

        except Exception as e:
            self.logger.error(f"Error converting and resampling Excel file {excel_file_path}: {e}")
        finally:
            if 'df' in locals(): del df
            if 'df_processed' in locals(): del df_processed
            if 'df_resampled' in locals(): del df_resampled
            gc.collect()

    def extract_data_from_matfile(self, file_path):
        """
        Extracts all 1D-convertible datasets from a .mat file (HDF5 format) and returns them as a DataFrame.
        
        Parameters:
        file_path (str): Path to the .mat file.

        Returns:
        pd.DataFrame: DataFrame with the extracted data or None if extraction fails.
        """
        data_dict = {}
        processed_with_h5py = False
        try:
            # Attempt with h5py first (for MAT v7.3+)
            self.logger.info(f"Attempting to read .mat file {file_path} with h5py...")
            with h5py.File(file_path, 'r') as mat_file:
                if 'meas' not in mat_file:
                    self.logger.warning(f"'meas' structure not found in .mat file {file_path} using h5py. Checking top-level.")
                    for key, item in mat_file.items(): # Fallback to top-level for h5py
                        if isinstance(item, h5py.Dataset):
                            try:
                                data = item[()]
                                if data.dtype.kind in ['S', 'O']: continue
                                if data.ndim == 1: data_dict[key] = data
                                elif data.ndim == 2 and (data.shape[0] == 1 or data.shape[1] == 1): data_dict[key] = data.flatten()
                                elif data.ndim == 0: data_dict[key] = np.array([data])
                            except Exception: pass # ignore problematic datasets at top level
                else:
                    meas_group = mat_file['meas']
                    for field_name, field_dataset in meas_group.items():
                        if isinstance(field_dataset, h5py.Dataset):
                            try:
                                data = field_dataset[()]
                                if data.dtype.kind in ['S', 'O']: continue
                                if data.ndim == 1: data_dict[field_name] = data
                                elif data.ndim == 2 and (data.shape[0] == 1 or data.shape[1] == 1): data_dict[field_name] = data.flatten()
                                elif data.ndim == 0: data_dict[field_name] = np.array([data])
                            except Exception: pass # ignore problematic fields
            if data_dict:
                processed_with_h5py = True
                self.logger.info(f"Successfully extracted data using h5py from {file_path}.")

        except OSError as e_h5py: # Catch h5py specific file opening errors (like signature not found)
            self.logger.warning(f"h5py failed to open {file_path} (likely not a HDF5-based .mat file): {e_h5py}. Attempting fallback with scipy.io.loadmat.")
        except Exception as e_h5py_other: # Catch other h5py errors during processing
            self.logger.error(f"An unexpected error occurred with h5py for {file_path}: {e_h5py_other}. Attempting fallback with scipy.io.loadmat.")

        if not processed_with_h5py or not data_dict: # If h5py failed or found no data, try scipy.io
            data_dict = {} # Reset data_dict for scipy
            try:
                self.logger.info(f"Attempting to read .mat file {file_path} with scipy.io.loadmat...")
                mat_data_sio = sio.loadmat(file_path, simplify_cells=True) # simplify_cells can help with struct access
                
                if 'meas' not in mat_data_sio:
                    self.logger.warning(f"'meas' structure not found in .mat file {file_path} using scipy.io.loadmat. Checking top-level variables.")
                    # Fallback for scipy: iterate through top-level variables
                    for key, value in mat_data_sio.items():
                        if key.startswith('__'): continue # Skip metadata keys
                        if isinstance(value, np.ndarray):
                            if np.issubdtype(value.dtype, np.number): # Check if numeric
                                if value.ndim == 1: data_dict[key] = value
                                elif value.ndim == 2 and (value.shape[0] == 1 or value.shape[1] == 1): data_dict[key] = value.flatten()
                                elif value.ndim == 0 : data_dict[key] = np.array([value]) # scalar
                else:
                    meas_struct_sio = mat_data_sio['meas']
                    if isinstance(meas_struct_sio, dict): # simplify_cells=True makes structs dict-like
                        for field_name, data_array in meas_struct_sio.items():
                            if isinstance(data_array, np.ndarray):
                                if np.issubdtype(data_array.dtype, np.number): # Check if numeric
                                    if data_array.ndim == 1: data_dict[field_name] = data_array
                                    elif data_array.ndim == 2 and (data_array.shape[0] == 1 or data_array.shape[1] == 1): data_dict[field_name] = data_array.flatten()
                                    elif data_array.ndim == 0 : data_dict[field_name] = np.array([data_array])
                    else: # Should not happen with simplify_cells=True, but as a fallback
                         self.logger.warning(f"'meas' in {file_path} with scipy.io was not a dict as expected (simplify_cells=True).")


                if data_dict:
                    self.logger.info(f"Successfully extracted data using scipy.io.loadmat from {file_path}.")
                else:
                    self.logger.warning(f"No suitable numeric datasets found using scipy.io.loadmat for {file_path}.")
                    return None # Both methods failed to find data

            except Exception as e_sio:
                self.logger.error(f"Error processing .mat file {file_path} with scipy.io.loadmat: {e_sio}", exc_info=True)
                return None # Both methods failed

        # Common post-processing for data_dict from either h5py or scipy.io
        if not data_dict:
            self.logger.warning(f"No data extracted from .mat file: {file_path} by either method.")
            return None

        lengths = [len(v) for v in data_dict.values() if isinstance(v, np.ndarray) and v.ndim == 1]
        if not lengths:
             self.logger.warning(f"No 1D numeric data arrays found in {file_path} to determine common length.")
             return None
        
        common_length = max(set(lengths), key=lengths.count)
        self.logger.info(f"Determined common length for numeric data in {file_path} as: {common_length}")
        
        final_data_dict = {}
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray) and value.ndim == 1 and len(value) == common_length:
                final_data_dict[key] = value
            else:
                self.logger.warning(f"Column '{key}' in {file_path} (length {len(value) if isinstance(value, np.ndarray) else 'N/A'}) does not match common length {common_length}. Skipping.")

        if not final_data_dict:
            self.logger.warning(f"No columns with consistent length {common_length} found in {file_path} after filtering.")
            return None

        df = pd.DataFrame(final_data_dict)
        
        if 'Timestamp' not in df.columns: # Ensure a time column for resampling
            if 'Time' in df.columns: df.rename(columns={'Time': 'Timestamp'}, inplace=True)
            elif 'time' in df.columns: df.rename(columns={'time': 'Timestamp'}, inplace=True)
        
        self.logger.info(f"Successfully processed and created DataFrame with {len(df.columns)} columns from .mat file: {file_path}")
        return df
        
    def _resample_data(self, df, sampling_frequency='1S'):
        """
        Resamples the DataFrame to a specified frequency based on the Time or Timestamp column.
        """
        try:
            time_col = None
            if 'Timestamp' in df.columns:
                time_col = 'Timestamp'
            elif 'Time' in df.columns: # Fallback for existing MAT file processing
                time_col = 'Time'

            if time_col is None:
                self.logger.error("No 'Time' or 'Timestamp' column found in the dataset for resampling.")
                return None
            
            # Convert time column to datetime objects
            if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                # Attempt to infer datetime format, or assume seconds if it's numeric
                try:
                    df[time_col] = pd.to_datetime(df[time_col])
                except ValueError: # If direct conversion fails, try assuming it's seconds from epoch or similar
                    df[time_col] = pd.to_datetime(df[time_col], unit='s', errors='coerce')

            if df[time_col].isnull().any():
                self.logger.warning(f"Null values found in '{time_col}' column after conversion. Resampling might be affected.")
                df.dropna(subset=[time_col], inplace=True) # Drop rows where time is NaT

            if df.empty:
                self.logger.error(f"DataFrame is empty after handling NaT in '{time_col}'. Cannot resample.")
                return None

            # Set time as index for resampling
            df.set_index(time_col, inplace=True)

            # Resample numeric columns using mean and interpolate
            numeric_cols = df.select_dtypes(include=np.number).columns
            df_resampled_numeric = df[numeric_cols].resample(sampling_frequency).mean().interpolate()

            # Handle non-numeric columns: resample and forward-fill
            non_numeric_cols = df.select_dtypes(exclude=np.number).columns
            if not non_numeric_cols.empty:
                # Ensure the index is available for non_numeric part before resampling
                df_non_numeric_indexed = df[non_numeric_cols]
                if not isinstance(df_non_numeric_indexed.index, pd.DatetimeIndex):
                     # This case should ideally not happen if set_index was successful above
                     # but as a safeguard, re-apply set_index if it got lost.
                     # This part is tricky as the original time_col might not be in df_non_numeric_indexed
                     # For simplicity, we assume the index from numeric resampling can be used.
                     # A more robust way would be to resample non-numeric with .first() or .ffill() directly
                     # if the time index is shared.
                     pass # Assuming index is already set from numeric part

                # Resample non-numeric columns (e.g., using first value in window, then ffill)
                df_resampled_non_numeric = df_non_numeric_indexed.resample(sampling_frequency).first().ffill()
                
                # Combine numeric and non-numeric
                df_resampled = pd.concat([df_resampled_numeric, df_resampled_non_numeric], axis=1)
            else:
                df_resampled = df_resampled_numeric
            
            # Ensure the resampled DataFrame has the same columns as the original numeric/non-numeric split,
            # in case some columns became all NaN and were dropped by resample().mean() or .first()
            # Reindex to ensure all original columns are present, filling with NaN if necessary, then ffill/bfill
            # This is a bit complex; simpler is to ensure resample().first().ffill() for non-numeric
            # and resample().mean().interpolate() for numeric handles most cases.
            # The concat should align on the new DatetimeIndex.

            # Reset index to keep time column
            df_resampled.reset_index(inplace=True)
            
            # Restore original column order if possible (excluding the original index if it was just 'Time')
            original_cols_order = [col for col in df.columns if col in df_resampled.columns] # df here is before set_index
            # df_resampled = df_resampled[original_cols_order] # This might fail if time_col name changed

            return df_resampled
        except Exception as e:
            self.logger.error(f"Error resampling data: {e}", exc_info=True)
            return None
    
    # The duplicate static method _extract_data_from_matfile should be fully removed by the previous replacement.
    # This SEARCH block is to ensure its complete removal if any remnants were left.
    # If the previous diff was perfect, this block might not find anything, which is fine.

    def _update_progress(self, progress_callback):
        """ Update the progress percentage and call the callback. """
        if progress_callback and self.total_files > 0:
            progress_value = int((self.processed_files / self.total_files) * 100)
            self.logger.debug(f"Progress: {progress_value}%")
            progress_callback(progress_value)

