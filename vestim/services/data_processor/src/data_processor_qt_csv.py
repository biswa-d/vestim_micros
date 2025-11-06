import os
import shutil
import gc  # Explicit garbage collector
import pandas as pd
import numpy as np
import h5py
from vestim.gateway.src.job_manager_qt import JobManager
import logging

class DataProcessorCSV:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.total_files = 0  # Total number of files to process (copy)
        self.processed_files = 0  # Keep track of total processed files

    def organize_and_convert_files(self, train_files, val_files, test_files, progress_callback=None, sampling_frequency=None, job_manager=None):
        # Ensure valid CSV files are provided
        if not all(f.endswith('.csv') for f in train_files + val_files + test_files):
            self.logger.error("Invalid file types. Only CSV files are accepted for CSV processor.")
            raise ValueError("Invalid file types. Only CSV files are accepted for CSV processor.")

        self.logger.info("Starting file organization and processing for CSV files.")
        
        if job_manager is None:
            job_manager = JobManager()

        job_id, job_folder = job_manager.create_new_job()
        self.logger.info(f"Job created with ID: {job_id}, Folder: {job_folder}")

        job_log_file = os.path.join(job_folder, 'job.log')
        self.switch_log_file(job_log_file)

        train_raw_folder = os.path.join(job_folder, 'train_data', 'raw_data')
        train_processed_folder = os.path.join(job_folder, 'train_data', 'processed_data')
        val_raw_folder = os.path.join(job_folder, 'val_data', 'raw_data')
        val_processed_folder = os.path.join(job_folder, 'val_data', 'processed_data')
        test_raw_folder = os.path.join(job_folder, 'test_data', 'raw_data')
        test_processed_folder = os.path.join(job_folder, 'test_data', 'processed_data')

        for folder in [train_raw_folder, train_processed_folder, val_raw_folder, val_processed_folder, test_raw_folder, test_processed_folder]:
            if os.path.exists(folder): # Clear if exists, then create
                shutil.rmtree(folder)
            os.makedirs(folder, exist_ok=True)
        self.logger.info(f"Created/Re-created data folders.")
        
        self.processed_files = 0
        self.total_files = len(train_files) + len(val_files) + len(test_files)
        if self.total_files == 0:
            self.logger.warning("No files provided for CSV processing.")
            return job_folder # Return early if no files

        # 1. Copy original CSVs to raw_data folders
        self.logger.info("Copying original CSVs to raw_data directories...")
        for original_file_path in train_files:
            dest_path = os.path.join(train_raw_folder, os.path.basename(original_file_path))
            shutil.copy(original_file_path, dest_path)
            # self.logger.info(f"Copied {original_file_path} to {dest_path}") # Too verbose
        
        for original_file_path in val_files:
            dest_path = os.path.join(val_raw_folder, os.path.basename(original_file_path))
            shutil.copy(original_file_path, dest_path)
            # self.logger.info(f"Copied {original_file_path} to {dest_path}") # Too verbose
        
        for original_file_path in test_files:
            dest_path = os.path.join(test_raw_folder, os.path.basename(original_file_path))
            shutil.copy(original_file_path, dest_path)
            # self.logger.info(f"Copied {original_file_path} to {dest_path}") # Too verbose

        # 2. Process CSVs from raw_data to processed_data
        self.logger.info("Processing CSV files from raw_data to processed_data...")

        # Process training files from raw_data
        for filename in os.listdir(train_raw_folder):
            if filename.endswith('.csv'):
                input_csv_path = os.path.join(train_raw_folder, filename)
                output_csv_path = os.path.join(train_processed_folder, filename)
                try:
                    self._process_standard_csv(input_csv_path, output_csv_path, sampling_frequency)
                    self.processed_files += 1
                    self._update_progress(progress_callback)
                except Exception as e:
                    self.logger.error(f"Failed to process CSV train file {input_csv_path}: {e}")
                    # Optionally, re-raise or handle as per error policy

        # Process validation files from raw_data
        for filename in os.listdir(val_raw_folder):
            if filename.endswith('.csv'):
                input_csv_path = os.path.join(val_raw_folder, filename)
                output_csv_path = os.path.join(val_processed_folder, filename)
                try:
                    self._process_standard_csv(input_csv_path, output_csv_path, sampling_frequency)
                    self.processed_files += 1
                    self._update_progress(progress_callback)
                except Exception as e:
                    self.logger.error(f"Failed to process CSV validation file {input_csv_path}: {e}")
                    # Optionally, re-raise or handle as per error policy

        # Process testing files from raw_data
        for filename in os.listdir(test_raw_folder):
            if filename.endswith('.csv'):
                input_csv_path = os.path.join(test_raw_folder, filename)
                output_csv_path = os.path.join(test_processed_folder, filename)
                try:
                    self._process_standard_csv(input_csv_path, output_csv_path, sampling_frequency)
                    self.processed_files += 1
                    self._update_progress(progress_callback)
                except Exception as e:
                    self.logger.error(f"Failed to process CSV test file {input_csv_path}: {e}")
                    # Optionally, re-raise or handle

        self.logger.info("CSV processing complete.")
        return job_folder

    def _process_csv_with_custom_header_skip(self, input_csv_path, output_csv_path, sampling_frequency=None):
        self.logger.info(f"Processing CSV with custom header skip: {input_csv_path}")
        try:
            # Headers at Excel row 30 (0-indexed line 29)
            # Data starts at Excel row 32 (0-indexed line 31)
            
            # Read headers from line 29
            df_headers = pd.read_csv(input_csv_path, skiprows=29, nrows=1, header=None, encoding='utf-8', on_bad_lines='skip')
            if df_headers.empty:
                self.logger.error(f"Could not read headers from {input_csv_path} at line 30 (0-indexed 29).")
                open(output_csv_path, 'w').close() # Create empty file
                return

            column_names = [str(name).strip() for name in df_headers.iloc[0].tolist()]

            # Read data from line 31 (skip first 31 lines: 0 to 30)
            df = pd.read_csv(input_csv_path, skiprows=31, header=None, names=column_names, encoding='utf-8', on_bad_lines='warn')

            if df.empty:
                self.logger.warning(f"No data found in {input_csv_path} after skipping initial rows. Saving header-only CSV.")
                pd.DataFrame(columns=column_names).to_csv(output_csv_path, index=False)
                return

            # Attempt to convert all columns to numeric, coercing errors
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df.dropna(how='all', inplace=True) # Drop rows where all values became NaN

            if df.empty:
                self.logger.warning(f"Data became empty after numeric conversion and NaN drop for {input_csv_path}. Saving header-only CSV.")
                pd.DataFrame(columns=column_names).to_csv(output_csv_path, index=False)
                return

            if sampling_frequency:
                self.logger.info(f"Resampling data from {input_csv_path} with frequency {sampling_frequency}")
                df_resampled = self._resample_data(df.copy(), sampling_frequency)
                if df_resampled is not None and not df_resampled.empty:
                    df = df_resampled
                else:
                    self.logger.warning(f"Resampling resulted in empty or None data for {input_csv_path}. Using original (cleaned) data.")
            
            df.to_csv(output_csv_path, index=False)
            self.logger.info(f"Successfully processed CSV and saved to {output_csv_path}")

        except pd.errors.EmptyDataError:
            self.logger.error(f"Pandas EmptyDataError: Likely header not found as expected or file is empty before data rows in {input_csv_path}. Skipping.")
            open(output_csv_path, 'w').close()
        except Exception as e:
            self.logger.error(f"Error processing CSV file {input_csv_path}: {e}")
            open(output_csv_path, 'w').close()
        finally:
            gc.collect()

    def _process_standard_csv(self, input_csv_path, output_csv_path, sampling_frequency=None):
        # self.logger.info(f"Processing standard CSV: {input_csv_path}") # Too verbose
        try:
            # Headers at row 1 (0-indexed 0), data starts at row 2 (0-indexed 1)
            df = pd.read_csv(input_csv_path, header=0, encoding='utf-8', on_bad_lines='warn')

            if df.empty:
                self.logger.warning(f"No data found in {input_csv_path} (or only headers). Saving empty/header-only CSV.")
                df.to_csv(output_csv_path, index=False) # Save headers if present, or empty if not
                return

            # Attempt to convert all columns to numeric, coercing errors
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df.dropna(how='all', inplace=True) # Drop rows where all values became NaN

            if df.empty:
                self.logger.warning(f"Data became empty after numeric conversion and NaN drop for {input_csv_path}. Saving header-only CSV.")
                # Create a DataFrame with original columns if df is empty but had columns
                original_columns = pd.read_csv(input_csv_path, nrows=0, encoding='utf-8').columns
                pd.DataFrame(columns=original_columns).to_csv(output_csv_path, index=False)
                return

            if sampling_frequency:
                self.logger.info(f"Resampling data from {input_csv_path} with frequency {sampling_frequency}")
                df_resampled = self._resample_data(df.copy(), sampling_frequency)
                if df_resampled is not None and not df_resampled.empty:
                    df = df_resampled
                else:
                    self.logger.warning(f"Resampling resulted in empty or None data for {input_csv_path}. Using original (cleaned) data.")
            
            df.to_csv(output_csv_path, index=False)
            # self.logger.info(f"Successfully processed standard CSV and saved to {output_csv_path}") # Too verbose

        except pd.errors.EmptyDataError:
            self.logger.error(f"Pandas EmptyDataError: File {input_csv_path} is empty or contains no data rows. Skipping.")
            open(output_csv_path, 'w').close() # Create empty file
        except Exception as e:
            self.logger.error(f"Error processing standard CSV file {input_csv_path}: {e}")
            open(output_csv_path, 'w').close() # Create empty file
        finally:
            gc.collect()

    # Commented out _convert_to_hdf5 as it's not used in the new CSV processing flow
    # def _convert_to_hdf5(self, csv_file, output_folder, progress_callback=None):
    #     # ... (original HDF5 code) ...

    def switch_log_file(self, job_log_file):
        """Switch logger to a job-specific log file by removing the previous handlers."""
        logger = logging.getLogger() # Get root logger
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close() # Close handler before removing

        job_file_handler = logging.FileHandler(job_log_file, mode='a') # Append mode
        job_file_handler.setLevel(logging.DEBUG)
        job_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(job_file_handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)
        logger.info(f"Switched logging to {job_log_file}")

    # _copy_file is not directly used by organize_and_convert_files for CSV processor anymore,
    # but kept for potential other uses or if other processors need it.
    def _copy_file(self, file_path, destination_folder, progress_callback=None):
        """ Copy a single file to the destination folder and update progress. """
        # This method is not directly called by the new CSV processor flow for its main processing loop,
        # as copying to raw is handled directly, and then processing from raw to processed occurs.
        # However, it's kept here as it might be a utility for other things or was previously used.
        # If it were to be used for progress, ensure self.total_files reflects that.
        dest_path = os.path.join(destination_folder, os.path.basename(file_path))
        shutil.copy(file_path, dest_path)
        self.logger.info(f"Copied {file_path} to {dest_path}")
        # self.processed_files += 1 # This would be part of a different progress logic
        # self._update_progress(progress_callback)


    def _update_progress(self, progress_callback):
        """ Update progress based on the number of files processed. """
        if progress_callback and self.total_files > 0:
            # Ensure progress_value does not exceed 100
            progress_value = min(int((self.processed_files / self.total_files) * 100), 100)
            self.logger.debug(f"Progress: {self.processed_files}/{self.total_files} -> {progress_value}%")
            progress_callback(progress_value)
        
    def _resample_data(self, df, sampling_frequency='1S'):
        """
        Resamples the DataFrame to a specified frequency based on the Time column.

        Parameters:
        df (pd.DataFrame): Input DataFrame with a 'Time' column.
        target_freq (str): Target frequency (e.g., '1S' for 1Hz, '100ms' for 10Hz).

        Returns:
        pd.DataFrame: Resampled DataFrame.
        """
        try:
            time_col = None
            if 'Timestamp' in df.columns: # Prefer 'Timestamp'
                time_col = 'Timestamp'
            elif 'Time' in df.columns:
                time_col = 'Time'
            
            if time_col is None:
                self.logger.error("No 'Time' or 'Timestamp' column found in the dataset for resampling.")
                return None # Or return df as is, if no time column means no resampling
            
            # Convert time column to datetime objects
            if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                try:
                    df[time_col] = pd.to_datetime(df[time_col])
                except ValueError:
                    try: # Attempt to parse as seconds from epoch if direct conversion fails
                        df[time_col] = pd.to_datetime(df[time_col], unit='s', errors='coerce')
                    except Exception as e_time_conv:
                         self.logger.error(f"Could not convert time column '{time_col}' to datetime: {e_time_conv}. Resampling skipped.")
                         return df # Return original df if time conversion fails

            if df[time_col].isnull().any():
                self.logger.warning(f"Null values found in '{time_col}' column after conversion. Dropping these rows before resampling.")
                df.dropna(subset=[time_col], inplace=True)

            if df.empty:
                self.logger.error(f"DataFrame is empty after handling NaT in '{time_col}'. Cannot resample.")
                return None

            df.set_index(time_col, inplace=True)

            # Resample numeric columns using mean and interpolate
            numeric_cols = df.select_dtypes(include=np.number).columns
            df_resampled_numeric = df[numeric_cols].resample(sampling_frequency).mean().interpolate()

            # Handle non-numeric columns: resample and forward-fill
            non_numeric_cols = df.select_dtypes(exclude=np.number).columns
            if not non_numeric_cols.empty:
                df_non_numeric_indexed = df[non_numeric_cols]
                # Resample non-numeric columns (e.g., using first value in window, then ffill)
                df_resampled_non_numeric = df_non_numeric_indexed.resample(sampling_frequency).first().ffill()
                df_resampled = pd.concat([df_resampled_numeric, df_resampled_non_numeric], axis=1)
            else:
                df_resampled = df_resampled_numeric
            
            df_resampled.reset_index(inplace=True)
            return df_resampled
        except Exception as e:
            self.logger.error(f"Error resampling data: {e}", exc_info=True)
            return None
