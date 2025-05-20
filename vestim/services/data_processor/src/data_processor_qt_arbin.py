# ---------------------------------------------------------------------------------
# Author: Biswanath Dehury
# Date: `{{date:2025-03-02}}`
# Version: 1.0.0
# Description: This is exactly same as DataProcessorSTLA now and changes in the future may be brought for this according to requirements.
# ---------------------------------------------------------------------------------


import os
import shutil
import scipy.io as sio
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
            
            # Define potential column names and their standard mappings
            column_mapping = {
                'Time': 'Timestamp', 'timestamp': 'Timestamp', 'Record Time': 'Timestamp',
                'Voltage': 'Voltage', 'voltage': 'Voltage', 'Voltage(V)': 'Voltage',
                'Current': 'Current', 'current': 'Current', 'Current(A)': 'Current',
                'Temperature': 'Temp', 'temperature': 'Temp', 'Battery_Temp_degC': 'Temp', 'Aux_Temperature_1(C)': 'Temp',
                'SOC': 'SOC', 'soc': 'SOC', 'SOC(%)': 'SOC'
            }
            
            # Rename columns based on the mapping
            df.rename(columns=column_mapping, inplace=True)
            
            # Select only the standard columns, if they exist
            standard_columns = ['Timestamp', 'Voltage', 'Current', 'Temp', 'SOC']
            df_processed = df[[col for col in standard_columns if col in df.columns]]

            # Ensure all standard columns are present, fill with NaN if not
            for col in standard_columns:
                if col not in df_processed.columns:
                    df_processed[col] = np.nan
            
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

            column_mapping = {
                'Time': 'Timestamp', 'timestamp': 'Timestamp', 'Record Time': 'Timestamp',
                'Voltage': 'Voltage', 'voltage': 'Voltage', 'Voltage(V)': 'Voltage',
                'Current': 'Current', 'current': 'Current', 'Current(A)': 'Current',
                'Temperature': 'Temp', 'temperature': 'Temp', 'Battery_Temp_degC': 'Temp', 'Aux_Temperature_1(C)': 'Temp',
                'SOC': 'SOC', 'soc': 'SOC', 'SOC(%)': 'SOC'
            }
            
            df.rename(columns=column_mapping, inplace=True)
            
            standard_columns = ['Timestamp', 'Voltage', 'Current', 'Temp', 'SOC']
            df_processed = df[[col for col in standard_columns if col in df.columns]]

            for col in standard_columns:
                if col not in df_processed.columns:
                    df_processed[col] = np.nan

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

            column_mapping = {
                'Time': 'Timestamp', 'timestamp': 'Timestamp', 'Record Time': 'Timestamp',
                'Voltage': 'Voltage', 'voltage': 'Voltage', 'Voltage(V)': 'Voltage',
                'Current': 'Current', 'current': 'Current', 'Current(A)': 'Current',
                'Temperature': 'Temp', 'temperature': 'Temp', 'Battery_Temp_degC': 'Temp', 'Aux_Temperature_1(C)': 'Temp',
                'SOC': 'SOC', 'soc': 'SOC', 'SOC(%)': 'SOC'
            }
            df.rename(columns=column_mapping, inplace=True)

            # Ensure 'Timestamp' column exists for resampling
            if 'Timestamp' not in df.columns:
                self.logger.error(f"'Timestamp' column not found in {csv_file_path} after mapping. Skipping resampling.")
                # Save as is, or handle error differently
                self._convert_csv_to_csv(csv_file_path, output_folder) # Fallback to non-resampled conversion
                return

            # Select standard columns
            standard_columns = ['Timestamp', 'Voltage', 'Current', 'Temp', 'SOC']
            df_processed = df[[col for col in standard_columns if col in df.columns]]
            for col in standard_columns: # Ensure all standard columns exist
                if col not in df_processed.columns:
                    df_processed[col] = np.nan
            
            # Rename 'Timestamp' to 'Time' for _resample_data compatibility if needed, or adapt _resample_data
            # For now, let's assume _resample_data can handle 'Timestamp' or we adapt it.
            # If _resample_data strictly expects 'Time', uncomment below:
            # df_processed.rename(columns={'Timestamp': 'Time'}, inplace=True)

            df_resampled = self._resample_data(df_processed.copy(), sampling_frequency) # Pass a copy to avoid SettingWithCopyWarning
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

            column_mapping = {
                'Time': 'Timestamp', 'timestamp': 'Timestamp', 'Record Time': 'Timestamp',
                'Voltage': 'Voltage', 'voltage': 'Voltage', 'Voltage(V)': 'Voltage',
                'Current': 'Current', 'current': 'Current', 'Current(A)': 'Current',
                'Temperature': 'Temp', 'temperature': 'Temp', 'Battery_Temp_degC': 'Temp', 'Aux_Temperature_1(C)': 'Temp',
                'SOC': 'SOC', 'soc': 'SOC', 'SOC(%)': 'SOC'
            }
            df.rename(columns=column_mapping, inplace=True)

            if 'Timestamp' not in df.columns:
                self.logger.error(f"'Timestamp' column not found in {excel_file_path} after mapping. Skipping resampling.")
                self._convert_excel_to_csv(excel_file_path, output_folder) # Fallback
                return

            standard_columns = ['Timestamp', 'Voltage', 'Current', 'Temp', 'SOC']
            df_processed = df[[col for col in standard_columns if col in df.columns]]
            for col in standard_columns:
                if col not in df_processed.columns:
                    df_processed[col] = np.nan
            
            # df_processed.rename(columns={'Timestamp': 'Time'}, inplace=True) # If _resample_data needs 'Time'

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
        Extracts specific fields from a .mat file and returns them as a DataFrame.
        
        Parameters:
        file_path (str): Path to the .mat file.

        Returns:
        pd.DataFrame: DataFrame with the extracted data or None if extraction fails.
        """
        try:
            # Load the .mat file
            mat_data = sio.loadmat(file_path)
            meas = mat_data.get('meas')
            
            if meas is None:
                print(f"No 'meas' structure found in {file_path}")
                return None

            # Define fields to extract, excluding non-numeric or unnecessary fields
            fields_to_extract = [
                'Time', 'Voltage', 'Current', 'Ah', 'SOC', 'Power',
                'Battery_Temp_degC', 'Ambient_Temp_degC'
            ]
            
            # Extract data into a dictionary (assuming data is stored in structures)
            data_dict = {}
            for field in fields_to_extract:
                if field in meas.dtype.names:
                    data_dict[field] = meas[field][0, 0].flatten()

            # Convert dictionary to a DataFrame
            df = pd.DataFrame(data_dict)
            df.rename(columns={'Battery_Temp_degC': 'Temp'}, inplace=True)

            return df

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return None
        
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

            # Resample using the specified frequency, interpolating missing values
            # Ensure only numeric columns are aggregated
            numeric_cols = df.select_dtypes(include=np.number).columns
            df_resampled = df[numeric_cols].resample(sampling_frequency).mean().interpolate()


            # Reset index to keep time column
            df_resampled.reset_index(inplace=True)

            return df_resampled
        except Exception as e:
            self.logger.error(f"Error resampling data: {e}")
            return None
    
    def _extract_data_from_matfile(file_path): # This seems to be a duplicate static method definition. Should be removed or be self.extract_data_from_matfile
        """
        Extracts specific fields from a .mat file and returns them as a DataFrame.
        
        Parameters:
        file_path (str): Path to the .mat file.

        Returns:
        pd.DataFrame: DataFrame with the extracted data or None if extraction fails.
        """
        try:
            # Load the .mat file
            mat_data = sio.loadmat(file_path)
            meas = mat_data.get('meas')
            
            if meas is None:
                print(f"No 'meas' structure found in {file_path}")
                return None

            # Define fields to extract, excluding non-numeric or unnecessary fields
            fields_to_extract = [
                'Time', 'Voltage', 'Current', 'Ah', 'SOC', 'Power',
                'Battery_Temp_degC', 'Ambient_Temp_degC', 'TimeStamp'
            ]
            
            # Extract data into a dictionary (assuming data is stored in structures)
            data_dict = {}
            for field in fields_to_extract:
                if field in meas.dtype.names:
                    data_dict[field] = meas[field][0, 0].flatten()

            # Convert dictionary to a DataFrame
            df = pd.DataFrame(data_dict)
            df.rename(columns={'Battery_Temp_degC': 'Temp'}, inplace=True)

            # Ensure 'Time' is present in the dataset
            if 'Time' not in df.columns:
                print("No 'Time' column found in the dataset.")
                return None

            return df

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return None
    
    def _update_progress(self, progress_callback):
        """ Update the progress percentage and call the callback. """
        if progress_callback and self.total_files > 0:
            progress_value = int((self.processed_files / self.total_files) * 100)
            self.logger.debug(f"Progress: {progress_value}%")
            progress_callback(progress_value)

