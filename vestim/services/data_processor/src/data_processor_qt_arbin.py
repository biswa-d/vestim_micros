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

        # Increment total file count for .mat files for conversion
        self.total_files += len([f for f in os.listdir(train_raw_folder) if f.endswith('.mat')])
        self.total_files += len([f for f in os.listdir(test_raw_folder) if f.endswith('.mat')])

        self.logger.info(f"Starting file conversion for {self.total_files} .mat files.")

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
        """ Convert files from .mat to .csv and resample to the appropriate frequency and update progress. """
        for root, _, files in os.walk(input_folder):
            total_files = len(files)  # Get the total number of files
            processed_files = 0  # Track processed files
            
            self.logger.info(f"Converting files in folder: {input_folder} to .csv")
            for file in tqdm(files, desc="Converting files"):
                if file.endswith('.mat'):
                    file_path = os.path.join(root, file)
                    self._convert_mat_to_csv(file_path, output_folder)
                    self.logger.info(f"Converted {file_path} to CSV")
                    processed_files += 1
                    self.processed_files += 1  # Update the overall processed files count
                    self._update_progress(progress_callback)

                # Explicitly clear memory after each conversion
                gc.collect()  # Explicit garbage collection after processing each file

    def _convert_and_resample_files(self, input_folder, output_folder, progress_callback=None, sampling_frequency='1S'):
        """ Convert files from .mat to .csv and resample to the appropriate frequency and update progress. """
        for root, _, files in os.walk(input_folder):
            total_files = len(files)  # Get the total number of files
            processed_files = 0  # Track processed files
            
            self.logger.info(f"Converting files in folder: {input_folder} to .csv")
            for file in tqdm(files, desc="Converting files"):
                if file.endswith('.mat'):
                    file_path = os.path.join(root, file)
                    self._convert_mat_to_csv_resampled(file_path, output_folder, sampling_frequency)
                    self.logger.info(f"Converted {file_path} to CSV")
                    processed_files += 1
                    self.processed_files += 1  # Update the overall processed files count
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
        Resamples the DataFrame to a specified frequency based on the Time column.

        Parameters:
        df (pd.DataFrame): Input DataFrame with a 'Time' column.
        target_freq (str): Target frequency (e.g., '1S' for 1Hz, '100ms' for 10Hz).

        Returns:
        pd.DataFrame: Resampled DataFrame.
        """
        try:
            # Ensure 'Time' is present in the dataset
            if 'Time' not in df.columns:
                print("No 'Time' column found in the dataset.")
                return None
            
            # Convert 'Time' to seconds (if not already datetime)
            if not pd.api.types.is_datetime64_any_dtype(df['Time']):
                df['Time'] = pd.to_datetime(df['Time'], unit='s')

            # Set time as index for resampling
            df.set_index('Time', inplace=True)

            # Resample using the specified frequency, interpolating missing values
            df_resampled = df.resample(sampling_frequency).mean(numeric_only=True).interpolate()

            # Reset index to keep 'Time' as a column
            df_resampled.reset_index(inplace=True)

            return df_resampled
        except Exception as e:
            self.logger.error(f"Error resampling data: {e}")
            return None
    
    def _extract_data_from_matfile(file_path):
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
            
    