import os
import shutil
import scipy.io
import numpy as np
import gc  # Explicit garbage collector
from vestim.gateway.src.job_manager_qt import JobManager
from tqdm import tqdm

import logging
from vestim.logger_config import setup_logger
class DataProcessorTesla:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.job_manager = JobManager()
        self.total_files = 0  # Total number of files to process (copy + convert)
        self.processed_files = 0  # Keep track of total processed files

    def organize_and_convert_files(self, train_files, test_files, progress_callback=None):
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
        # Process and convert files
        self._convert_files(train_raw_folder, train_processed_folder, progress_callback)
        self._convert_files(test_raw_folder, test_processed_folder, progress_callback)

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

    def _convert_files(self, input_folder, output_folder, progress_callback=None):
        """ Convert files from .mat to .csv and update progress. """
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

    def _convert_mat_to_csv(self, mat_file, output_folder):
        """ Convert .mat file to .csv and delete large arrays after processing. """
        data = scipy.io.loadmat(mat_file)
        if 'meas' in data:
            meas = data['meas'][0, 0]
            Timestamp = meas['Time'].flatten()
            Voltage = meas['Voltage'].flatten()
            Current = meas['Current'].flatten()
            Temp = meas['Battery_Temp_degC'].flatten()
            SOC = meas['SOC'].flatten()

            # Combine data and write to CSV
            combined_data = np.column_stack((Timestamp, Voltage, Current, Temp, SOC))
            header = ['Timestamp', 'Voltage', 'Current', 'Temp', 'SOC']
            csv_file_name = os.path.join(output_folder, os.path.splitext(os.path.basename(mat_file))[0] + '.csv')

            # Save the CSV
            np.savetxt(csv_file_name, combined_data, delimiter=",", header=",".join(header), comments='', fmt='%s')
            print(f'Data successfully written to {csv_file_name}')

            # Delete large arrays to free memory
            del data, Timestamp, Voltage, Current, Temp, SOC, combined_data
            gc.collect()  # Force garbage collection
            self.logger.info(f"Converted {mat_file} to CSV successfully.")
        else:
            self.logger.warning(f"Skipping file {mat_file}: 'meas' field not found")

        # Explicitly remove temporary variables from memory
        del mat_file
        gc.collect()  # Ensure memory cleanup

    def _update_progress(self, progress_callback):
        """ Update the progress percentage and call the callback. """
        if progress_callback and self.total_files > 0:
            progress_value = int((self.processed_files / self.total_files) * 100)
            self.logger.debug(f"Progress: {progress_value}%")
            progress_callback(progress_value)
