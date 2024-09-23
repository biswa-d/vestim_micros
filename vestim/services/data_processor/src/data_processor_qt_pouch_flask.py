import os, requests
import shutil
import gc  # Explicit garbage collector
import pandas as pd
import h5py
from vestim.gateway.src.job_manager_qt import JobManager
import logging

class DataProcessorPouch:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.job_folder = None
        self.job_id = None
        self.total_files = 0  # Total number of files to process (copy)
        self.processed_files = 0  # Keep track of total processed files
        # Get required objects from the relevant singleton manaers
    
    def fetch_job_folder(self):
        """Fetches and stores the job folder from the Job Manager API."""
        if self.job_folder is None:
            try:
                response_job = requests.get("http://localhost:5000/job_manager/get_job_folder")
                if response_job.status_code == 200:
                    self.job_folder = response_job.json()['job_folder']
                else:
                    raise Exception("Failed to fetch job folder")
            except Exception as e:
                self.logger.error(f"Error fetching job folder: {str(e)}")
                raise e
    
    def fetch_job_id(self):
        """Fetches and stores the job folder from the Job Manager API."""
        if self.job_id is None:
            try:
                response_id = requests.get("http://localhost:5000/job_manager/get_job_id")
                if response_id.status_code == 200:
                    self.job_id = response_id.json()['job_id']
                else:
                    raise Exception("Failed to fetch job ID")
            except Exception as e:
                self.logger.error(f"Error fetching job id: {str(e)}")
                raise e


    def organize_and_convert_files(self, train_files, test_files, progress_callback=None):
        """Organizes and converts files, with progress updates."""
        
        print("Entering organize_and_convert_files")
        if not all(f.endswith('.csv') for f in train_files + test_files):
            self.logger.error("Invalid file types. Only CSV files are accepted.")
            raise ValueError("Invalid file types. Only CSV files are accepted.")

        self.logger.info("Starting file organization and copy.")
        print(f"number of train files: {len(train_files)}")
        print(f"number of test files: {len(test_files)}")   
        
        self.fetch_job_folder()
        self.fetch_job_id()
        self.logger.info(f"Job created with ID: {self.job_id}, Folder: {self.job_folder}")

        # Switch logger to job-specific log file
        job_log_file = os.path.join(self.job_folder, 'job.log')
        self.switch_log_file(job_log_file)

        # Create directories for raw and processed data
        train_raw_folder = os.path.join(self.job_folder, 'train', 'raw_data')
        train_processed_folder = os.path.join(self.job_folder, 'train', 'processed_data')
        test_raw_folder = os.path.join(self.job_folder, 'test', 'raw_data')
        test_processed_folder = os.path.join(self.job_folder, 'test', 'processed_data')

        # Clear the processed data folders before proceeding
        if os.path.exists(train_processed_folder):
            shutil.rmtree(train_processed_folder)
        if os.path.exists(test_processed_folder):
            shutil.rmtree(test_processed_folder)
        
        os.makedirs(train_raw_folder, exist_ok=True)
        os.makedirs(train_processed_folder, exist_ok=True)
        os.makedirs(test_raw_folder, exist_ok=True)
        os.makedirs(test_processed_folder, exist_ok=True)
        self.logger.info(f"Created folders: {train_raw_folder}, {train_processed_folder}, {test_raw_folder}, {test_processed_folder}")
        
        # Reset processed files counter
        self.processed_files = 0
        self.total_files = len(train_files) + len(test_files)  # Dynamically set based on actual file count

        # Copy files to the raw data directories
        for file in train_files:
            self._copy_file(file, train_raw_folder, progress_callback)
        for file in test_files:
            self._copy_file(file, test_raw_folder, progress_callback)

        # Convert the copied CSV files to HDF5 and store in the processed data folder
        for file in train_files:
            self._convert_to_hdf5(file, train_processed_folder, progress_callback)
        for file in test_files:
            self._convert_to_hdf5(file, test_processed_folder, progress_callback)


    def _convert_to_hdf5(self, csv_file, output_folder, progress_callback=None):
        """Convert a CSV file to HDF5 and save in the processed folder."""
        df = pd.read_csv(csv_file)

        # Extract filename without extension
        filename = os.path.splitext(os.path.basename(csv_file))[0]
        
        # Path to save the HDF5 file
        hdf5_file = os.path.join(output_folder, f'{filename}.h5')

        # Save the DataFrame into HDF5 format
        with h5py.File(hdf5_file, 'w') as hdf:
            hdf.create_dataset('SOC', data=df['SOC'].values)
            hdf.create_dataset('Voltage', data=df['Voltage'].values)
            hdf.create_dataset('Current', data=df['Current'].values)
            hdf.create_dataset('Temp', data=df['Temp'].values)

        self.logger.info(f"Converted {csv_file} to HDF5 format at {hdf5_file}")
        self.processed_files += 1
        
        # Call the progress callback
        if progress_callback:
            self._update_progress(progress_callback)

    def _copy_file(self, file, destination_folder, progress_callback=None):
        """Copies a file and updates progress."""
        dest_path = os.path.join(destination_folder, os.path.basename(file))
        shutil.copy(file, dest_path)
        self.logger.info(f"Copied {file} to {dest_path}")
        self.processed_files += 1
        
        # Call the progress callback
        if progress_callback:
            self._update_progress(progress_callback)

    def switch_log_file(self, job_log_file):
        """Switch logger to a job-specific log file by removing the previous handlers."""
        
        # Get the root logger or the main logger used across the scripts
        logger = logging.getLogger()

        # Remove all existing handlers (this includes the default log handler)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Now add the new job-specific log handler
        job_file_handler = logging.FileHandler(job_log_file)
        job_file_handler.setLevel(logging.DEBUG)
        job_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(job_file_handler)
        
        # Optionally, also log to the console (for debugging or real-time info)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)

        # Log a message to confirm the switch
        logger.info(f"Switched logging to {job_log_file}")


    def _update_progress(self, progress_callback):
        """ Update progress based on the number of files processed. """
        if progress_callback and self.total_files > 0:
            progress_value = int((self.processed_files / self.total_files) * 100)
            self.logger.debug(f"Progress: {progress_value}%")
            progress_callback(progress_value)