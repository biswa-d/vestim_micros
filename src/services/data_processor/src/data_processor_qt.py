import os
import shutil
import scipy.io
import numpy as np
import gc  # Explicit garbage collector
from src.gateway.src.job_manager import JobManager
from tqdm import tqdm

class DataProcessor:
    def __init__(self):
        self.job_manager = JobManager()
        self.total_files = 0  # Total number of files to process (copy + convert)
        self.processed_files = 0  # Keep track of total processed files

    def organize_and_convert_files(self, train_files, test_files, progress_callback=None):
        # Ensure `total_files` is calculated upfront and is non-zero
        self.total_files = len(train_files) + len(test_files)

        if self.total_files == 0:
            raise ValueError("No files to process.")

        job_id, job_folder = self.job_manager.create_new_job()

        # Create directories for raw and processed data
        train_raw_folder = os.path.join(job_folder, 'train', 'raw_data')
        train_processed_folder = os.path.join(job_folder, 'train', 'processed_data')
        test_raw_folder = os.path.join(job_folder, 'test', 'raw_data')
        test_processed_folder = os.path.join(job_folder, 'test', 'processed_data')

        os.makedirs(train_raw_folder, exist_ok=True)
        os.makedirs(train_processed_folder, exist_ok=True)
        os.makedirs(test_raw_folder, exist_ok=True)
        os.makedirs(test_processed_folder, exist_ok=True)

        # Reset processed files counter before processing starts
        self.processed_files = 0

        # Process copying and converting files
        self._copy_files(train_files, train_raw_folder, progress_callback)
        self._copy_files(test_files, test_raw_folder, progress_callback)

        # Increment total file count for .mat files for conversion
        self.total_files += len([f for f in os.listdir(train_raw_folder) if f.endswith('.mat')])
        self.total_files += len([f for f in os.listdir(test_raw_folder) if f.endswith('.mat')])

        # Process and convert files
        self._convert_files(train_raw_folder, train_processed_folder, progress_callback)
        self._convert_files(test_raw_folder, test_processed_folder, progress_callback)

        return job_folder

    def _copy_files(self, files, destination_folder, progress_callback=None):
        """ Copy the files to a destination folder and update progress. """
        total_files = len(files)  # Get the total number of files
        processed_files = 0  # Track the number of processed files

        for file_path in files:
            dest_path = os.path.join(destination_folder, os.path.basename(file_path))
            shutil.copy(file_path, dest_path)
            print(f'Copied {file_path} to {dest_path}')  # Debugging

            # Update progress based on the number of files processed
            processed_files += 1
            self.processed_files += 1  # Update the overall processed files count
            self._update_progress(progress_callback)

    def _convert_files(self, input_folder, output_folder, progress_callback=None):
        """ Convert files from .mat to .csv and update progress. """
        for root, _, files in os.walk(input_folder):
            total_files = len(files)  # Get the total number of files
            processed_files = 0  # Track processed files

            for file in tqdm(files, desc="Converting files"):
                if file.endswith('.mat'):
                    file_path = os.path.join(root, file)
                    self._convert_mat_to_csv(file_path, output_folder)
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
        else:
            print(f'Skipping file {mat_file}: "meas" field not found')

        # Explicitly remove temporary variables from memory
        del mat_file
        gc.collect()  # Ensure memory cleanup

    def _update_progress(self, progress_callback):
        """ Update the progress percentage and call the callback. """
        if progress_callback and self.total_files > 0:
            progress_value = int((self.processed_files / self.total_files) * 100)
            progress_callback(progress_value)
