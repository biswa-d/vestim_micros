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

    def organize_and_convert_files(self, train_files, test_files):
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

        # Copy files and process them
        self._copy_files(train_files, train_raw_folder)
        self._copy_files(test_files, test_raw_folder)

        # Process and convert files
        self._convert_files(train_raw_folder, train_processed_folder)
        self._convert_files(test_raw_folder, test_processed_folder)

        # Clear memory explicitly
        del train_files, test_files
        gc.collect()  # Call garbage collector to free memory

        return job_folder

    def _copy_files(self, files, destination_folder):
        """ Copy the files to a destination folder. """
        for file_path in files:
            dest_path = os.path.join(destination_folder, os.path.basename(file_path))
            shutil.copy(file_path, dest_path)
            print(f'Copied {file_path} to {dest_path}')  # Debugging

    def _convert_files(self, input_folder, output_folder):
        """ Convert files from .mat to .csv and delete temporary objects. """
        for root, _, files in os.walk(input_folder):
            for file in tqdm(files, desc="Converting files"):
                if file.endswith('.mat'):
                    file_path = os.path.join(root, file)
                    self._convert_mat_to_csv(file_path, output_folder)

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
