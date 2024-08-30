import os
import shutil
import scipy.io
import numpy as np
from src.gateway.src.job_manager import JobManager

class DataProcessor:
    def __init__(self):
        self.job_manager = JobManager()

    def organize_and_convert_files(self, train_files, test_files):
        job_id, job_folder = self.job_manager.create_new_job()

        train_raw_folder = os.path.join(job_folder, 'train', 'raw_data')
        train_processed_folder = os.path.join(job_folder, 'train', 'processed_data')
        test_raw_folder = os.path.join(job_folder, 'test', 'raw_data')
        test_processed_folder = os.path.join(job_folder, 'test', 'processed_data')

        os.makedirs(train_raw_folder, exist_ok=True)
        os.makedirs(train_processed_folder, exist_ok=True)
        os.makedirs(test_raw_folder, exist_ok=True)
        os.makedirs(test_processed_folder, exist_ok=True)

        self._copy_files(train_files, train_raw_folder)
        self._copy_files(test_files, test_raw_folder)

        self._convert_files(train_raw_folder, train_processed_folder)
        self._convert_files(test_raw_folder, test_processed_folder)

        return job_folder

    def _copy_files(self, files, destination_folder):
        for file_path in files:
            dest_path = os.path.join(destination_folder, os.path.basename(file_path))
            shutil.copy(file_path, dest_path)
            print(f'Copied {file_path} to {dest_path}')  # Debugging

    def _convert_files(self, input_folder, output_folder):
        for root, _, files in os.walk(input_folder):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith('.mat'):
                    self._convert_mat_to_csv(file_path, output_folder)

    def _convert_mat_to_csv(self, mat_file, output_folder):
        data = scipy.io.loadmat(mat_file)
        if 'meas' in data:
            meas = data['meas'][0, 0]
            Timestamp = meas['Time'].flatten()
            Voltage = meas['Voltage'].flatten()
            Current = meas['Current'].flatten()
            Temp = meas['Battery_Temp_degC'].flatten()
            SOC = meas['SOC'].flatten()

            combined_data = np.column_stack((Timestamp, Voltage, Current, Temp, SOC))
            header = ['Timestamp', 'Voltage', 'Current', 'Temp', 'SOC']

            csv_file_name = os.path.join(output_folder, os.path.splitext(os.path.basename(mat_file))[0] + '.csv')
            np.savetxt(csv_file_name, combined_data, delimiter=",", header=",".join(header), comments='', fmt='%s')
            print(f'Data successfully written to {csv_file_name}')
        else:
            print(f'Skipping file {mat_file}: "meas" field not found')
