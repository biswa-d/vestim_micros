import os
import shutil
# import scipy.io as sio # Not needed for STLA if it handles Excel/CSV
import numpy as np
import gc  # Explicit garbage collector
from vestim.gateway.src.job_manager_qt import JobManager
from tqdm import tqdm
import pandas as pd

import logging

class DataProcessorSTLA:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.job_manager = JobManager()
        self.total_files = 0  # Total number of files to process (copy + convert)
        self.processed_files = 0  # Keep track of total processed files

    def organize_and_convert_files(self, train_files, val_files, test_files, progress_callback=None, sampling_frequency=None):
        # Ensure `total_files` is calculated upfront and is non-zero
        self.logger.info("Starting file organization and conversion.")
        self.total_files = len(train_files) + len(val_files) + len(test_files)

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
        val_raw_folder = os.path.join(job_folder, 'val_data', 'raw_data')
        val_processed_folder = os.path.join(job_folder, 'val_data', 'processed_data')
        test_raw_folder = os.path.join(job_folder, 'test_data', 'raw_data')
        test_processed_folder = os.path.join(job_folder, 'test_data', 'processed_data')

        os.makedirs(train_raw_folder, exist_ok=True)
        os.makedirs(train_processed_folder, exist_ok=True)
        os.makedirs(val_raw_folder, exist_ok=True)
        os.makedirs(val_processed_folder, exist_ok=True)
        os.makedirs(test_raw_folder, exist_ok=True)
        os.makedirs(test_processed_folder, exist_ok=True)
        self.logger.info(f"Created folders: {train_raw_folder}, {train_processed_folder}, {val_raw_folder}, {val_processed_folder}, {test_raw_folder}, {test_processed_folder}")

        # Reset processed files counter before processing starts
        self.processed_files = 0

        # Process copying and converting files
        self._copy_files(train_files, train_raw_folder, progress_callback)
        self._copy_files(val_files, val_raw_folder, progress_callback)
        self._copy_files(test_files, test_raw_folder, progress_callback)

        # Increment total file count for relevant STLA files (Excel, CSV) for conversion
        stla_extensions = ('.xlsx', '.xls', '.csv')
        self.total_files += len([f for f in os.listdir(train_raw_folder) if f.lower().endswith(stla_extensions)])
        self.total_files += len([f for f in os.listdir(val_raw_folder) if f.lower().endswith(stla_extensions)])
        self.total_files += len([f for f in os.listdir(test_raw_folder) if f.lower().endswith(stla_extensions)])

        self.logger.info(f"Starting file conversion for relevant Excel/CSV files.")

        # **Check if resampling is needed**
        if sampling_frequency is None:
            self.logger.info("No resampling selected. Performing standard conversion.")
            self._convert_files(train_raw_folder, train_processed_folder, progress_callback)
            self._convert_files(val_raw_folder, val_processed_folder, progress_callback)
            self._convert_files(test_raw_folder, test_processed_folder, progress_callback)
        else:
            self.logger.info(f"Resampling enabled. Resampling frequency: {sampling_frequency} Hz")
            self._convert_and_resample_files(train_raw_folder, train_processed_folder, progress_callback, sampling_frequency)
            self._convert_and_resample_files(val_raw_folder, val_processed_folder, progress_callback, sampling_frequency)
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
                file_path = os.path.join(root, file)
                file_lower = file.lower()
                converted = False
                if file_lower.endswith(('.xlsx', '.xls')):
                    self._convert_excel_to_csv(file_path, output_folder)
                    self.logger.info(f"Converted Excel {file_path} to CSV")
                    converted = True
                elif file_lower.endswith('.csv'):
                    self._convert_csv_to_csv(file_path, output_folder) # Assuming direct copy or minimal processing
                    self.logger.info(f"Processed CSV {file_path}")
                    converted = True
                
                if converted:
                    processed_files += 1
                    self.processed_files += 1
                    self._update_progress(progress_callback)

                gc.collect()

    def _convert_and_resample_files(self, input_folder, output_folder, progress_callback=None, sampling_frequency='1S'):
        """ Convert Excel/CSV files to resampled CSV and update progress. """
        for root, _, files in os.walk(input_folder):
            self.logger.info(f"Converting and resampling files in folder: {input_folder} to .csv")
            for file in tqdm(files, desc="Converting and resampling files"):
                file_path = os.path.join(root, file)
                file_lower = file.lower()
                converted_resampled = False
                if file_lower.endswith(('.xlsx', '.xls')):
                    self._convert_excel_to_csv_resampled(file_path, output_folder, sampling_frequency)
                    self.logger.info(f"Converted and resampled Excel {file_path} to CSV")
                    converted_resampled = True
                elif file_lower.endswith('.csv'):
                    self._convert_csv_to_csv_resampled(file_path, output_folder, sampling_frequency)
                    self.logger.info(f"Converted and resampled CSV {file_path}")
                    converted_resampled = True

                if converted_resampled:
                    processed_files += 1 # This variable seems unused here, self.processed_files is global
                    self.processed_files += 1
                    self._update_progress(progress_callback)
                
                gc.collect()

    # --- Add CSV and Excel conversion methods (should be similar to Arbin's, without column filtering) ---
    def _convert_csv_to_csv(self, csv_file_path, output_folder):
        """Processes a CSV file, saving it to the output folder."""
        try:
            df = pd.read_csv(csv_file_path)
            # No column filtering or renaming, save as is for STLA.
            df_processed = df
            
            csv_file_name = os.path.join(output_folder, os.path.basename(csv_file_path))
            df_processed.to_csv(csv_file_name, index=False)
            self.logger.info(f"Successfully processed STLA CSV {csv_file_path} to {csv_file_name}")
        except Exception as e:
            self.logger.error(f"Error processing STLA CSV file {csv_file_path}: {e}")
        finally:
            if 'df' in locals(): del df
            gc.collect()

    def _convert_excel_to_csv(self, excel_file_path, output_folder):
        """Converts an Excel file to CSV for STLA."""
        try:
            df = pd.read_excel(excel_file_path, sheet_name=0)
            # No column filtering or renaming, save as is for STLA.
            df_processed = df

            csv_file_name = os.path.join(output_folder, os.path.splitext(os.path.basename(excel_file_path))[0] + '.csv')
            df_processed.to_csv(csv_file_name, index=False)
            self.logger.info(f"Successfully converted STLA Excel {excel_file_path} to {csv_file_name}")
        except Exception as e:
            self.logger.error(f"Error converting STLA Excel file {excel_file_path}: {e}")
        finally:
            if 'df' in locals(): del df
            gc.collect()

    def _convert_csv_to_csv_resampled(self, csv_file_path, output_folder, sampling_frequency='1S'):
        """Converts and resamples a CSV file for STLA."""
        try:
            df = pd.read_csv(csv_file_path)
            if 'Timestamp' not in df.columns and 'Time' in df.columns:
                df.rename(columns={'Time': 'Timestamp'}, inplace=True)
            elif 'timestamp' in df.columns:
                 df.rename(columns={'timestamp': 'Timestamp'}, inplace=True)

            if 'Timestamp' not in df.columns:
                self.logger.error(f"A recognizable time column for resampling not found in STLA CSV {csv_file_path}. Saving as is.")
                self._convert_csv_to_csv(csv_file_path, output_folder)
                return
            
            df_resampled = self._resample_data(df.copy(), sampling_frequency)
            if df_resampled is None or df_resampled.empty:
                self.logger.warning(f"Resampling failed or resulted in empty data for STLA CSV {csv_file_path}. Saving original processed version.")
                self._convert_csv_to_csv(csv_file_path, output_folder) # Fallback
                return

            csv_file_name = os.path.join(output_folder, os.path.basename(csv_file_path))
            df_resampled.to_csv(csv_file_name, index=False)
            self.logger.info(f"Successfully converted and resampled STLA CSV {csv_file_path} to {csv_file_name}")
        except Exception as e:
            self.logger.error(f"Error converting/resampling STLA CSV {csv_file_path}: {e}")
        finally:
            if 'df' in locals(): del df
            if 'df_resampled' in locals(): del df_resampled
            gc.collect()

    def _convert_excel_to_csv_resampled(self, excel_file_path, output_folder, sampling_frequency='1S'):
        """Converts and resamples an Excel file for STLA."""
        try:
            df = pd.read_excel(excel_file_path, sheet_name=0)
            if 'Timestamp' not in df.columns and 'Time' in df.columns:
                df.rename(columns={'Time': 'Timestamp'}, inplace=True)
            elif 'timestamp' in df.columns:
                 df.rename(columns={'timestamp': 'Timestamp'}, inplace=True)

            if 'Timestamp' not in df.columns:
                self.logger.error(f"A recognizable time column for resampling not found in STLA Excel {excel_file_path}. Saving as is.")
                self._convert_excel_to_csv(excel_file_path, output_folder) # Fallback
                return
            
            df_resampled = self._resample_data(df.copy(), sampling_frequency)
            if df_resampled is None or df_resampled.empty:
                self.logger.warning(f"Resampling failed or resulted in empty data for STLA Excel {excel_file_path}. Saving original processed version.")
                self._convert_excel_to_csv(excel_file_path, output_folder) # Fallback
                return

            csv_file_name = os.path.join(output_folder, os.path.splitext(os.path.basename(excel_file_path))[0] + '.csv')
            df_resampled.to_csv(csv_file_name, index=False)
            self.logger.info(f"Successfully converted and resampled STLA Excel {excel_file_path} to {csv_file_name}")
        except Exception as e:
            self.logger.error(f"Error converting/resampling STLA Excel {excel_file_path}: {e}")
        finally:
            if 'df' in locals(): del df
            if 'df_resampled' in locals(): del df_resampled
            gc.collect()
    # --- End of added CSV and Excel methods ---

    # _resample_data should be the same as the one in DataProcessorArbin (updated version)
    # If it's not, it needs to be updated here as well. Assuming it is for now.
    # For brevity, I'm not repeating the _resample_data method here if it's identical.
    # If it was different or older in STLA, it would need the same update as Arbin's.
    # Let's assume it's the updated one that handles non-numeric types.
    def _resample_data(self, df, sampling_frequency='1S'): # Ensure this is the updated version
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
                return None
            
            if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                try:
                    df[time_col] = pd.to_datetime(df[time_col])
                except ValueError:
                    try:
                        df[time_col] = pd.to_datetime(df[time_col], unit='s', errors='coerce')
                    except Exception as e_time_conv:
                         self.logger.error(f"Could not convert time column '{time_col}' to datetime: {e_time_conv}. Resampling skipped.")
                         return df

            if df[time_col].isnull().any():
                self.logger.warning(f"Null values found in '{time_col}' column after conversion. Dropping these rows before resampling.")
                df.dropna(subset=[time_col], inplace=True)

            if df.empty:
                self.logger.error(f"DataFrame is empty after handling NaT in '{time_col}'. Cannot resample.")
                return None

            df.set_index(time_col, inplace=True)

            numeric_cols = df.select_dtypes(include=np.number).columns
            df_resampled_numeric = df[numeric_cols].resample(sampling_frequency).mean().interpolate()

            non_numeric_cols = df.select_dtypes(exclude=np.number).columns
            if not non_numeric_cols.empty:
                df_non_numeric_indexed = df[non_numeric_cols]
                df_resampled_non_numeric = df_non_numeric_indexed.resample(sampling_frequency).first().ffill()
                df_resampled = pd.concat([df_resampled_numeric, df_resampled_non_numeric], axis=1)
            else:
                df_resampled = df_resampled_numeric
            
            df_resampled.reset_index(inplace=True)
            return df_resampled
        except Exception as e:
            self.logger.error(f"Error resampling data: {e}", exc_info=True)
            return None
    
    # Remove the static _extract_data_from_matfile as it's not used by STLA
    # def _extract_data_from_matfile(file_path):
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
            
    