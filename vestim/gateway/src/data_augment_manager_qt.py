# ---------------------------------------------------------------------------------
# Author: Biswanath Dehury
# Date: `2025-04-14`
# Version: 1.0.0
# Description: 
# Manager for data augmentation operations - provides functionality to:
# 1. Load data from job folders
# 2. Apply resampling operations to standardize data frequency
# 3. Create new columns using custom formulas provided by users
# 4. Save augmented data back to the job folder
# This class serves as an intermediary between the GUI and the data processing services
# ---------------------------------------------------------------------------------

import os
import io # Import io for string buffer
import glob # Import glob
import json # Added for metadata
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Union, Any
from PyQt5.QtCore import QObject, pyqtSignal # Import QObject and pyqtSignal

# Removed QMessageBox import as it will be handled in the GUI thread
# from PyQt5.QtWidgets import QMessageBox 

from vestim.logger_config import setup_logger
from vestim.services.data_processor.src.data_augment_service import DataAugmentService
from vestim.gateway.src.job_manager_qt import JobManager # Corrected import
from vestim.services import normalization_service # Added for normalization
import pandas as pd # Added for pd.api.types

# Set up logging
logger = setup_logger(log_file='data_augment_manager.log')

# Enhanced default exclusion list for normalization - includes common timestamp and index columns
DEFAULT_NORM_EXCLUDE_COLS = [
    'time', 'Time', 'timestamp', 'Timestamp', 'datetime', 'DateTime', 'DATE', 'Date',
    'Epoch', 'epoch', 'Cycle_Index', 'cycle_index', 'Step_Index', 'step_index', 
    'File_Index', 'file_index', 'Index', 'index', 'ID', 'id', 'Cycle', 'cycle',
    'Step', 'step', 'TimeStamp', 'TIMESTAMP', 'Time_s', 'time_s', 'seconds',
    'Status', 'status'  # Added Status as it's often categorical, not truly numeric
]

class DataAugmentManager(QObject): # Inherit from QObject
    """Manager class for data augmentation operations"""
    
    # Signal to emit when a formula error occurs
    formulaErrorOccurred = pyqtSignal(str)
    # Signal to report progress (0-100), potentially useful for GUI updates
    augmentationProgress = pyqtSignal(int)
    # Signal to indicate completion (success or failure type)
    augmentationFinished = pyqtSignal(str, list) # job_folder, metadata list

    def __init__(self, job_manager=None):
        """Initialize the DataAugmentManager"""
        super().__init__() # Call QObject constructor
        self.logger = logging.getLogger(__name__)
        self.service = DataAugmentService()
        self.job_manager = job_manager if job_manager else JobManager()
    
    def _set_job_context(self, job_folder: str):
        """Sets the JobManager's context to the given job_folder."""
        if not job_folder or not os.path.isdir(job_folder):
            self.logger.error(f"Invalid job_folder provided to _set_job_context: {job_folder}")
            raise ValueError(f"Invalid job folder: {job_folder}")
            
        job_id = os.path.basename(job_folder)
        if not job_id.startswith("job_"): # Basic validation
             self.logger.warning(f"Job folder '{job_id}' might not be a valid job ID format.")

        if self.job_manager.get_job_id() != job_id:
            self.logger.info(f"Setting JobManager's current job_id to: {job_id} (from path: {job_folder})")
            self.job_manager.job_id = job_id 
        elif self.job_manager.get_job_folder() != job_folder:
            self.logger.info(f"JobManager's job_id '{job_id}' matches, but ensuring folder context is updated using path: {job_folder}")
            self.job_manager.job_id = job_id

    def apply_augmentations(self,
                           job_folder: str,
                           padding_length: Optional[int] = None,
                           resampling_frequency: Optional[str] = None,
                           column_formulas: Optional[List[Tuple[str, str]]] = None,
                           normalize_data: bool = False,
                           normalization_feature_columns: Optional[List[str]] = None,
                           normalization_exclude_columns: Optional[List[str]] = None,
                           scaler_filename: str = "augmentation_scaler.joblib",
                           filter_configs: Optional[List[Dict[str, Any]]] = None,
                           noise_configs: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, List[Dict[str, Any]]]:
       """
       Apply data augmentations (resampling, column creation, padding) to each file
       in the processed_data directories and saves them back, overwriting originals.
       Order: 1. Resampling, 2. Filtering, 3. Column Creation, 4. Normalization, 5. Padding.
       """
       self.logger.info(f"Starting file-by-file augmentation for job: {job_folder}")
       self.logger.info(f"Normalization requested: {normalize_data}")
       if normalize_data:
           self.logger.info(f"Normalization feature columns: {normalization_feature_columns}")
           self.logger.info(f"Normalization exclude columns: {normalization_exclude_columns}")
       self._set_job_context(job_folder)

       self.augmentationProgress.emit(0)

       processed_files_metadata = []
       
       try:
           global_scaler = None
           saved_scaler_path = None
           actual_columns_to_normalize = []

           train_processed_dir = self.job_manager.get_train_folder()
           val_processed_dir = self.job_manager.get_val_folder()
           test_processed_dir = self.job_manager.get_test_folder()

           all_files_to_process = []
           train_files_for_stats_calc = []

           if train_processed_dir and os.path.isdir(train_processed_dir):
                train_files_for_stats_calc.extend(glob.glob(os.path.join(train_processed_dir, "*.csv")))
                all_files_to_process.extend(train_files_for_stats_calc)
           
           if val_processed_dir and os.path.isdir(val_processed_dir):
                all_files_to_process.extend(glob.glob(os.path.join(val_processed_dir, "*.csv")))
           if test_processed_dir and os.path.isdir(test_processed_dir):
                all_files_to_process.extend(glob.glob(os.path.join(test_processed_dir, "*.csv")))
            
           if normalize_data:
                if not train_files_for_stats_calc:
                    self.logger.warning("Normalization requested, but no training files found to calculate statistics. Skipping normalization.")
                    normalize_data = False
                else:
                    self.logger.info("Preparing for normalization: performing preliminary processing on training files to gather data for stats.")
                    dataframes_for_stats = []
                    for train_file_path_for_stats in train_files_for_stats_calc:
                        try:
                            df_temp_for_stats = pd.read_csv(train_file_path_for_stats)
                            if resampling_frequency and resampling_frequency != 'None' and not df_temp_for_stats.empty:
                                df_temp_for_stats = self.service.resample_data(df_temp_for_stats, resampling_frequency)
                            
                            if filter_configs and df_temp_for_stats is not None and not df_temp_for_stats.empty:
                                for config in filter_configs:
                                    output_column_name = config.get('output_column_name')
                                    df_temp_for_stats = self.service.apply_butterworth_filter(
                                        df_temp_for_stats,
                                        column_name=config['column'],
                                        corner_frequency=config['corner_frequency'],
                                        sampling_rate=config['sampling_rate'],
                                        filter_order=config.get('filter_order', 4),
                                        output_column_name=output_column_name
                                    )

                            if column_formulas and df_temp_for_stats is not None and not df_temp_for_stats.empty:
                                df_temp_for_stats = self.service.create_columns(df_temp_for_stats, column_formulas)
                            
                            if df_temp_for_stats is not None and not df_temp_for_stats.empty:
                                dataframes_for_stats.append(df_temp_for_stats)
                        except Exception as e_preproc:
                            self.logger.error(f"Error during preliminary processing of {train_file_path_for_stats} for stats: {e_preproc}. Skipping.")
                            continue
                    
                    if not dataframes_for_stats:
                        self.logger.error("No valid DataFrames generated from training files for stats calculation. Skipping normalization.")
                        normalize_data = False
                    else:
                        feature_columns_for_scaler_basis = []
                        if normalization_feature_columns:
                            feature_columns_for_scaler_basis = list(normalization_feature_columns)
                        else:
                            first_df_for_cols = dataframes_for_stats[0]
                            feature_columns_for_scaler_basis = [col for col in first_df_for_cols.columns if pd.api.types.is_numeric_dtype(first_df_for_cols[col])]

                        if not feature_columns_for_scaler_basis:
                            self.logger.warning("No basis feature columns for normalization. Skipping normalization.")
                            normalize_data = False
                        else:
                            if normalization_exclude_columns:
                                actual_columns_to_normalize = [col for col in feature_columns_for_scaler_basis if col not in normalization_exclude_columns]
                            else:
                                normalized_exclude_set = {col.lower().replace(" ", "") for col in DEFAULT_NORM_EXCLUDE_COLS}
                                actual_columns_to_normalize = [
                                    col for col in feature_columns_for_scaler_basis
                                    if col.lower().replace(" ", "") not in normalized_exclude_set
                                ]

                            if not actual_columns_to_normalize:
                                self.logger.warning("No columns remaining for normalization after exclusions. Skipping normalization.")
                                normalize_data = False
                            else:
                                self.logger.info(f"Final actual columns to normalize: {actual_columns_to_normalize}")
                                scaler_output_dir = os.path.join(job_folder, "scalers")
                                os.makedirs(scaler_output_dir, exist_ok=True)

                                stats = normalization_service.calculate_global_dataset_stats(
                                    data_items=dataframes_for_stats,
                                    feature_columns=actual_columns_to_normalize
                                )
                                if stats:
                                    global_scaler = normalization_service.create_scaler_from_stats(stats, actual_columns_to_normalize)
                                    if global_scaler:
                                        # Extract job_id from job_folder for metadata
                                        job_id = os.path.basename(job_folder)
                                        saved_scaler_path = normalization_service.save_scaler(global_scaler, scaler_output_dir, filename=scaler_filename, job_id=job_id)
                                        if not saved_scaler_path:
                                            self.logger.error("Failed to save global scaler. Normalization will be skipped.")
                                            normalize_data = False
                                            global_scaler = None
                                    else:
                                        self.logger.error("Failed to create global scaler from stats. Normalization will be skipped.")
                                        normalize_data = False
                                else:
                                    self.logger.error("Failed to calculate global stats for normalization. Normalization will be skipped.")
                                    normalize_data = False
           
           if not all_files_to_process:
               self.logger.info("No CSV files found in processed directories to augment.")
               self.augmentationProgress.emit(100)
               self.service.update_augmentation_metadata(job_folder, processed_files_metadata)
               self.augmentationFinished.emit(job_folder, processed_files_metadata)
               return job_folder, processed_files_metadata

           total_files = len(all_files_to_process)
           self.logger.info(f"Found {total_files} CSV files to process.")

           for i, file_path in enumerate(all_files_to_process):
                file_metadata = {'filepath': file_path, 'status': 'Skipped', 'error': 'Unknown reason'}
                df = None
                try:
                    df = pd.read_csv(file_path)
                    file_metadata['original_shape'] = df.shape
                    
                    actual_resampling_frequency_for_padding = None

                    if resampling_frequency and resampling_frequency != 'None' and df is not None and not df.empty:
                        df = self.service.resample_data(df, resampling_frequency)
                        if df is not None and not df.empty:
                            actual_resampling_frequency_for_padding = resampling_frequency
                    
                    if filter_configs and df is not None and not df.empty:
                        for config in filter_configs:
                            try:
                                df = self.service.apply_butterworth_filter(
                                    df,
                                    column_name=config['column'],
                                    corner_frequency=config['corner_frequency'],
                                    sampling_rate=config['sampling_rate'],
                                    filter_order=config.get('filter_order', 4),
                                    output_column_name=config.get('output_column_name')
                                )
                            except Exception as e_filter:
                                self.logger.error(f"Error applying filter for {file_path}: {e_filter}", exc_info=True)
                   
                    formula_error_occurred = False
                    if column_formulas and df is not None and not df.empty:
                        try:
                            df = self.service.create_columns(df, column_formulas, log_details=(i == 0))
                        except ValueError as e_formula:
                            error_msg = f"Error applying formula to {os.path.basename(file_path)}: {e_formula}"
                            self.logger.error(error_msg, exc_info=True)
                            self.formulaErrorOccurred.emit(error_msg) 
                            file_metadata['status'] = 'Failed'
                            file_metadata['error'] = error_msg
                            formula_error_occurred = True 
                    
                    # Apply noise injection if configured
                    if not formula_error_occurred and noise_configs and df is not None and not df.empty:
                        # Determine if this is a training/validation file for apply_to filtering
                        is_train_or_val = 'train' in file_path.lower() or 'val' in file_path.lower()
                        
                        for noise_config in noise_configs:
                            try:
                                # Check if we should apply noise to this file type
                                apply_to = noise_config.get('apply_to', 'train_val')
                                if apply_to == 'train_val' and not is_train_or_val:
                                    continue  # Skip test files when apply_to is train_val
                                    
                                df = self.service.apply_noise_injection(
                                    df,
                                    column_name=noise_config['column'],
                                    noise_type=noise_config['noise_type'],
                                    noise_level_percent=noise_config['noise_level']
                                )
                            except Exception as e_noise:
                                self.logger.error(f"Error applying noise injection for {file_path}: {e_noise}", exc_info=True)
                    
                    if not formula_error_occurred and padding_length and padding_length > 0 and df is not None and not df.empty:
                        df = self.service.pad_data(df, padding_length, resample_freq_for_time_padding=actual_resampling_frequency_for_padding)

                    if not formula_error_occurred and normalize_data and global_scaler and df is not None and not df.empty:
                        try:
                            df = self.service.apply_normalization(df, global_scaler, actual_columns_to_normalize)
                        except Exception as e_norm:
                            self.logger.error(f"Error during normalization for {file_path}: {e_norm}", exc_info=True)
                            file_metadata['status'] = 'Failed'
                            file_metadata['error'] = f"Normalization error: {e_norm}"

                    if file_metadata['status'] != 'Failed' and df is not None and not df.empty:
                        self.service.save_single_augmented_file(df, file_path)
                        file_metadata['augmented_shape'] = df.shape
                        file_metadata['columns'] = df.columns.tolist()
                        file_metadata['status'] = 'Success'
                        file_metadata.pop('error', None) 
                    elif not formula_error_occurred and (df is None or df.empty):
                        file_metadata['status'] = 'Failed'
                        file_metadata['error'] = 'DataFrame became empty/None during processing.'
                except Exception as e_file: 
                    if not formula_error_occurred:
                        self.logger.error(f"Failed to process file {file_path}: {e_file}", exc_info=True)
                        file_metadata['status'] = 'Failed'
                        file_metadata['error'] = str(e_file)
                
                processed_files_metadata.append(file_metadata)

                if formula_error_occurred:
                    self.logger.warning("Stopping augmentation process due to formula error.")
                    break

                current_progress = int(((i + 1) / total_files) * 95)
                self.augmentationProgress.emit(current_progress)

           normalization_info = {
               'applied': normalize_data and global_scaler is not None,
               'scaler_path': os.path.relpath(saved_scaler_path, job_folder) if saved_scaler_path else None,
               'normalized_columns': actual_columns_to_normalize if normalize_data and global_scaler else []
           }
           
           # Prepare resampling info
           resampling_info = {
               'applied': resampling_frequency is not None,
               'frequency': resampling_frequency
           } if resampling_frequency else {'applied': False}
           
           # Prepare padding info  
           padding_info = {
               'applied': padding_length is not None and padding_length > 0,
               'length': padding_length,
               'resampling_frequency_for_padding': resampling_frequency
           } if (padding_length and padding_length > 0) else {'applied': False}
           
           self.service.update_augmentation_metadata(
               job_folder, processed_files_metadata, 
               filter_configs=filter_configs, 
               normalization_info=normalization_info, 
               column_formulas=column_formulas,
               resampling_info=resampling_info,
               padding_info=padding_info
           )
           
           self._save_job_metadata(job_folder, normalize_data and global_scaler is not None, saved_scaler_path, actual_columns_to_normalize)

           try:
               train_folder_path = self.job_manager.get_train_folder_path() if hasattr(self.job_manager, 'get_train_folder_path') else ""
               val_folder_path = self.job_manager.get_val_folder_path() if hasattr(self.job_manager, 'get_val_folder_path') else ""
               test_folder_path = self.job_manager.get_test_folder_path() if hasattr(self.job_manager, 'get_test_folder_path') else ""
               self._save_simple_data_reference_safe(job_folder, train_folder_path, val_folder_path, test_folder_path)
           except Exception as ref_error:
               self.logger.warning(f"Could not save data reference (non-critical): {ref_error}")

           self.augmentationProgress.emit(100)
           self.logger.info(f"File-by-file augmentation completed for job: {job_folder}")
           self.augmentationFinished.emit(job_folder, processed_files_metadata)
           return job_folder, processed_files_metadata

       except Exception as e:
            self.logger.error(f"Critical error during apply_augmentations for job {job_folder}: {e}", exc_info=True)
            self.augmentationProgress.emit(0) 
            if processed_files_metadata: 
                 self.service.update_augmentation_metadata(job_folder, processed_files_metadata)
            raise 
    
    def _save_job_metadata(self, job_folder, normalization_applied, scaler_path, normalized_columns):
        """Consolidates and saves the job_metadata.json file."""
        metadata_file_path = os.path.join(job_folder, "job_metadata.json")
        try:
            job_meta = {}
            if os.path.exists(metadata_file_path):
                with open(metadata_file_path, 'r') as f_meta:
                    job_meta = json.load(f_meta)
            
            job_meta['normalization_applied'] = normalization_applied
            if normalization_applied and scaler_path:
                job_meta['scaler_path'] = os.path.relpath(scaler_path, job_folder)
                job_meta['normalized_columns'] = normalized_columns
            else:
                job_meta.pop('scaler_path', None)
                job_meta.pop('normalized_columns', None)
                job_meta.pop('scaler_stats_path', None)

            with open(metadata_file_path, 'w') as f_meta:
                json.dump(job_meta, f_meta, indent=4)
            self.logger.info(f"Job metadata saved to {metadata_file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save job metadata: {e}", exc_info=True)
    
    def resample_data(self, df: pd.DataFrame, frequency: str) -> pd.DataFrame:
        """
        Resample data to the specified frequency
        """
        return self.service.resample_data(df, frequency)
    
    def validate_formula(self, formula: str, df: pd.DataFrame) -> bool:
        """
        Validate a formula against a DataFrame to ensure it can be evaluated
        """
        try:
            is_valid, _ = self.service.validate_formula(formula, df)
            return is_valid
        except Exception as e:
            self.logger.error(f"Formula validation failed in manager: {str(e)}")
            return False
    
    def get_column_info(self, job_folder: str) -> Dict[str, Dict[str, Any]]:
        """
        Get information about columns in the dataset.
        """
        self.logger.info(f"Getting column info for job: {job_folder}")
        self._set_job_context(job_folder)
        
        current_job_id = self.job_manager.get_job_id()
        if not current_job_id:
            raise ValueError("Job context (job_id) not set in JobManager for get_column_info.")

        train_processed_dir = self.job_manager.get_train_folder()
        if not train_processed_dir or not os.path.isdir(train_processed_dir):
            self.logger.error(f"Train processed directory not found for get_column_info: {train_processed_dir}")
            return {} 

        train_files = glob.glob(os.path.join(train_processed_dir, "*.csv"))
        if not train_files:
            self.logger.info("No train files found in processed directory for get_column_info.")
            return {}

        try:
            first_train_file_df = pd.read_csv(train_files[0])
            return self.service.get_column_info(first_train_file_df)
        except Exception as e:
            self.logger.error(f"Failed to load first train file for get_column_info: {e}")
            return {}

    def get_sample_train_dataframe(self, job_folder: str) -> Optional[pd.DataFrame]:
        """
        Loads the first CSV file from the train processed directory for a given job folder.
        """
        self.logger.info(f"Attempting to load sample train dataframe for GUI from job: {job_folder}")
        self._set_job_context(job_folder)

        current_job_id = self.job_manager.get_job_id()
        if not current_job_id:
            return None

        try:
            train_processed_dir = self.job_manager.get_train_folder()
            if not train_processed_dir or not os.path.isdir(train_processed_dir):
                return None

            train_files = glob.glob(os.path.join(train_processed_dir, "*.csv"))
            if not train_files:
                return None
            
            first_file_path = train_files[0]
            df = pd.read_csv(first_file_path)
            return df

        except Exception as e:
            self.logger.error(f"Failed to load sample train dataframe for job {current_job_id}: {e}", exc_info=True)
            return None

    def _save_simple_data_reference(self, job_folder: str):
        """
        Save simple file references and sample counts for future reference.
        """
        try:
            train_raw_dir = os.path.join(job_folder, 'train_data', 'raw_data')
            val_raw_dir = os.path.join(job_folder, 'val_data', 'raw_data')
            test_raw_dir = os.path.join(job_folder, 'test_data', 'raw_data')
            
            data_reference = {
                'timestamp': datetime.now().isoformat(),
                'job_folder': job_folder,
                'train_files': [],
                'validation_files': [],
                'test_files': [],
                'total_train_samples': 0,
                'total_validation_samples': 0,
                'total_test_samples': 0
            }
            
            if os.path.exists(train_raw_dir):
                for filename in os.listdir(train_raw_dir):
                    if filename.endswith('.csv'):
                        file_path = os.path.join(train_raw_dir, filename)
                        try:
                            sample_count = sum(1 for _ in open(file_path)) - 1
                            data_reference['train_files'].append({'filename': filename, 'samples': sample_count})
                            data_reference['total_train_samples'] += sample_count
                        except:
                            data_reference['train_files'].append({'filename': filename, 'samples': 'unknown'})
            
            if os.path.exists(val_raw_dir):
                for filename in os.listdir(val_raw_dir):
                    if filename.endswith('.csv'):
                        file_path = os.path.join(val_raw_dir, filename)
                        try:
                            sample_count = sum(1 for _ in open(file_path)) - 1
                            data_reference['validation_files'].append({'filename': filename, 'samples': sample_count})
                            data_reference['total_validation_samples'] += sample_count
                        except:
                            data_reference['validation_files'].append({'filename': filename, 'samples': 'unknown'})
            
            if os.path.exists(test_raw_dir):
                for filename in os.listdir(test_raw_dir):
                    if filename.endswith('.csv'):
                        file_path = os.path.join(test_raw_dir, filename)
                        try:
                            sample_count = sum(1 for _ in open(file_path)) - 1
                            data_reference['test_files'].append({'filename': filename, 'samples': sample_count})
                            data_reference['total_test_samples'] += sample_count
                        except:
                            data_reference['test_files'].append({'filename': filename, 'samples': 'unknown'})
            
            reference_file = os.path.join(job_folder, 'data_files_reference.txt')
            with open(reference_file, 'w') as f:
                f.write("DATA FILES REFERENCE\n")
                f.write("=" * 50 + "\n")
                f.write(f"Job: {os.path.basename(job_folder)}\n")
                f.write(f"Created: {data_reference['timestamp']}\n\n")
                
                f.write(f"TRAINING FILES ({len(data_reference['train_files'])} files, {data_reference['total_train_samples']:,} total samples):\n")
                for file_info in data_reference['train_files']:
                    f.write(f"  • {file_info['filename']} - {file_info['samples']:,} samples\n")
                f.write("\n")
                
                f.write(f"VALIDATION FILES ({len(data_reference['validation_files'])} files, {data_reference['total_validation_samples']:,} total samples):\n")
                for file_info in data_reference['validation_files']:
                    f.write(f"  • {file_info['filename']} - {file_info['samples']:,} samples\n")
                f.write("\n")
                
                f.write(f"TEST FILES ({len(data_reference['test_files'])} files, {data_reference['total_test_samples']:,} total samples):\n")
                for file_info in data_reference['test_files']:
                    f.write(f"  • {file_info['filename']} - {file_info['samples']:,} samples\n")
                f.write("\n")
                
                f.write("NOTE: Original data files remain in their source locations.\n")
                f.write("Scaler statistics (min/max values) are saved separately in scalers/ folder.\n")
            
            json_file = os.path.join(job_folder, 'data_files_reference.json')
            with open(json_file, 'w') as f:
                json.dump(data_reference, f, indent=2)
            
            self.logger.info(f"Saved simple data reference to: {reference_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving simple data reference: {e}", exc_info=True)

    def _save_simple_data_reference_safe(self, job_folder: str, train_folder_path: str = "", val_folder_path: str = "", test_folder_path: str = ""):
        """
        Save simple file references in a way that won't interfere with multiprocessing.
        """
        try:
            import os
            import json
            from datetime import datetime
            
            train_raw_dir = os.path.join(job_folder, 'train_data', 'raw_data')
            val_raw_dir = os.path.join(job_folder, 'val_data', 'raw_data')
            test_raw_dir = os.path.join(job_folder, 'test_data', 'raw_data')
            
            data_reference = {
                'timestamp': datetime.now().isoformat(),
                'job_folder': os.path.basename(job_folder),
                'original_data_sources': {
                    'train': train_folder_path,
                    'validation': val_folder_path,
                    'test': test_folder_path
                },
                'train_files': [],
                'validation_files': [],
                'test_files': [],
                'total_train_samples': 0,
                'total_validation_samples': 0,
                'total_test_samples': 0
            }
            
            def count_csv_lines(file_path):
                try:
                    with open(file_path, 'r') as f:
                        return max(0, sum(1 for _ in f) - 1)
                except:
                    return 0
            
            if os.path.exists(train_raw_dir):
                for filename in os.listdir(train_raw_dir):
                    if filename.lower().endswith('.csv'):
                        file_path = os.path.join(train_raw_dir, filename)
                        sample_count = count_csv_lines(file_path)
                        data_reference['train_files'].append({'filename': filename, 'samples': sample_count})
                        data_reference['total_train_samples'] += sample_count
            
            if os.path.exists(val_raw_dir):
                for filename in os.listdir(val_raw_dir):
                    if filename.lower().endswith('.csv'):
                        file_path = os.path.join(val_raw_dir, filename)
                        sample_count = count_csv_lines(file_path)
                        data_reference['validation_files'].append({'filename': filename, 'samples': sample_count})
                        data_reference['total_validation_samples'] += sample_count
            
            if os.path.exists(test_raw_dir):
                for filename in os.listdir(test_raw_dir):
                    if filename.lower().endswith('.csv'):
                        file_path = os.path.join(test_raw_dir, filename)
                        sample_count = count_csv_lines(file_path)
                        data_reference['test_files'].append({'filename': filename, 'samples': sample_count})
                        data_reference['total_test_samples'] += sample_count
            
            reference_file = os.path.join(job_folder, 'data_files_reference.txt')
            with open(reference_file, 'w') as f:
                f.write("DATA FILES REFERENCE\n")
                f.write("=" * 50 + "\n")
                f.write(f"Job: {data_reference['job_folder']}\n")
                f.write(f"Created: {data_reference['timestamp']}\n\n")
                f.write("ORIGINAL DATA SOURCES:\n")
                f.write(f"  - Training: {data_reference['original_data_sources']['train']}\n")
                f.write(f"  - Validation: {data_reference['original_data_sources']['validation']}\n")
                f.write(f"  - Test: {data_reference['original_data_sources']['test']}\n\n")
                
                f.write(f"TRAINING FILES ({len(data_reference['train_files'])} files, {data_reference['total_train_samples']:,} total samples):\n")
                for file_info in data_reference['train_files']:
                    samples_str = f"{file_info['samples']:,}" if isinstance(file_info['samples'], int) else str(file_info['samples'])
                    f.write(f"  • {file_info['filename']} - {samples_str} samples\n")
                f.write("\n")
                
                f.write(f"VALIDATION FILES ({len(data_reference['validation_files'])} files, {data_reference['total_validation_samples']:,} total samples):\n")
                for file_info in data_reference['validation_files']:
                    samples_str = f"{file_info['samples']:,}" if isinstance(file_info['samples'], int) else str(file_info['samples'])
                    f.write(f"  • {file_info['filename']} - {samples_str} samples\n")
                f.write("\n")
                
                f.write(f"TEST FILES ({len(data_reference['test_files'])} files, {data_reference['total_test_samples']:,} total samples):\n")
                for file_info in data_reference['test_files']:
                    samples_str = f"{file_info['samples']:,}" if isinstance(file_info['samples'], int) else str(file_info['samples'])
                    f.write(f"  • {file_info['filename']} - {samples_str} samples\n")
                f.write("\n")
                
                f.write("NOTE: Original data files remain in their source locations.\n")
                f.write("Scaler statistics (min/max values) are saved separately in scalers/ folder.\n")
            
            json_file = os.path.join(job_folder, 'data_files_reference.json')
            with open(json_file, 'w') as f:
                json.dump(data_reference, f, indent=2)
            
            if hasattr(self, 'logger') and self.logger:
                self.logger.info(f"Saved simple data reference to: {reference_file}")
            
        except Exception as e:
            if hasattr(self, 'logger') and self.logger:
                self.logger.warning(f"Error saving simple data reference: {e}")
