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
# 
# This class serves as an intermediary between the GUI and the data processing services
# ---------------------------------------------------------------------------------

import os
import io # Import io for string buffer
import glob # Import glob
import json # Added for metadata
import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Dict, Optional, Union, Any
from PyQt5.QtCore import QObject, pyqtSignal # Import QObject and pyqtSignal

# Removed QMessageBox import as it will be handled in the GUI thread
# from PyQt5.QtWidgets import QMessageBox 

from vestim.logger_config import setup_logger
from vestim.services.data_processor.src.data_augment_service import DataAugmentService
from vestim.gateway.src.job_manager_qt import JobManager # Corrected import
from vestim.services import normalization_service # Added for normalization
# Set up logging
logger = setup_logger(log_file='data_augment_manager.log')

class DataAugmentManager(QObject): # Inherit from QObject
    """Manager class for data augmentation operations"""
    
    # Signal to emit when a formula error occurs
    formulaErrorOccurred = pyqtSignal(str)
    # Signal to report progress (0-100), potentially useful for GUI updates
    augmentationProgress = pyqtSignal(int)
    # Signal to indicate completion (success or failure type)
    augmentationFinished = pyqtSignal(str, list) # job_folder, metadata list

    def __init__(self):
        """Initialize the DataAugmentManager"""
        super().__init__() # Call QObject constructor
        self.logger = logging.getLogger(__name__)
        self.service = DataAugmentService()
        self.job_manager = JobManager() # Instantiate JobManager
    
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

    # This method is deprecated as apply_augmentations now handles file-by-file loading.
    # def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
    #     # ... (original commented out code) ...

    def apply_augmentations(self,
                           job_folder: str,
                           padding_length: Optional[int] = None,
                           resampling_frequency: Optional[str] = None,
                           column_formulas: Optional[List[Tuple[str, str]]] = None,
                           normalize_data: bool = False,
                           normalization_feature_columns: Optional[List[str]] = None,
                           normalization_exclude_columns: Optional[List[str]] = None,
                           scaler_filename: str = "augmentation_scaler.joblib") -> Tuple[str, List[Dict[str, Any]]]:
       """
       Apply data augmentations (resampling, column creation, padding) to each file
       in the processed_data directories and saves them back, overwriting originals.
       Order: 1. Resampling, 2. Column Creation, 3. Padding.
       
       Args:
           job_folder: Path to the root job folder.
           padding_length: Number of rows to prepend for padding.
           resampling_frequency: Target frequency for resampling (e.g., "1Hz").
           column_formulas: List of tuples (column_name, formula) for creating new columns.
           
       Returns:
           A tuple containing:
               - Path to the root job folder.
               - A list of dictionaries, where each dictionary contains metadata about a processed file 
                 (including 'filepath', 'status', and 'error' if any).
       """
       normalize_data_initially_requested = normalize_data # Store the initial request
       self.logger.info(f"Starting file-by-file augmentation for job: {job_folder}")
       self.logger.info(f"Normalization requested: {normalize_data}")
       if normalize_data:
           self.logger.info(f"Normalization feature columns: {normalization_feature_columns}")
           self.logger.info(f"Normalization exclude columns: {normalization_exclude_columns}")
       self._set_job_context(job_folder)

       # Emit progress signal instead of using callback directly
       self.augmentationProgress.emit(0)

       processed_files_metadata = []
       
       try:
           global_scaler = None
           saved_scaler_path = None
           actual_columns_to_normalize = []

           train_processed_dir = self.job_manager.get_train_folder()
           test_processed_dir = self.job_manager.get_test_folder()

           if not train_processed_dir or not os.path.isdir(train_processed_dir):
                self.logger.warning(f"Train processed directory not found or invalid: {train_processed_dir}")
            
           if not test_processed_dir or not os.path.isdir(test_processed_dir):
                self.logger.warning(f"Test processed directory not found or invalid: {test_processed_dir}")

           all_files_to_process = []
           train_files_for_stats_calc = []

           if train_processed_dir and os.path.isdir(train_processed_dir):
                train_files_for_stats_calc.extend(glob.glob(os.path.join(train_processed_dir, "*.csv")))
                all_files_to_process.extend(train_files_for_stats_calc)
           if test_processed_dir and os.path.isdir(test_processed_dir):
                all_files_to_process.extend(glob.glob(os.path.join(test_processed_dir, "*.csv")))
            
           if normalize_data:
                if not train_files_for_stats_calc:
                    self.logger.warning("Normalization requested, but no training files found to calculate statistics. Skipping normalization.")
                    normalize_data = False # Disable normalization if no training data for stats
                elif not normalization_feature_columns:
                    self.logger.warning("Normalization requested, but no feature columns specified. Skipping normalization.")
                    normalize_data = False
                else:
                    self.logger.info("Preparing for normalization: performing preliminary processing on training files to gather data for stats.")
                    dataframes_for_stats = []
                    # Temporarily apply resampling and column creation to training files to get data for stats
                    for train_file_path_for_stats in train_files_for_stats_calc:
                        try:
                            df_temp_for_stats = pd.read_csv(train_file_path_for_stats) # TODO: Consider a more generic data loader if not always CSV
                            
                            # Apply resampling if enabled (using parameters passed to apply_augmentations)
                            if resampling_frequency and not df_temp_for_stats.empty:
                                df_temp_for_stats = self.service.resample_data(df_temp_for_stats, resampling_frequency)
                            
                            # Apply column creation if enabled
                            if column_formulas and df_temp_for_stats is not None and not df_temp_for_stats.empty:
                                df_temp_for_stats = self.service.create_columns(df_temp_for_stats, column_formulas)
                            
                            if df_temp_for_stats is not None and not df_temp_for_stats.empty:
                                dataframes_for_stats.append(df_temp_for_stats)
                            else:
                                self.logger.warning(f"Temporary DataFrame for stats from {train_file_path_for_stats} became empty/None after pre-processing. Skipping for stats.")
                        except Exception as e_preproc:
                            self.logger.error(f"Error during preliminary processing of {train_file_path_for_stats} for stats: {e_preproc}. Skipping this file for stats calculation.")
                            continue
                    
                    if not dataframes_for_stats:
                        self.logger.error("No valid DataFrames generated from training files for stats calculation after preliminary processing. Skipping normalization.")
                        normalize_data = False
                    else:
                        # Determine actual columns to normalize
                        # If normalization_feature_columns is None, infer from the first processed DataFrame
                        inferred_feature_columns = []
                        if not normalization_feature_columns and dataframes_for_stats:
                            first_df_for_cols = dataframes_for_stats[0]
                            inferred_feature_columns = [col for col in first_df_for_cols.columns if pd.api.types.is_numeric_dtype(first_df_for_cols[col])]
                            self.logger.info(f"Inferred numeric feature columns for normalization: {inferred_feature_columns}")
                            if not inferred_feature_columns:
                                self.logger.warning("Could not infer any numeric columns for normalization. Skipping normalization.")
                                normalize_data = False
                        
                        current_feature_columns_source = normalization_feature_columns if normalization_feature_columns else inferred_feature_columns

                        if normalize_data: # Re-check as it might have been set to False above
                            actual_columns_to_normalize = list(current_feature_columns_source) # Make a copy
                            if normalization_exclude_columns:
                                actual_columns_to_normalize = [col for col in actual_columns_to_normalize if col not in normalization_exclude_columns]
                            
                            if not actual_columns_to_normalize:
                                self.logger.warning("No columns remaining for normalization after exclusions. Skipping normalization.")
                                normalize_data = False
                            else:
                                self.logger.info(f"Actual columns to normalize after considering formulas and exclusions: {actual_columns_to_normalize}")
                                stats = normalization_service.calculate_global_dataset_stats(
                                    data_items=dataframes_for_stats, # Pass list of pre-processed DataFrames
                                    feature_columns=actual_columns_to_normalize
                                    # data_reading_func and read_kwargs are not used when data_items are DataFrames
                                )
                                if stats:
                                    global_scaler = normalization_service.create_scaler_from_stats(stats, actual_columns_to_normalize)
                                    if global_scaler:
                                        scaler_output_dir = os.path.join(job_folder, "scalers")
                                        saved_scaler_path = normalization_service.save_scaler(global_scaler, scaler_output_dir, filename=scaler_filename)
                                        if saved_scaler_path:
                                            self.logger.info(f"Global scaler saved to: {saved_scaler_path}")
                                            
                                            # --- Store scaler path and normalization info in job_metadata.json ---
                                            metadata_file_path = os.path.join(job_folder, "job_metadata.json")
                                            try:
                                                job_meta = {}
                                                if os.path.exists(metadata_file_path):
                                                    with open(metadata_file_path, 'r') as f_meta:
                                                        job_meta = json.load(f_meta)
                                                
                                                job_meta['normalization_applied'] = True
                                                # Store path relative to the job_folder for portability
                                                job_meta['scaler_path'] = os.path.relpath(saved_scaler_path, job_folder)
                                                job_meta['normalized_columns'] = actual_columns_to_normalize
                                                
                                                with open(metadata_file_path, 'w') as f_meta:
                                                    json.dump(job_meta, f_meta, indent=4)
                                                self.logger.info(f"Normalization metadata (scaler path, columns) saved to {metadata_file_path}")
                                            except Exception as e_meta_save:
                                                self.logger.error(f"Failed to save normalization metadata to {metadata_file_path}: {e_meta_save}")
                                            # --- End store metadata ---
                                        else:
                                            self.logger.error("Failed to save global scaler. Normalization will be skipped.")
                                            normalize_data = False # Disable if scaler not saved
                                            global_scaler = None
                                    else:
                                        self.logger.error("Failed to create global scaler from stats. Normalization will be skipped.")
                                        normalize_data = False # Disable if scaler not created
                        else:
                            self.logger.error("Failed to calculate global stats for normalization. Normalization will be skipped.")
                            normalize_data = False # Disable if stats calculation failed
            
            # If normalization was attempted but ultimately skipped, record this
           if normalize_data_initially_requested and not global_scaler: # Check if user wanted it AND it failed
                metadata_file_path = os.path.join(job_folder, "job_metadata.json")
                try:
                    job_meta = {}
                    if os.path.exists(metadata_file_path):
                        with open(metadata_file_path, 'r') as f_meta:
                            job_meta = json.load(f_meta)
                    job_meta['normalization_applied'] = False
                    job_meta.pop('scaler_path', None)
                    job_meta.pop('normalized_columns', None)
                    with open(metadata_file_path, 'w') as f_meta:
                        json.dump(job_meta, f_meta, indent=4)
                    self.logger.info(f"Recorded that normalization was attempted but skipped in {metadata_file_path}")
                except Exception as e_meta_save_fail:
                    self.logger.error(f"Failed to update normalization status (skipped) in {metadata_file_path}: {e_meta_save_fail}")

           if not all_files_to_process:
               self.logger.info("No CSV files found in processed directories to augment.")
               self.augmentationProgress.emit(100)
               self.service.update_augmentation_metadata(job_folder, processed_files_metadata)
               self.augmentationFinished.emit(job_folder, processed_files_metadata) # Emit finished signal
               return job_folder, processed_files_metadata # Still return for direct calls if needed

           total_files = len(all_files_to_process)
           self.logger.info(f"Found {total_files} CSV files to process.")

           for i, file_path in enumerate(all_files_to_process):
                self.logger.info(f"Processing file ({i+1}/{total_files}): {file_path}")
                file_metadata = {'filepath': file_path, 'status': 'Skipped', 'error': 'Unknown reason'} # Default status
                df = None # Initialize df
                try:
                    df = pd.read_csv(file_path)
                    file_metadata['original_shape'] = df.shape
                    
                    actual_resampling_frequency_for_padding = None # Store the frequency used for padding time

                    # 1. Resampling
                    if resampling_frequency and df is not None and not df.empty:
                        self.logger.info(f"DataFrame before resampling for {file_path}:")
                        buffer = io.StringIO()
                        df.info(buf=buffer)
                        info_str = buffer.getvalue()
                        self.logger.info(f"\n{info_str}\n{df.head().to_string()}")

                        self.logger.info(f"Resampling {file_path} to {resampling_frequency}")
                        df_after_resample = self.service.resample_data(df, resampling_frequency)
                        
                        self.logger.info(f"DataFrame after resampling for {file_path}:")
                        if df_after_resample is not None and not df_after_resample.empty:
                            buffer_after = io.StringIO()
                            df_after_resample.info(buf=buffer_after)
                            info_str_after = buffer_after.getvalue()
                            self.logger.info(f"\n{info_str_after}\n{df_after_resample.head().to_string()}")
                            df = df_after_resample
                            actual_resampling_frequency_for_padding = resampling_frequency # Store for padding
                        else:
                            self.logger.warning(f"DataFrame is empty or None after resampling for {file_path}. Subsequent steps might be skipped.")
                            df = df_after_resample # df could be None or empty here
                    
                    # 2. Column Creation
                    formula_error_occurred = False # Flag to indicate if a formula error stopped processing
                    if column_formulas and df is not None and not df.empty:
                        try:
                            self.logger.info(f"Applying {len(column_formulas)} column formulas to {file_path} (after potential resampling)")
                            df = self.service.create_columns(df, column_formulas)
                            self.logger.info(f"Shape after column creation for {file_path}: {df.shape if df is not None else 'None'}")
                        except ValueError as e_formula:
                            error_msg = f"Error applying formula to {os.path.basename(file_path)}: {e_formula}"
                            self.logger.error(error_msg, exc_info=True)
                            self.formulaErrorOccurred.emit(error_msg) 
                            file_metadata['status'] = 'Failed'
                            file_metadata['error'] = error_msg
                            formula_error_occurred = True 
                        except Exception as e_col_create:
                            self.logger.error(f"Unexpected error during column creation for {file_path}: {e_col_create}", exc_info=True)
                            file_metadata['status'] = 'Failed'
                            file_metadata['error'] = f"Unexpected column creation error: {e_col_create}"
                            # For other unexpected errors during column creation, we might still want to stop or handle differently.
                            # For now, it will be caught by the broader e_file exception if not saved.
                    
                    # 3. Normalization (New Step - only if no formula error and df is valid)
                    if not formula_error_occurred and normalize_data and global_scaler and df is not None and not df.empty:
                        self.logger.info(f"Applying normalization to {file_path} using global scaler for columns: {actual_columns_to_normalize}")
                        try:
                            # Assuming DataAugmentService will have a method to call normalization_service.transform_data
                            # Or, we can call it directly if DataAugmentService doesn't need to be involved here.
                            # For now, let's assume we add a method to DataAugmentService.
                            df = self.service.apply_normalization(df, global_scaler, actual_columns_to_normalize) # This method needs to be added to DataAugmentService
                            self.logger.info(f"Shape after normalization for {file_path}: {df.shape if df is not None else 'None'}")
                        except Exception as e_norm:
                            self.logger.error(f"Error during normalization for {file_path}: {e_norm}", exc_info=True)
                            # Decide if this should be a critical failure for the file
                            file_metadata['status'] = 'Failed'
                            file_metadata['error'] = f"Normalization error: {e_norm}"
                            # Potentially skip saving this file or stop, for now, it will mark as failed.
                            # To prevent saving, we could set df to None or re-raise, but that might be too disruptive.
                            # Let's assume it marks as failed and continues to padding if df is still valid.

                    # 4. Padding (only if no formula error and df is valid)
                    if not formula_error_occurred and padding_length and padding_length > 0 and df is not None and not df.empty:
                        self.logger.info(f"Applying padding of {padding_length} to {file_path} (after potential resampling, column creation, and normalization)")
                        df = self.service.pad_data(df, padding_length, resample_freq_for_time_padding=actual_resampling_frequency_for_padding)
                        self.logger.info(f"Shape after padding for {file_path}: {df.shape if df is not None else 'None'}")

                    # Save if no formula error occurred and df is valid (and no critical normalization error made df invalid)
                    if file_metadata['status'] != 'Failed' and df is not None and not df.empty: # Check status too
                        self.service.save_single_augmented_file(df, file_path)
                        file_metadata['augmented_shape'] = df.shape
                        file_metadata['columns'] = df.columns.tolist()
                        file_metadata['status'] = 'Success'
                        file_metadata.pop('error', None) 
                    elif not formula_error_occurred and (df is None or df.empty):
                        self.logger.warning(f"DataFrame for {file_path} is empty or None before saving (post-augmentation). Skipping save. Original file remains.")
                        if file_metadata['status'] != 'Failed': 
                            file_metadata['status'] = 'Failed'
                            file_metadata['error'] = 'DataFrame became empty/None during processing (e.g., resampling or other steps).'
                        file_metadata['augmented_shape'] = (0,0) 
                        file_metadata['columns'] = file_metadata.get('columns', []) # Keep original columns if available
                    # If formula_error_occurred, metadata is already set.

                except Exception as e_file: 
                    if not formula_error_occurred: # Avoid double logging if formula error already handled
                        self.logger.error(f"Failed to process file {file_path}: {e_file}", exc_info=True)
                        file_metadata['status'] = 'Failed'
                        file_metadata['error'] = str(e_file)
                
                processed_files_metadata.append(file_metadata)

                if formula_error_occurred:
                    self.logger.warning("Stopping augmentation process due to formula error.")
                    break # Exit the loop over files

                current_progress = int(((i + 1) / total_files) * 95)
                self.augmentationProgress.emit(current_progress)

           self.service.update_augmentation_metadata(job_folder, processed_files_metadata)

           self.augmentationProgress.emit(100)
           self.logger.info(f"File-by-file augmentation completed (or stopped) for job: {job_folder}")
           self.augmentationFinished.emit(job_folder, processed_files_metadata)
           return job_folder, processed_files_metadata

       except Exception as e:
            self.logger.error(f"Critical error during apply_augmentations for job {job_folder}: {e}", exc_info=True)
            self.augmentationProgress.emit(0) 
            if processed_files_metadata: 
                 self.service.update_augmentation_metadata(job_folder, processed_files_metadata)
            raise 
    
    def resample_data(self, df: pd.DataFrame, frequency: str) -> pd.DataFrame:
        """
        Resample data to the specified frequency
        
        Args:
            df: DataFrame to resample
            frequency: Target frequency (e.g., "1Hz")
            
        Returns:
            Resampled DataFrame
        """
        return self.service.resample_data(df, frequency)
    
    def validate_formula(self, formula: str, df: pd.DataFrame) -> bool: # Original signature, service handles tuple
        """
        Validate a formula against a DataFrame to ensure it can be evaluated
        
        Args:
            formula: Formula string to validate
            df: DataFrame with columns that may be referenced in the formula
            
        Returns:
            True if formula is valid, False otherwise
        """
        try:
            is_valid, _ = self.service.validate_formula(formula, df) # Unpack tuple
            return is_valid
        except Exception as e:
            self.logger.error(f"Formula validation failed in manager: {str(e)}")
            return False
    
    def get_column_info(self, job_folder: str) -> Dict[str, Dict[str, Any]]:
        """
        Get information about columns in the dataset. 
        Note: This currently loads the first train file to get column info.
        """
        self.logger.info(f"Getting column info for job: {job_folder}")
        self._set_job_context(job_folder)
        
        current_job_id = self.job_manager.get_job_id()
        if not current_job_id:
            self.logger.error("JobManager job_id is not set prior to get_column_info call.")
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
        Used by the GUI to get column names for UI setup (e.g., formula dialog).

        Args:
            job_folder: The root path of the job.

        Returns:
            A pandas DataFrame of the first train file, or None if not found.
        """
        self.logger.info(f"Attempting to load sample train dataframe for GUI from job: {job_folder}")
        self._set_job_context(job_folder)

        current_job_id = self.job_manager.get_job_id()
        if not current_job_id:
            self.logger.error("JobManager job_id is not set prior to get_sample_train_dataframe call.")
            return None

        try:
            train_processed_dir = self.job_manager.get_train_folder()
            if not train_processed_dir or not os.path.isdir(train_processed_dir):
                self.logger.warning(f"Train processed directory not found for job {current_job_id}: {train_processed_dir}")
                return None

            train_files = glob.glob(os.path.join(train_processed_dir, "*.csv"))
            if not train_files:
                self.logger.warning(f"No CSV files found in train processed directory for job {current_job_id}: {train_processed_dir}")
                return None
            
            first_file_path = train_files[0]
            self.logger.info(f"Loading sample train file for GUI: {first_file_path}")
            df = pd.read_csv(first_file_path)
            self.logger.info(f"Successfully loaded sample train file: {first_file_path}, shape: {df.shape}")
            return df

        except Exception as e:
            self.logger.error(f"Failed to load sample train dataframe for job {current_job_id}: {e}", exc_info=True)
            return None