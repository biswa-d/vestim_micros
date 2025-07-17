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
           val_processed_dir = self.job_manager.get_val_folder()
           test_processed_dir = self.job_manager.get_test_folder()

           if not train_processed_dir or not os.path.isdir(train_processed_dir):
                self.logger.warning(f"Train processed directory not found or invalid: {train_processed_dir}")
            
           if not val_processed_dir or not os.path.isdir(val_processed_dir):
                self.logger.warning(f"Validation processed directory not found or invalid: {val_processed_dir}")
            
           if not test_processed_dir or not os.path.isdir(test_processed_dir):
                self.logger.warning(f"Test processed directory not found or invalid: {test_processed_dir}")

           all_files_to_process = []
           train_files_for_stats_calc = []

           # CRITICAL: Only use training files to fit the scaler (proper ML practice)
           if train_processed_dir and os.path.isdir(train_processed_dir):
                train_files_for_stats_calc.extend(glob.glob(os.path.join(train_processed_dir, "*.csv")))
                all_files_to_process.extend(train_files_for_stats_calc)
           
           # Add validation and test files for processing (but NOT for scaler fitting)
           if val_processed_dir and os.path.isdir(val_processed_dir):
                all_files_to_process.extend(glob.glob(os.path.join(val_processed_dir, "*.csv")))
           if test_processed_dir and os.path.isdir(test_processed_dir):
                all_files_to_process.extend(glob.glob(os.path.join(test_processed_dir, "*.csv")))
            
           if normalize_data:
                if not train_files_for_stats_calc:
                    self.logger.warning("Normalization requested, but no training files found to calculate statistics. Skipping normalization.")
                    normalize_data = False # This is the mutable processing flag
                else:
                    # All checks passed so far for initial conditions, proceed with pre-processing for stats
                    self.logger.info("Preparing for normalization: performing preliminary processing on training files to gather data for stats.")
                    self.logger.info("IMPORTANT: Scaler will be fit ONLY on training data (proper ML practice)")
                    self.logger.info(f"Training files for scaler fitting: {len(train_files_for_stats_calc)} files")
                    self.logger.info(f"Total files to normalize (train+val+test): {len(all_files_to_process)} files")
                    
                    dataframes_for_stats = []
                    for train_file_path_for_stats in train_files_for_stats_calc:
                        try:
                            df_temp_for_stats = pd.read_csv(train_file_path_for_stats)
                            if resampling_frequency and not df_temp_for_stats.empty:
                                df_temp_for_stats = self.service.resample_data(df_temp_for_stats, resampling_frequency)
                            if column_formulas and df_temp_for_stats is not None and not df_temp_for_stats.empty:
                                df_temp_for_stats = self.service.create_columns(df_temp_for_stats, column_formulas)
                            
                            if df_temp_for_stats is not None and not df_temp_for_stats.empty:
                                dataframes_for_stats.append(df_temp_for_stats)
                            else:
                                self.logger.warning(f"Temp DataFrame for stats from {train_file_path_for_stats} became empty/None. Skipping for stats.")
                        except Exception as e_preproc:
                            self.logger.error(f"Error during preliminary processing of {train_file_path_for_stats} for stats: {e_preproc}. Skipping.")
                            continue
                    
                    if not dataframes_for_stats:
                        self.logger.error("No valid DataFrames generated from training files for stats calculation. Skipping normalization.")
                        normalize_data = False
                    else:
                        # Determine feature columns for the scaler
                        feature_columns_for_scaler_basis = []
                        if normalization_feature_columns: # User explicitly provided columns
                            feature_columns_for_scaler_basis = list(normalization_feature_columns)
                            self.logger.info(f"Using user-provided feature columns for normalization basis: {feature_columns_for_scaler_basis}")
                        else: # Infer numeric columns from the first (processed) training DataFrame
                            first_df_for_cols = dataframes_for_stats[0]
                            feature_columns_for_scaler_basis = [col for col in first_df_for_cols.columns if pd.api.types.is_numeric_dtype(first_df_for_cols[col])]
                            self.logger.info(f"Inferred numeric columns for normalization basis: {feature_columns_for_scaler_basis}")

                        if not feature_columns_for_scaler_basis:
                            self.logger.warning("No basis feature columns (user-provided or inferred) for normalization. Skipping normalization.")
                            normalize_data = False
                        else:
                            # Apply exclusions to get the final list of columns to normalize
                            if normalization_exclude_columns: # User-defined exclusions
                                actual_columns_to_normalize = [col for col in feature_columns_for_scaler_basis if col not in normalization_exclude_columns]
                                self.logger.info(f"Applying user-defined exclusions: {normalization_exclude_columns}")
                            else: # Default exclusions
                                actual_columns_to_normalize = [col for col in feature_columns_for_scaler_basis if col not in DEFAULT_NORM_EXCLUDE_COLS]
                                self.logger.info(f"Applying default exclusions: {DEFAULT_NORM_EXCLUDE_COLS}")

                            if not actual_columns_to_normalize:
                                self.logger.warning("No columns remaining for normalization after exclusions. Skipping normalization.")
                                normalize_data = False
                            else:
                                self.logger.info(f"Final actual columns to normalize: {actual_columns_to_normalize}")
                                
                                scaler_output_dir = os.path.join(job_folder, "scalers") # Define earlier for stats file
                                try:
                                    os.makedirs(scaler_output_dir, exist_ok=True)
                                except OSError as e_mkdir:
                                    self.logger.error(f"Could not create scaler directory {scaler_output_dir}: {e_mkdir}. Normalization may fail to save outputs.")
                                    # Potentially set normalize_data = False here if dir creation is critical

                                # Now, calculate stats using these columns and the processed DataFrames
                                stats = normalization_service.calculate_global_dataset_stats(
                                    data_items=dataframes_for_stats,
                                    feature_columns=actual_columns_to_normalize
                                )
                                if stats:
                                    # --- Save the calculated global min/max stats to a JSON file ---
                                    global_min_series = stats.get('min')
                                    global_max_series = stats.get('max')
                                    stats_json_path = None # Initialize

                                    if global_min_series is not None and global_max_series is not None:
                                        stats_to_save_dict = {
                                            'comment': f"Global min/max statistics used for scaler '{scaler_filename}' on job '{os.path.basename(job_folder)}'",
                                            'normalized_columns_for_stats': actual_columns_to_normalize, # Columns these stats are for
                                            'global_min': global_min_series.to_dict(),
                                            'global_max': global_max_series.to_dict()
                                        }
                                        stats_json_path = os.path.join(scaler_output_dir, "scaler_global_stats.json")
                                        try:
                                            with open(stats_json_path, 'w') as f_stats:
                                                json.dump(stats_to_save_dict, f_stats, indent=4)
                                            self.logger.info(f"Saved global min/max stats to {stats_json_path}")
                                        except Exception as e_stats_save:
                                            self.logger.error(f"Failed to save global stats JSON to {stats_json_path}: {e_stats_save}")
                                            stats_json_path = None # Indicate failure to save
                                    else:
                                        self.logger.warning("Stats dictionary from calculate_global_dataset_stats was missing 'min' or 'max' series.")
                                    # --- End save global stats ---

                                    global_scaler = normalization_service.create_scaler_from_stats(stats, actual_columns_to_normalize)
                                    if global_scaler:
                                        # scaler_output_dir is already defined and created
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
                                                if stats_json_path: # Add path to the stats JSON if it was saved
                                                    job_meta['scaler_stats_path'] = os.path.relpath(stats_json_path, job_folder)
                                                
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
                    
                    actual_resampling_frequency_for_padding = None # Store the frequency used for padding

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
           
           # Save simple data file reference for future traceability
           try:
               self._save_simple_data_reference_safe(job_folder)
           except Exception as ref_error:
               self.logger.warning(f"Could not save data reference (non-critical): {ref_error}")

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

    def _save_simple_data_reference(self, job_folder: str):
        """
        Save simple file references and sample counts for future reference.
        This is called during augmentation when we already have access to the data.
        """
        try:
            # Get raw data directories (where original files are before processing)
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
            
            # Collect training file names and count samples
            if os.path.exists(train_raw_dir):
                for filename in os.listdir(train_raw_dir):
                    if filename.endswith('.csv'):
                        file_path = os.path.join(train_raw_dir, filename)
                        try:
                            # Quick sample count (just count lines)
                            sample_count = sum(1 for _ in open(file_path)) - 1  # -1 for header
                            data_reference['train_files'].append({
                                'filename': filename,
                                'samples': sample_count
                            })
                            data_reference['total_train_samples'] += sample_count
                        except:
                            # If can't count, just add filename
                            data_reference['train_files'].append({'filename': filename, 'samples': 'unknown'})
            
            # Collect validation file names and count samples
            if os.path.exists(val_raw_dir):
                for filename in os.listdir(val_raw_dir):
                    if filename.endswith('.csv'):
                        file_path = os.path.join(val_raw_dir, filename)
                        try:
                            sample_count = sum(1 for _ in open(file_path)) - 1
                            data_reference['validation_files'].append({
                                'filename': filename,
                                'samples': sample_count
                            })
                            data_reference['total_validation_samples'] += sample_count
                        except:
                            data_reference['validation_files'].append({'filename': filename, 'samples': 'unknown'})
            
            # Collect test file names and count samples
            if os.path.exists(test_raw_dir):
                for filename in os.listdir(test_raw_dir):
                    if filename.endswith('.csv'):
                        file_path = os.path.join(test_raw_dir, filename)
                        try:
                            sample_count = sum(1 for _ in open(file_path)) - 1
                            data_reference['test_files'].append({
                                'filename': filename,
                                'samples': sample_count
                            })
                            data_reference['total_test_samples'] += sample_count
                        except:
                            data_reference['test_files'].append({'filename': filename, 'samples': 'unknown'})
            
            # Save simple reference file
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
            
            # Also save as JSON for programmatic access if needed
            json_file = os.path.join(job_folder, 'data_files_reference.json')
            with open(json_file, 'w') as f:
                json.dump(data_reference, f, indent=2)
            
            self.logger.info(f"Saved simple data reference to: {reference_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving simple data reference: {e}", exc_info=True)

    def _save_simple_data_reference_safe(self, job_folder: str):
        """
        Save simple file references in a way that won't interfere with multiprocessing.
        Uses only basic Python operations and avoids any potential import issues.
        """
        try:
            import os
            import json
            from datetime import datetime
            
            # Get raw data directories (where original files are before processing)
            train_raw_dir = os.path.join(job_folder, 'train_data', 'raw_data')
            val_raw_dir = os.path.join(job_folder, 'val_data', 'raw_data')
            test_raw_dir = os.path.join(job_folder, 'test_data', 'raw_data')
            
            data_reference = {
                'timestamp': datetime.now().isoformat(),
                'job_folder': os.path.basename(job_folder),  # Just job name, not full path
                'train_files': [],
                'validation_files': [],
                'test_files': [],
                'total_train_samples': 0,
                'total_validation_samples': 0,
                'total_test_samples': 0
            }
            
            # Helper function to count lines safely
            def count_csv_lines(file_path):
                try:
                    with open(file_path, 'r') as f:
                        return max(0, sum(1 for _ in f) - 1)  # -1 for header
                except:
                    return 0
            
            # Collect training file names and count samples
            if os.path.exists(train_raw_dir):
                for filename in os.listdir(train_raw_dir):
                    if filename.lower().endswith('.csv'):
                        file_path = os.path.join(train_raw_dir, filename)
                        sample_count = count_csv_lines(file_path)
                        data_reference['train_files'].append({
                            'filename': filename,
                            'samples': sample_count
                        })
                        data_reference['total_train_samples'] += sample_count
            
            # Collect validation file names and count samples
            if os.path.exists(val_raw_dir):
                for filename in os.listdir(val_raw_dir):
                    if filename.lower().endswith('.csv'):
                        file_path = os.path.join(val_raw_dir, filename)
                        sample_count = count_csv_lines(file_path)
                        data_reference['validation_files'].append({
                            'filename': filename,
                            'samples': sample_count
                        })
                        data_reference['total_validation_samples'] += sample_count
            
            # Collect test file names and count samples
            if os.path.exists(test_raw_dir):
                for filename in os.listdir(test_raw_dir):
                    if filename.lower().endswith('.csv'):
                        file_path = os.path.join(test_raw_dir, filename)
                        sample_count = count_csv_lines(file_path)
                        data_reference['test_files'].append({
                            'filename': filename,
                            'samples': sample_count
                        })
                        data_reference['total_test_samples'] += sample_count
            
            # Save simple reference file
            reference_file = os.path.join(job_folder, 'data_files_reference.txt')
            with open(reference_file, 'w') as f:
                f.write("DATA FILES REFERENCE\n")
                f.write("=" * 50 + "\n")
                f.write(f"Job: {data_reference['job_folder']}\n")
                f.write(f"Created: {data_reference['timestamp']}\n\n")
                
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
            
            # Also save as JSON for programmatic access if needed
            json_file = os.path.join(job_folder, 'data_files_reference.json')
            with open(json_file, 'w') as f:
                json.dump(data_reference, f, indent=2)
            
            # Use logger only if available, don't fail if not
            if hasattr(self, 'logger') and self.logger:
                self.logger.info(f"Saved simple data reference to: {reference_file}")
            
        except Exception as e:
            # Fail silently to avoid affecting main workflow
            if hasattr(self, 'logger') and self.logger:
                self.logger.warning(f"Error saving simple data reference: {e}")