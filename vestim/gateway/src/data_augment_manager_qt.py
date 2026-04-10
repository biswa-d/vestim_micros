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
from vestim.services.data_processor.src import normalization_service # Added for normalization
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
                           source_sampling_rate_override_hz: Optional[float] = None,
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
           resampling_source_hz = None
           resampling_source_mode = 'unknown'
           effective_filter_padding = self.service.determine_effective_filter_padding(
               filter_configs,
               padding_length
           ) if filter_configs else 0

           if filter_configs:
               self.logger.info(
                   f"Temporary pre-filter padding enabled: {effective_filter_padding} rows "
                   f"(added before filtering, removed after filtering)."
               )
           elif padding_length and padding_length > 0:
               self.logger.info(f"Persistent padding enabled (no filters): {padding_length} rows.")

           if train_processed_dir and os.path.isdir(train_processed_dir):
                train_files_for_stats_calc.extend(glob.glob(os.path.join(train_processed_dir, "*.csv")))
                all_files_to_process.extend(train_files_for_stats_calc)
           
           if val_processed_dir and os.path.isdir(val_processed_dir):
                all_files_to_process.extend(glob.glob(os.path.join(val_processed_dir, "*.csv")))
           if test_processed_dir and os.path.isdir(test_processed_dir):
                all_files_to_process.extend(glob.glob(os.path.join(test_processed_dir, "*.csv")))

           if resampling_frequency and resampling_frequency != 'None':
                if source_sampling_rate_override_hz is not None:
                    resampling_source_hz = float(source_sampling_rate_override_hz)
                    resampling_source_mode = 'manual_override'
                else:
                    candidate_files = train_files_for_stats_calc if train_files_for_stats_calc else all_files_to_process
                    last_profile_reason = None
                    last_profile_file = None
                    for candidate_file in candidate_files:
                        try:
                            candidate_df = pd.read_csv(candidate_file)
                            sampling_profile = self.service.detect_sampling_profile(candidate_df)
                            estimated_hz = sampling_profile.get('source_hz')
                            last_profile_reason = sampling_profile.get('reason')
                            last_profile_file = candidate_file
                            if estimated_hz and estimated_hz > 0:
                                resampling_source_hz = float(estimated_hz)
                                resampling_source_mode = sampling_profile.get('source_mode', 'auto_detected')
                                break
                        except Exception as e_est:
                            self.logger.debug(f"Sampling-rate estimation skipped for {candidate_file}: {e_est}")

                if resampling_source_hz:
                    self.logger.info(
                        f"Resampling source sampling rate resolved: {resampling_source_hz:.6g} Hz "
                        f"(mode={resampling_source_mode})"
                    )
                else:
                    self.logger.warning(
                        f"Could not infer source sampling rate from data; reason={last_profile_reason}, "
                        f"last_checked_file={last_profile_file}. Resampling service fallback will be used."
                    )
            
           if normalize_data:
                if not train_files_for_stats_calc:
                    self.logger.warning("Normalization requested, but no training files found to calculate statistics. Skipping normalization.")
                    normalize_data = False
                else:
                    self.logger.info("Preparing for normalization: performing preliminary processing on training files to gather data for stats.")
                    dataframes_for_stats = []
                    file_stats_data = []  # Track file-wise data for detailed statistics
                    for train_file_path_for_stats in train_files_for_stats_calc:
                        try:
                            df_temp_for_stats = pd.read_csv(train_file_path_for_stats)

                            if filter_configs and df_temp_for_stats is not None and not df_temp_for_stats.empty:
                                if effective_filter_padding > 0:
                                    df_temp_for_stats = self.service.pad_data(
                                        df_temp_for_stats,
                                        effective_filter_padding,
                                        resample_freq_for_time_padding=None
                                    )

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

                                if effective_filter_padding > 0:
                                    df_temp_for_stats = self.service.remove_padding(
                                        df_temp_for_stats,
                                        effective_filter_padding
                                    )

                            if column_formulas and df_temp_for_stats is not None and not df_temp_for_stats.empty:
                                df_temp_for_stats = self.service.create_columns(df_temp_for_stats, column_formulas)

                            if resampling_frequency and resampling_frequency != 'None' and not df_temp_for_stats.empty:
                                df_temp_for_stats = self.service.resample_data(
                                    df_temp_for_stats,
                                    resampling_frequency,
                                    source_sampling_rate_hz=resampling_source_hz
                                )
                                if self.service.last_resample_source_hz and not resampling_source_hz:
                                    resampling_source_hz = self.service.last_resample_source_hz
                                    resampling_source_mode = self.service.last_resample_source_mode or 'auto_detected'
                            
                            if df_temp_for_stats is not None and not df_temp_for_stats.empty:
                                dataframes_for_stats.append(df_temp_for_stats)
                                # Store file-wise data for detailed statistics
                                file_stats_data.append({
                                    'filename': os.path.basename(train_file_path_for_stats),
                                    'dataframe': df_temp_for_stats.copy()
                                })
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

                            # Hard safety: never normalize time/date/sample/index-like columns.
                            actual_columns_to_normalize = [
                                col for col in actual_columns_to_normalize
                                if not any(key in col.lower().replace(" ", "") for key in ['time', 'date', 'timestamp', 'sample', 'index'])
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
                                        saved_scaler_path = normalization_service.save_scaler(
                                            global_scaler, 
                                            scaler_output_dir, 
                                            filename=scaler_filename, 
                                            job_id=job_id,
                                            file_stats_data=file_stats_data
                                        )
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
                formula_error_occurred = False
                try:
                    df = pd.read_csv(file_path)
                    file_metadata['original_shape'] = df.shape

                    original_time_col = self.service._find_time_column(df)
                    original_time_snapshot = None
                    if original_time_col and original_time_col in df.columns:
                        original_time_snapshot = df[original_time_col].copy(deep=True)

                    # If processed_data already lost timestamp values, try recovering from matching raw_data file.
                    if (
                        (original_time_snapshot is None)
                        or (pd.Series(original_time_snapshot).notna().sum() == 0)
                    ):
                        try:
                            raw_file_path = file_path.replace(
                                os.path.join('processed_data', ''),
                                os.path.join('raw_data', '')
                            )
                            if os.path.exists(raw_file_path):
                                raw_df = pd.read_csv(raw_file_path)
                                raw_time_col = self.service._find_time_column(raw_df)
                                if raw_time_col and raw_time_col in raw_df.columns:
                                    raw_snapshot = raw_df[raw_time_col].copy(deep=True)
                                    if pd.Series(raw_snapshot).notna().sum() > 0:
                                        original_time_col = raw_time_col
                                        original_time_snapshot = raw_snapshot
                                        file_metadata['timestamp_snapshot_source'] = 'raw_data'
                                        self.logger.info(
                                            f"[{os.path.basename(file_path)}] Using timestamp snapshot from raw_data file: "
                                            f"{os.path.basename(raw_file_path)} (column={raw_time_col})"
                                        )
                        except Exception as e_raw_ts:
                            self.logger.warning(
                                f"[{os.path.basename(file_path)}] Could not load timestamp snapshot from raw_data: {e_raw_ts}",
                                exc_info=True
                            )

                    actual_resampling_frequency_for_padding = None
                    
                    if filter_configs and df is not None and not df.empty:
                        if effective_filter_padding > 0:
                            self.logger.info(f"[{os.path.basename(file_path)}] Adding temporary pre-filter padding: {effective_filter_padding} rows")
                            df = self.service.pad_data(
                                df,
                                effective_filter_padding,
                                resample_freq_for_time_padding=None
                            )

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

                        if effective_filter_padding > 0 and df is not None and not df.empty:
                            self.logger.info(f"[{os.path.basename(file_path)}] Removing temporary pre-filter padding: {effective_filter_padding} rows")
                            df = self.service.remove_padding(df, effective_filter_padding)

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

                    if resampling_frequency and resampling_frequency != 'None' and df is not None and not df.empty:
                        df = self.service.resample_data(
                            df,
                            resampling_frequency,
                            source_sampling_rate_hz=resampling_source_hz
                        )
                        if self.service.last_resample_source_hz and not resampling_source_hz:
                            resampling_source_hz = self.service.last_resample_source_hz
                            resampling_source_mode = self.service.last_resample_source_mode or 'auto_detected'
                        if df is not None and not df.empty:
                            actual_resampling_frequency_for_padding = resampling_frequency
                    
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
                        if filter_configs:
                            self.logger.debug(
                                f"[{os.path.basename(file_path)}] Persistent padding skipped because temporary filter padding is already applied/removed."
                            )
                        else:
                            df = self.service.pad_data(
                                df,
                                padding_length,
                                resample_freq_for_time_padding=actual_resampling_frequency_for_padding
                            )

                    if not formula_error_occurred and normalize_data and global_scaler and df is not None and not df.empty:
                        try:
                            # Explicit snapshot/restore safeguard to keep timestamp intact through normalization.
                            timestamp_col = self.service._find_time_column(df)
                            timestamp_snapshot = None
                            if timestamp_col and timestamp_col in df.columns:
                                timestamp_snapshot = df[timestamp_col].copy(deep=True)

                            df = self.service.apply_normalization(df, global_scaler, actual_columns_to_normalize)

                            if (
                                timestamp_snapshot is not None
                                and timestamp_col in df.columns
                                and len(df) == len(timestamp_snapshot)
                            ):
                                df[timestamp_col] = timestamp_snapshot.values
                                file_metadata['timestamp_restored_after_normalization'] = True
                        except Exception as e_norm:
                            self.logger.error(f"Error during normalization for {file_path}: {e_norm}", exc_info=True)
                            file_metadata['status'] = 'Failed'
                            file_metadata['error'] = f"Normalization error: {e_norm}"

                    # Final hard safeguard: if timestamp became missing/empty after augmentation,
                    # restore from original per-file snapshot (with alignment when lengths differ).
                    if (
                        file_metadata.get('status') != 'Failed'
                        and df is not None
                        and not df.empty
                        and original_time_snapshot is not None
                        and len(original_time_snapshot) > 0
                    ):
                        try:
                            target_time_col = self.service._find_time_column(df) or original_time_col
                            if target_time_col not in df.columns:
                                df[target_time_col] = np.nan

                            target_time_non_null = int(df[target_time_col].notna().sum())
                            if target_time_non_null == 0:
                                if len(original_time_snapshot) == len(df):
                                    df[target_time_col] = original_time_snapshot.values
                                    file_metadata['timestamp_restored_after_resampling'] = True
                                    file_metadata['timestamp_restore_mode'] = 'direct_copy'
                                else:
                                    original_series = pd.Series(original_time_snapshot).reset_index(drop=True)
                                    original_parsed = pd.to_datetime(original_series, errors='coerce', dayfirst=True)
                                    if original_parsed.notna().sum() >= 2:
                                        src_x = np.linspace(0.0, 1.0, num=len(original_parsed), dtype=float)
                                        dst_x = np.linspace(0.0, 1.0, num=len(df), dtype=float)
                                        src_ns = original_parsed.astype('int64').to_numpy(dtype=float)
                                        restored_ns = np.interp(dst_x, src_x, src_ns)
                                        restored_time = pd.to_datetime(restored_ns.astype('int64'), errors='coerce')
                                        df[target_time_col] = restored_time
                                        file_metadata['timestamp_restored_after_resampling'] = True
                                        file_metadata['timestamp_restore_mode'] = 'interpolated_datetime'
                                    else:
                                        src_pos = np.linspace(0, len(original_series) - 1, num=len(df))
                                        nearest_idx = np.clip(np.round(src_pos).astype(int), 0, len(original_series) - 1)
                                        df[target_time_col] = original_series.iloc[nearest_idx].values
                                        file_metadata['timestamp_restored_after_resampling'] = True
                                        file_metadata['timestamp_restore_mode'] = 'nearest_index'

                                restored_non_null = int(pd.Series(df[target_time_col]).notna().sum())
                                self.logger.info(
                                    f"[{os.path.basename(file_path)}] Timestamp safeguard applied: "
                                    f"column={target_time_col}, restored_non_null={restored_non_null}, "
                                    f"mode={file_metadata.get('timestamp_restore_mode', 'n/a')}"
                                )

                            final_non_null = int(pd.Series(df[target_time_col]).notna().sum())
                            if final_non_null == 0:
                                file_metadata['status'] = 'Failed'
                                file_metadata['error'] = (
                                    f"Timestamp column '{target_time_col}' remained empty after restoration safeguards."
                                )
                                self.logger.error(
                                    f"[{os.path.basename(file_path)}] {file_metadata['error']}"
                                )
                        except Exception as e_ts_restore:
                            self.logger.warning(
                                f"[{os.path.basename(file_path)}] Timestamp safeguard failed: {e_ts_restore}",
                                exc_info=True
                            )

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
               'frequency': resampling_frequency,
               'source_frequency_hz': resampling_source_hz,
               'source_frequency_mode': resampling_source_mode,
               'mixed_source_rates_hz': self.service.last_sampling_profile.get('mixed_rates_hz', []) if self.service.last_sampling_profile else [],
               'source_detection_reason': self.service.last_sampling_profile.get('reason') if self.service.last_sampling_profile else None
           } if resampling_frequency else {'applied': False}
           
           # Prepare padding info  
           padding_info = {
               'applied': True,
               'length': effective_filter_padding if filter_configs else padding_length,
               'resampling_frequency_for_padding': resampling_frequency,
               'mode': 'temporary_pre_filter' if filter_configs else 'persistent',
               'removed_before_save': bool(filter_configs),
               'user_padding_length': padding_length if (padding_length and padding_length > 0) else 0,
               'auto_filter_min_padding_length': effective_filter_padding if filter_configs else 0
           } if ((padding_length and padding_length > 0) or (filter_configs and effective_filter_padding > 0)) else {'applied': False}
           
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

    def detect_source_sampling_from_job(self, job_folder: str, max_files_to_scan: int = 36) -> Dict[str, Any]:
        """
        Detect source sampling profile by scanning representative CSV files in a job.
        Priority: train/processed_data -> train/raw_data -> val/test processed/raw.
        Returns profile with additional context fields for GUI/logging.
        """
        result: Dict[str, Any] = {
            'source_hz': None,
            'source_mode': 'unknown',
            'mixed': False,
            'mixed_rates_hz': [],
            'reason': 'no_files_scanned',
            'source_file': None,
            'source_stage': None,
            'scanned_files': 0
        }

        self._set_job_context(job_folder)

        # Original/source cadence should come from raw files first.
        stage_dirs: List[Tuple[str, str]] = [
            ('train_raw', os.path.join(job_folder, 'train_data', 'raw_data')),
            ('val_raw', os.path.join(job_folder, 'val_data', 'raw_data')),
            ('test_raw', os.path.join(job_folder, 'test_data', 'raw_data')),
            ('train_processed', os.path.join(job_folder, 'train_data', 'processed_data')),
            ('val_processed', os.path.join(job_folder, 'val_data', 'processed_data')),
            ('test_processed', os.path.join(job_folder, 'test_data', 'processed_data')),
        ]

        available_files = 0
        for _, stage_dir in stage_dirs:
            if os.path.isdir(stage_dir):
                available_files += len(glob.glob(os.path.join(stage_dir, "*.csv")))

        if available_files <= 0:
            self.logger.warning(f"Sampling detection at GUI launch: no CSV files found for job {job_folder}.")
            return result

        stage_count = max(1, len(stage_dirs))
        per_stage_cap = max(1, int(np.ceil(float(max_files_to_scan) / float(stage_count))))

        last_reason = 'unknown'
        for stage_name, stage_dir in stage_dirs:
            if result['scanned_files'] >= max_files_to_scan:
                break
            if not os.path.isdir(stage_dir):
                continue

            stage_files = sorted(glob.glob(os.path.join(stage_dir, "*.csv")))
            stage_scanned = 0
            for file_path in stage_files:
                if result['scanned_files'] >= max_files_to_scan or stage_scanned >= per_stage_cap:
                    break
                try:
                    df = pd.read_csv(file_path)
                    profile = self.service.detect_sampling_profile(df)
                    result['scanned_files'] += 1
                    stage_scanned += 1

                    detected_hz = profile.get('source_hz')
                    last_reason = profile.get('reason', 'unknown')

                    if not detected_hz and last_reason == 'no_time_column':
                        preview_cols = ", ".join([str(c) for c in list(df.columns)[:8]])
                        self.logger.debug(
                            f"Sampling detection no_time_column for {stage_name}:{os.path.basename(file_path)}; "
                            f"columns_preview=[{preview_cols}]"
                        )

                    if detected_hz and detected_hz > 0:
                        result.update(profile)
                        result['source_file'] = file_path
                        result['source_stage'] = stage_name
                        self.logger.info(
                            f"Sampling detection at GUI launch: detected {float(detected_hz):.6g} Hz "
                            f"(mode={profile.get('source_mode', 'auto_detected')}, mixed={profile.get('mixed', False)}) "
                            f"from {stage_name}: {file_path}"
                        )
                        return result
                except Exception as e:
                    self.logger.debug(f"Sampling detection skipped for {file_path}: {e}")

        result['reason'] = last_reason
        self.logger.warning(
            f"Sampling detection at GUI launch: unavailable after scanning {result['scanned_files']} file(s), "
            f"reason={last_reason}, job={job_folder}"
        )
        return result

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
            
            # Save only JSON file (which is actually used by training system)
            json_file = os.path.join(job_folder, 'data_files_reference.json')
            with open(json_file, 'w') as f:
                json.dump(data_reference, f, indent=2)
            
            self.logger.info(f"Saved data reference to: {json_file}")
            
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
