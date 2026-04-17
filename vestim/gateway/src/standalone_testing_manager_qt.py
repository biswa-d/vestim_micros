import os
import json
import datetime
import re
import pandas as pd
import joblib
import torch
import numpy as np
import time
import gc, sys
from PyQt5.QtCore import QObject, pyqtSignal
from vestim.services.data_processor.src.data_augment_service import DataAugmentService
from vestim.services.model_training.src.FNN_model import FNNModel
from vestim.services.model_training.src.LSTM_model import LSTMModel
from vestim.services.model_training.src.GRU_model import GRUModel
from vestim.services.model_testing.src.testing_service import apply_inference_filter
from vestim.services.model_testing.src.continuous_testing_service import ContinuousTestingService
from vestim.services.data_processor.src import normalization_service as norm_svc

class VEstimStandaloneTestingManager(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    results_ready = pyqtSignal(dict)
    augmentation_required = pyqtSignal(pd.DataFrame, list)

    def __init__(self, job_folder_path, test_data_path, session_timestamp=None, inference_filter_override=None):
        super().__init__()
        self.job_folder_path = job_folder_path
        self.test_data_path = test_data_path
        self.session_timestamp = session_timestamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.inference_filter_override = inference_filter_override or {}
        self.data_augment_service = DataAugmentService()
        self.test_df = None
        self.overall_results = {}
        self.padding_length = 0
        self.inference_filter_override = inference_filter_override or None
        self.resampling_applied = False
        self.resampling_frequency = None
        self.augmented_test_df = None
        self.inference_test_file_path = None

    @staticmethod
    def _extract_timestamp_like_array(df: pd.DataFrame):
        """Extract a usable x-axis from a dataframe, preferring time-like columns, then sample/index columns."""
        if df is None or df.empty:
            return None

        def _looks_like_datetime_text(series: pd.Series) -> bool:
            sample = series.dropna().astype(str).str.strip().head(10)
            if sample.empty:
                return False
            if sample.str.fullmatch(r'\d+').all():
                return False
            return sample.str.contains(r'[-/:T ]', regex=True).any()

        def _valid_series_or_none(series: pd.Series):
            if series is None or series.empty:
                return None
            non_empty = series.dropna()
            if non_empty.empty:
                return None
            if pd.api.types.is_datetime64_any_dtype(non_empty):
                return series.values
            if pd.api.types.is_numeric_dtype(non_empty):
                return None
            if non_empty.dtype == object:
                cleaned = non_empty.astype(str).str.strip()
                cleaned = cleaned[~cleaned.isin(['', 'nan', 'NaN', 'None', 'NaT'])]
                if cleaned.empty or not _looks_like_datetime_text(cleaned):
                    return None
                parsed = pd.to_datetime(cleaned, errors='coerce', dayfirst=True)
                if parsed.notna().sum() >= 2:
                    return parsed.values
                return None
            return None

        for col in df.columns:
            normalized = col.lower().replace(" ", "")
            if normalized in {'time(h)', 'time_hours', 'hours'}:
                continue
            if 'time' in normalized or 'date' in normalized:
                valid = _valid_series_or_none(df[col])
                if valid is not None:
                    return valid

        for col in df.columns:
            normalized = col.lower().replace(" ", "")
            if 'sample' in normalized or normalized in ['index', 'idx']:
                valid = _valid_series_or_none(df[col])
                if valid is not None:
                    return valid

        return None

    @staticmethod
    def _build_time_hours_from_axis(axis_values, length_fallback: int):
        """Build a stable Time (h) array from an existing axis; falls back to 1 Hz index-based hours."""
        if axis_values is None:
            return np.arange(length_fallback, dtype=float) / 3600.0

        axis_series = pd.Series(np.ravel(np.asarray(axis_values)))
        if axis_series.empty:
            return np.arange(length_fallback, dtype=float) / 3600.0

        def _parse_datetime(series: pd.Series) -> pd.Series:
            if pd.api.types.is_datetime64_any_dtype(series):
                return series

            parsed_default = pd.to_datetime(series, errors='coerce')
            parsed_dayfirst = pd.to_datetime(series, errors='coerce', dayfirst=True)
            parsed = parsed_dayfirst if parsed_dayfirst.notna().sum() > parsed_default.notna().sum() else parsed_default
            if parsed.notna().sum() >= 2:
                return parsed

            numeric = pd.to_numeric(series, errors='coerce')
            if numeric.notna().sum() >= 2:
                for unit in ['s', 'ms', 'us', 'ns']:
                    parsed_unit = pd.to_datetime(numeric, unit=unit, errors='coerce')
                    if parsed_unit.notna().sum() >= 2:
                        return parsed_unit

            return parsed

        parsed = _parse_datetime(axis_series)
        if parsed.notna().sum() >= 2:
            valid = parsed.dropna()
            base_time = valid.iloc[0]
            elapsed = (parsed.ffill().bfill() - base_time).dt.total_seconds() / 3600.0
            elapsed = elapsed.astype(float)
            elapsed = np.maximum(elapsed.to_numpy(), 0.0)
            return elapsed

        return np.arange(len(axis_series), dtype=float) / 3600.0
    
    def start(self):
        """Start the testing process (called by test selection GUI)"""
        self.start_testing()
        
    def start_testing(self):
        """Public interface method for starting testing (called by GUI)"""
        self.run_test()

    def run_test(self):
        try:
            self.resampling_applied = False
            self.resampling_frequency = None
            self.augmented_test_df = None
            self.inference_test_file_path = None
            self.progress.emit("Starting test...")
            self.progress.emit("Loading configurations...")
            if self.inference_filter_override:
                self.progress.emit(
                    f"Using standalone inference filter override: {self.inference_filter_override.get('INFERENCE_FILTER_TYPE', 'None')}"
                )
            job_metadata_path = os.path.join(self.job_folder_path, 'job_metadata.json')
            self.aug_metadata_path = os.path.join(self.job_folder_path, 'augmentation_metadata.json')
            
            with open(job_metadata_path, 'r') as f:
                self.job_metadata = json.load(f)

            self.progress.emit(f"Loading test data from {os.path.basename(self.test_data_path)}...")
            
            # Add support for different file types
            file_extension = os.path.splitext(self.test_data_path)[1].lower()
            if file_extension == '.csv':
                self.test_df = pd.read_csv(self.test_data_path)
            elif file_extension in ['.xlsx', '.xls']:
                self.test_df = pd.read_excel(self.test_data_path, sheet_name=0) # Default to first sheet
            else:
                # Fallback or error
                self.error.emit(f"Unsupported file type: {file_extension}. Please use .csv or .xlsx.")
                self.finished.emit()
                return

            self.original_test_df = self.test_df.copy()  # Store original for later saving
            
            # Try to automatically apply augmentation steps
            if os.path.exists(self.aug_metadata_path):
                self.progress.emit("Found augmentation metadata. Applying automatic augmentation...")
                augmented_df = self._apply_automatic_augmentation(self.test_df)
                self.resume_test_with_augmented_data(augmented_df)
            else:
                # Fallback to old validation approach
                if self.job_metadata.get('normalization_applied', False):
                    normalized_columns = self.job_metadata.get('normalized_columns')
                    missing_cols = [col for col in normalized_columns if col not in self.test_df.columns]
                    if missing_cols:
                        raise ValueError(f"Test data is missing columns required by the scaler: {missing_cols}")
                
                self.resume_test_with_augmented_data(self.test_df)

        except ValueError as e:
            if "missing columns required by the scaler" in str(e) or "Could not apply" in str(e):
                self.progress.emit(f"Automatic augmentation failed: {e}")
                self.progress.emit("Opening manual augmentation interface...")
                filter_configs = self._load_augmentation_configs()
                if filter_configs:
                    # Convert to legacy format for GUI compatibility
                    legacy_configs = [config for config in filter_configs if config.get("type") == "filter"]
                    self.augmentation_required.emit(self.test_df, legacy_configs)
                else:
                    self.progress.emit("Could not find augmentation steps. Please prepare the data manually.")
                    self.finished.emit()
            else:
                self.error.emit(f"Augmentation validation error: {e}")
                self.finished.emit()
        except Exception as e:
            self.error.emit(f"Testing failed: {e}")
            self.finished.emit()

    def resume_test_with_augmented_data(self, augmented_df):
        try:
            # Create consolidated test files directory using session timestamp
            test_file_basename = os.path.splitext(os.path.basename(self.test_data_path))[0]
            
            # Main job folder level: new_test_timestamp_files (single directory for all test files)
            new_tests_dir = os.path.join(self.job_folder_path, f'new_test_{self.session_timestamp}_files')
            os.makedirs(new_tests_dir, exist_ok=True)
            
            # Save raw test data for reference
            raw_test_file = os.path.join(new_tests_dir, f"raw_{test_file_basename}.csv")
            self.original_test_df.to_csv(raw_test_file, index=False)
            
            # Save augmented test data (the one that will be used for inference)
            augmented_test_file = os.path.join(new_tests_dir, f"augmented_{test_file_basename}.csv")
            augmented_df.to_csv(augmented_test_file, index=False)
            
            self.progress.emit(f"Test files saved in: {os.path.basename(new_tests_dir)}")
            self.progress.emit(f"Raw file: raw_{test_file_basename}.csv")
            self.progress.emit(f"Augmented file: augmented_{test_file_basename}.csv")
            
            # Actually save the files now
            if hasattr(self, 'original_test_df') and self.original_test_df is not None:
                self.original_test_df.to_csv(raw_test_file, index=False)
            else:
                # If original_test_df is not available, save from file
                original_df = pd.read_csv(self.test_data_path)
                original_df.to_csv(raw_test_file, index=False)
                
            # Save the augmented data
            augmented_df.to_csv(augmented_test_file, index=False)
            self.inference_test_file_path = augmented_test_file
            
            # Store test session metadata
            self.test_session_id = self.session_timestamp
            self.session_dir = new_tests_dir
            
            self.test_df = augmented_df
            self.augmented_test_df = augmented_df.copy()
            self.progress.emit("Resuming test with augmented data...")
            



            scaler = None
            if self.job_metadata.get('normalization_applied', False):

                self.progress.emit("Loading scaler for denormalization...")
                scaler_path = os.path.normpath(os.path.join(self.job_folder_path, 'scalers', 'augmentation_scaler.joblib'))


                scaler = norm_svc.load_scaler(scaler_path)
                if scaler:
                    self.progress.emit("✓ Scaler loaded successfully")
                    normalized_columns = self.job_metadata.get('normalized_columns')
                    
                    # Only normalize columns that actually exist in the test data
                    # (augmentation may not have been applied yet for raw test files)
                    available_normalized_cols = [col for col in normalized_columns if col in self.test_df.columns]
                    available_normalized_cols = [
                        col for col in available_normalized_cols
                        if not any(key in col.lower().replace(" ", "") for key in ['time', 'date', 'timestamp', 'sample', 'index'])
                    ]
                    missing_cols = [col for col in normalized_columns if col not in self.test_df.columns]
                    
                    if missing_cols:
                        self.progress.emit(f"Warning: Some normalized columns not found in test data: {missing_cols}")
                        self.progress.emit("This is normal if you're using a raw test file without augmentation.")
                    
                    if not available_normalized_cols:
                        self.progress.emit("Warning: No normalized columns found in test data. Skipping normalization.")
                    else:
                        # Get the actual columns that the scaler was trained on
                        scaler_features = list(scaler.feature_names_in_) if hasattr(scaler, 'feature_names_in_') else available_normalized_cols
                        self.progress.emit(f"[DEBUG] Scaler feature_names_in_: {scaler_features}")
                        self.progress.emit(f"[DEBUG] Test DataFrame columns: {list(self.test_df.columns)}")
                        self.progress.emit(f"[DEBUG] Test DataFrame dtypes:\n{self.test_df.dtypes}")
                        
                        # Pre-process object-type columns (like 'Prog Time', 'Step Time') to numeric
                        # before calling scaler.transform(), since the scaler was trained on numeric values
                        mm_ss_pattern = re.compile(r'^\d+:\d+(?:\.\d+)?$')
                        hh_mm_ss_pattern = re.compile(r'^\d+:\d{2}:\d{2}(?:\.\d+)?$')

                        for col in self.test_df.columns:
                            col_dtype_str = str(self.test_df[col].dtype)
                            if col in scaler_features and col_dtype_str in ('object', 'string', 'str'):
                                self.progress.emit(f"[DEBUG] Found object column in scaler: '{col}'")
                                try:
                                    col_as_str = self.test_df[col].astype(str).str.strip()
                                    non_empty = col_as_str[col_as_str != '']
                                    sample_val = non_empty.iloc[0] if len(non_empty) > 0 else ""
                                    self.progress.emit(f"[DEBUG] Sample value from '{col}': {sample_val}")

                                    # MM:SS(.s) formatted elapsed time (e.g. 56:43.4)
                                    if sample_val and mm_ss_pattern.match(sample_val):
                                        self.progress.emit(f"[DEBUG] Detected MM:SS format in '{col}', converting to seconds")
                                        def mm_ss_to_seconds(x):
                                            if pd.isna(x):
                                                return np.nan
                                            value = str(x).strip()
                                            if value == '':
                                                return np.nan
                                            if mm_ss_pattern.match(value):
                                                parts = value.split(':')
                                                return int(parts[0]) * 60 + float(parts[1])
                                            numeric = pd.to_numeric(value, errors='coerce')
                                            return numeric if not pd.isna(numeric) else np.nan
                                        
                                        self.test_df[col] = self.test_df[col].apply(mm_ss_to_seconds)
                                        self.progress.emit(f"[DEBUG] After conversion, '{col}' dtype: {self.test_df[col].dtype}")
                                        self.progress.emit(f"Converted '{col}' from MM:SS format to seconds.")

                                    # HH:MM:SS(.s) duration format
                                    elif sample_val and hh_mm_ss_pattern.match(sample_val):
                                        self.progress.emit(f"[DEBUG] Detected HH:MM:SS format in '{col}', converting to seconds")

                                        def hh_mm_ss_to_seconds(x):
                                            if pd.isna(x):
                                                return np.nan
                                            value = str(x).strip()
                                            if value == '':
                                                return np.nan
                                            if hh_mm_ss_pattern.match(value):
                                                parts = value.split(':')
                                                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
                                            numeric = pd.to_numeric(value, errors='coerce')
                                            return numeric if not pd.isna(numeric) else np.nan

                                        self.test_df[col] = self.test_df[col].apply(hh_mm_ss_to_seconds)
                                        self.progress.emit(f"[DEBUG] After conversion, '{col}' dtype: {self.test_df[col].dtype}")
                                        self.progress.emit(f"Converted '{col}' from HH:MM:SS format to seconds.")

                                    # Date-time strings (e.g. 10-13-2025 19:33:39.903)
                                    elif sample_val and any(ch in sample_val for ch in ('-', '/')) and ':' in sample_val and ' ' in sample_val:
                                        self.progress.emit(f"[DEBUG] Detected datetime-like format in '{col}', converting to elapsed seconds")
                                        parsed_dt = pd.to_datetime(self.test_df[col], errors='coerce')
                                        if parsed_dt.notna().any():
                                            first_ts = parsed_dt.dropna().iloc[0]
                                            self.test_df[col] = (parsed_dt - first_ts).dt.total_seconds()
                                            self.progress.emit(f"[DEBUG] After datetime conversion, '{col}' dtype: {self.test_df[col].dtype}")
                                            self.progress.emit(f"Converted '{col}' from datetime to elapsed seconds.")
                                        else:
                                            self.test_df[col] = pd.to_numeric(self.test_df[col], errors='coerce')
                                            self.progress.emit(f"[DEBUG] Datetime parse failed; fallback to to_numeric for '{col}'")

                                    else:
                                        self.progress.emit(f"[DEBUG] No ':' found, trying generic numeric conversion for '{col}'")
                                        # Try generic numeric conversion
                                        self.test_df[col] = pd.to_numeric(self.test_df[col], errors='coerce')
                                        self.progress.emit(f"[DEBUG] After to_numeric, '{col}' dtype: {self.test_df[col].dtype}")
                                        self.progress.emit(f"Converted '{col}' to numeric.")
                                except (ValueError, TypeError, AttributeError) as e:
                                    self.progress.emit(f"[ERROR] Could not convert '{col}': {e}")

                        # Only transform columns that the scaler knows about
                        cols_to_transform = [col for col in scaler_features if col in self.test_df.columns]
                        missing_scaler_features = [col for col in scaler_features if col not in self.test_df.columns]
                        self.progress.emit(f"[DEBUG] Columns to transform: {cols_to_transform}")

                        if missing_scaler_features:
                            raise ValueError(
                                "Augmented test data is missing scaler-required columns: "
                                f"{missing_scaler_features}. "
                                "This usually means some augmentation steps (e.g., filter outputs) were not applied."
                            )
                        
                        if cols_to_transform:
                            # Final dtype check before transform
                            self.progress.emit(f"[DEBUG] Final dtypes before scaler.transform():")
                            for col in cols_to_transform:
                                self.progress.emit(f"[DEBUG]   {col}: {self.test_df[col].dtype}")
                            
                            self.test_df[cols_to_transform] = scaler.transform(self.test_df[cols_to_transform])
                            self.progress.emit(f"Normalization applied to {len(cols_to_transform)} columns.")
                            # Ensure inference path uses the fully prepared (normalized) test file
                            if self.inference_test_file_path:
                                self.test_df.to_csv(self.inference_test_file_path, index=False)
                        else:
                            self.progress.emit("Warning: No columns could be matched with scaler.")

                else:
                    self.progress.emit("Warning: Failed to load scaler, predictions will be on normalized scale")
            else:

                self.progress.emit("Normalization was not applied during training. Skipping.")

            models_dir = os.path.join(self.job_folder_path, 'models')


            if not os.path.exists(models_dir):
                self.progress.emit("No models directory found in job folder.")
                self.finished.emit()
                return
                
            # Scan all task directories in all model architecture folders
            self.progress.emit("Scanning for trained models...")

            task_directories = self._scan_task_directories(models_dir)

            
            if not task_directories:
                self.progress.emit("No trained models found.")

                self.finished.emit()
                return
                
            self.progress.emit(f"Found {len(task_directories)} trained models:")

            for i, task_info in enumerate(task_directories):
                arch_name = task_info.get('architecture_name', 'Unknown')
                task_name = task_info.get('task_name', 'Unknown')

                self.progress.emit(f"  - {arch_name}/{task_name}")
                model_type = task_info.get('model_type', 'Unknown')
                self.progress.emit(f"  - {arch_name}/{task_name} ({model_type})")
            
            # Use the session timestamp that was passed to constructor
            self.progress.emit(f"\nStandalone test session: {self.session_timestamp}")

            
            # Test each model using existing task structure
            self.progress.emit(f"\n{'='*60}")
            self.progress.emit("STARTING STANDALONE TESTING")
            self.progress.emit(f"{'='*60}")

            
            successful_tests = 0
            failed_tests = 0
            
            for i, task_info in enumerate(task_directories):
                try:

                    self.progress.emit(f"\n--- Testing {i+1}/{len(task_directories)}: {task_info['architecture_name']}/{task_info['task_name']} ---")
                    success = self._test_task_model(task_info, self.test_df.copy(), scaler, self.job_metadata)

                    if success:
                        successful_tests += 1
                    else:
                        failed_tests += 1
                except Exception as e:
                    self.progress.emit(f"ERROR: {e}")
                    failed_tests += 1
            
            # Final summary
            self.progress.emit(f"\n{'='*60}")
            self.progress.emit("STANDALONE TESTING COMPLETE")
            self.progress.emit(f"{'='*60}")
            self.progress.emit(f"Total models tested: {len(task_directories)}")
            self.progress.emit(f"Successful: {successful_tests}")
            self.progress.emit(f"Failed: {failed_tests}")
            self.progress.emit(f"Test session: {self.session_timestamp}")
            
            self.finished.emit()

        except Exception as e:
            self.progress.emit(f"\nERROR: {e}")
            self.finished.emit()

    def _apply_automatic_augmentation(self, df):
        """Automatically apply all augmentation steps from metadata."""
        try:
            with open(self.aug_metadata_path, 'r') as f:
                metadata = json.load(f)
            
            result_df = df.copy()
            self.padding_length = 0

            padding_info = metadata.get('padding', {})

            # Apply filters
            applied_filters = metadata.get('applied_filters', [])
            temp_filter_padding_length = 0

            if applied_filters and padding_info.get('applied', False):
                padding_mode = padding_info.get('mode')
                removed_before_save = padding_info.get('removed_before_save', False)

                if padding_mode == 'temporary_pre_filter' or removed_before_save:
                    temp_filter_padding_length = int(padding_info.get('length', 0) or 0)

            if applied_filters and temp_filter_padding_length > 0:
                self.progress.emit(f"Applying temporary pre-filter padding (length: {temp_filter_padding_length})...")
                result_df = self.data_augment_service.pad_data(
                    result_df,
                    temp_filter_padding_length,
                    resample_freq_for_time_padding=padding_info.get('resampling_frequency_for_padding')
                )

            for filter_config in applied_filters:
                column_to_filter = filter_config['column']
                # Check if column exists, if not provide helpful error
                if column_to_filter not in result_df.columns:
                    available_cols = ', '.join(result_df.columns.tolist())
                    raise ValueError(
                        f"Filter requires column '{column_to_filter}' but it's not in test data.\n"
                        f"Available columns: {available_cols}\n"
                        f"Make sure your test file has the same raw column names as training data."
                    )
                self.progress.emit(f"Applying Butterworth filter to '{column_to_filter}'...")
                result_df = self.data_augment_service.apply_butterworth_filter(
                    result_df,
                    column_name=column_to_filter,
                    corner_frequency=filter_config['corner_frequency'],
                    sampling_rate=filter_config['sampling_rate'],
                    filter_order=filter_config['filter_order'],
                    output_column_name=filter_config['output_column_name']
                )

            if applied_filters and temp_filter_padding_length > 0:
                self.progress.emit(f"Removing temporary filter padding (length: {temp_filter_padding_length})...")
                result_df = self.data_augment_service.remove_padding(result_df, temp_filter_padding_length)
                self.padding_length = 0

            # Apply calculated columns before resampling so derived features are resampled with the rest.
            created_columns = metadata.get('created_columns', [])
            if created_columns:
                column_formulas = [(col['column_name'], col['formula']) for col in created_columns]
                self.progress.emit(f"Creating {len(column_formulas)} calculated columns...")
                result_df = self.data_augment_service.create_columns(result_df, column_formulas)

            # Apply resampling after filtering and column creation to match the main augmentation pipeline.
            resampling_info = metadata.get('resampling', {})
            self.resampling_applied = bool(resampling_info.get('applied', False))
            self.resampling_frequency = resampling_info.get('frequency')
            if resampling_info.get('applied', False):
                self.progress.emit(f"Applying resampling to {resampling_info.get('frequency', 'unknown')} frequency...")
                source_hz = resampling_info.get('source_frequency_hz', resampling_info.get('original_sample_rate_hz'))
                result_df = self.data_augment_service.resample_data(
                    result_df,
                    resampling_info.get('frequency'),
                    source_sampling_rate_hz=source_hz
                )

            # Apply persistent test padding (for warmup) after filter-related temporary padding is removed.
            if padding_info.get('applied', False):
                persistent_scope = str(padding_info.get('persistent_scope', 'legacy')).strip().lower()
                if persistent_scope == 'test_only':
                    persistent_padding_length = int(padding_info.get('user_padding_length', 0) or 0)
                else:
                    # Legacy behavior: only non-filter metadata used `length` as persistent padding.
                    persistent_padding_length = int(padding_info.get('length', 0) or 0) if (not applied_filters) else 0

                if persistent_padding_length > 0:
                    self.padding_length = persistent_padding_length
                    self.progress.emit(f"Applying persistent test padding (length: {persistent_padding_length})...")
                    result_df = self.data_augment_service.pad_data(
                        result_df,
                        persistent_padding_length,
                        resample_freq_for_time_padding=padding_info.get('resampling_frequency_for_padding')
                    )
            
            self.progress.emit(f"✓ Automatic augmentation completed. Shape: {result_df.shape}")
            return result_df
            
        except Exception as e:
            raise ValueError(f"Could not apply automatic augmentation: {e}")

    def _load_augmentation_configs(self):
        """Load augmentation configurations, preferring structured JSON over text log."""
        # Try to load from structured JSON first
        if hasattr(self, 'aug_metadata_path') and os.path.exists(self.aug_metadata_path):
            try:
                self.progress.emit("Loading augmentation configuration from structured metadata...")
                with open(self.aug_metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                filter_configs = metadata.get('applied_filters', [])
                created_columns = metadata.get('created_columns', [])
                
                # Combine filters and created columns into a single list for processing
                all_configs = []
                
                # Add filter configurations
                for config in filter_configs:
                    all_configs.append({
                        "type": "filter",
                        "column": config.get("column"),
                        "output_column_name": config.get("output_column_name"),
                        "filter_order": config.get("filter_order"),
                        "corner_frequency": config.get("corner_frequency"),
                        "sampling_rate": config.get("sampling_rate")
                    })
                
                # Add created column configurations
                for config in created_columns:
                    all_configs.append({
                        "type": "calculated",
                        "column_name": config.get("column_name"),
                        "formula": config.get("formula")
                    })
                
                self.progress.emit(f"Found {len(filter_configs)} filters and {len(created_columns)} calculated columns in structured metadata.")
                return all_configs
                
            except Exception as e:
                self.progress.emit(f"Warning: Could not parse structured metadata: {e}")
        
        # Fallback - return empty list since we removed text log support
        return []

    def _scan_task_directories(self, models_dir):
        """Scan for all task directories in the existing job structure."""
        task_directories = []
        
        try:
            # Iterate through architecture folders (FNN_32_64, LSTM_64_128, etc.)
            for arch_folder in os.listdir(models_dir):
                arch_path = os.path.join(models_dir, arch_folder)
                if not os.path.isdir(arch_path):
                    continue
                
                # Iterate through task folders (B1024_rep-1, B256_rep-1, etc.)
                for task_folder in os.listdir(arch_path):
                    task_path = os.path.join(arch_path, task_folder)
                    if not os.path.isdir(task_path):
                        continue
                    
                    task_info_file = os.path.join(task_path, 'task_info.json')
                    best_model_file = os.path.join(task_path, 'best_model.pth')
                    
                    if os.path.exists(task_info_file) and os.path.exists(best_model_file):
                        try:
                            # Load task information
                            with open(task_info_file, 'r') as f:
                                task_info = json.load(f)
                            
                            # Add path information
                            task_info['architecture_name'] = arch_folder
                            task_info['task_name'] = task_folder
                            task_info['task_path'] = task_path
                            task_info['model_file'] = best_model_file
                            task_info['task_info_file'] = task_info_file
                            
                            task_directories.append(task_info)
                            
                        except Exception as e:
                            self.progress.emit(f"Warning: Could not load task info from {task_info_file}: {e}")
                    else:
                        if not os.path.exists(task_info_file):
                            self.progress.emit(f"Warning: Missing task_info.json in {task_path}")
                        if not os.path.exists(best_model_file):
                            self.progress.emit(f"Warning: Missing best_model.pth in {task_path}")
        
        except Exception as e:
            self.progress.emit(f"Error scanning task directories: {e}")
        
        return task_directories

    def _test_task_model(self, task_info, test_df, scaler, job_metadata):
        """Test a single model using the existing task structure."""
        model = None
        try:
            task_info = dict(task_info)
            arch_name = task_info['architecture_name']
            task_name = task_info['task_name']
            task_path = task_info['task_path']
            model_file = task_info['model_file']
            
            self.progress.emit(f"  Architecture: {arch_name}")
            self.progress.emit(f"  Task: {task_name}")
            
            # Define device for model and tensors
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Extract model configuration from task_info
            model_metadata = task_info.get('model_metadata', {})
            model_type = model_metadata.get('model_type', task_info.get('model_type', 'FNN'))
            hyperparams = dict(task_info.get('hyperparams', {}))
            if self.inference_filter_override:
                hyperparams.update(self.inference_filter_override)
                task_info['hyperparams'] = hyperparams
            data_config = task_info.get('data_config', {})
            training_config = task_info.get('training_config', {})
            
            feature_columns = hyperparams.get('FEATURE_COLUMNS', data_config.get('feature_columns', []))
            target_column = hyperparams.get('TARGET_COLUMN', data_config.get('target_column'))
            
            lookback_val = hyperparams.get('LOOKBACK', data_config.get('lookback', 0))
            lookback = 0 if lookback_val == "N/A" or lookback_val is None else int(lookback_val)
            
            training_method = hyperparams.get('TRAINING_METHOD', training_config.get('training_method', 'Sequential'))
            
            self.progress.emit(f"  Model Type: {model_type}")
            self.progress.emit(f"  Target: {target_column}")
            self.progress.emit(f"  Features: {len(feature_columns)} columns")
            if lookback > 0:
                self.progress.emit(f"  Lookback: {lookback} time steps")
            
            # Validate required columns
            missing_cols = [col for col in feature_columns if col not in test_df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Use the exact same inference engine as the main testing loop.
            inference_file_path = self.inference_test_file_path or self.test_data_path
            self.progress.emit(f"  Running continuous inference on: {os.path.basename(inference_file_path)}")
            start_time = time.time()

            service_task = dict(task_info)
            service_task['job_metadata'] = job_metadata
            service_task['job_folder_augmented_from'] = self.job_folder_path
            service_task['hyperparams'] = dict(service_task.get('hyperparams', {}))
            self._apply_inference_filter_override(service_task['hyperparams'])
            if 'data_loader_params' not in service_task or not service_task.get('data_loader_params'):
                service_task['data_loader_params'] = {
                    'feature_columns': feature_columns,
                    'target_column': target_column
                }

            continuous_testing_service = ContinuousTestingService(device=device)
            file_results = continuous_testing_service.run_continuous_testing(
                task=service_task,
                model_path=model_file,
                test_file_path=inference_file_path,
                is_first_file=True,
                warmup_samples=lookback
            )

            if file_results is None:
                raise ValueError("Continuous testing service returned no results.")

            predictions_final = np.ravel(np.asarray(file_results.get('predictions', [])))
            actual_values = np.ravel(np.asarray(file_results.get('true_values', [])))
            inference_time = time.time() - start_time

            if len(predictions_final) == 0 or len(actual_values) == 0:
                raise ValueError("Continuous testing returned empty predictions/targets.")

            timestamps = None
            if isinstance(self.augmented_test_df, pd.DataFrame) and not self.augmented_test_df.empty:
                timestamps = self._extract_timestamp_like_array(self.augmented_test_df)
            if timestamps is None and isinstance(test_df, pd.DataFrame) and not test_df.empty:
                timestamps = self._extract_timestamp_like_array(test_df)
            if timestamps is None and inference_file_path and os.path.exists(inference_file_path) and inference_file_path.lower().endswith('.csv'):
                try:
                    inference_df = pd.read_csv(inference_file_path)
                    timestamps = self._extract_timestamp_like_array(inference_df)
                except Exception:
                    timestamps = None

            if timestamps is None:
                timestamps = np.arange(len(predictions_final))

            timestamps = np.ravel(np.asarray(timestamps))
            target_len = min(len(predictions_final), len(actual_values), len(timestamps))

            if len(predictions_final) != len(actual_values) or len(predictions_final) != len(timestamps):
                self.progress.emit(
                    f"  Warning: prediction/target/timestamp length mismatch "
                    f"({len(predictions_final)} vs {len(actual_values)} vs {len(timestamps)}). "
                    f"Truncating to {target_len}."
                )

            if target_len <= 0:
                raise ValueError(
                    f"Invalid test output lengths: pred={len(predictions_final)}, true={len(actual_values)}, timestamps={len(timestamps)}"
                )

            predictions_final = predictions_final[-target_len:]
            actual_values = actual_values[-target_len:]
            timestamps = timestamps[-target_len:]
            time_hours = self._build_time_hours_from_axis(timestamps, target_len)

            # Count model parameters from metadata when available
            total_params = hyperparams.get('NUM_LEARNABLE_PARAMS', model_metadata.get('num_learnable_params', 'N/A'))
            task_info['model_parameters'] = int(total_params) if isinstance(total_params, (int, float)) else total_params
            
            # Create individual task result directory using session timestamp: new_test_result_{session_timestamp}
            test_result_dir = os.path.join(task_path, f'new_test_result_{self.session_timestamp}')
            os.makedirs(test_result_dir, exist_ok=True)
            
            test_file_name = os.path.splitext(os.path.basename(self.test_data_path))[0]
            
            # Determine target column display name and error units (match main loop logic)
            if "voltage" in target_column.lower():
                target_display = "Voltage"
                error_unit = "mV"
                error_multiplier = 1000.0
            elif "soc" in target_column.lower():
                target_display = "SOC"
                error_unit = "% SOC"
                error_multiplier = 100.0
            elif "temperature" in target_column.lower() or "temp" in target_column.lower():
                target_display = "Temperature"
                error_unit = "°C"
                error_multiplier = 1.0
            else:
                target_display = target_column.title()
                error_unit = "units"
                error_multiplier = 1.0
            
            # Continuous service already handles warmup and returns aligned prediction/target pairs.
            predicted_values = predictions_final

            # Calculate error and finalize DataFrame for CSV
            if actual_values is not None and predicted_values is not None:
                actual_values_for_metrics = actual_values
                predicted_values_for_metrics = predicted_values
                errors_raw = predicted_values - actual_values
                errors_display = errors_raw * error_multiplier
                
                # Create a clean dataframe with only essential data for predictions CSV
                final_df = pd.DataFrame({
                    'Timestamp': timestamps,
                    'Time (h)': time_hours,
                    f'True_{target_display}': actual_values,
                    f'Predicted_{target_display}': predicted_values,
                    f'Error ({error_unit})': errors_display
                })
            else:
                actual_values_for_metrics = None # Ensure this is defined
                # Create minimal dataframe even if predictions failed
                final_df = pd.DataFrame({'Error': ['No predictions generated']})
            
            # Save predictions file
            predictions_file = os.path.join(test_result_dir, f"{test_file_name}_predictions.csv")
            final_df.to_csv(predictions_file, index=False)
            self.progress.emit(f"  ✓ Predictions saved: {predictions_file}")
            
            # Calculate metrics and emit results for GUI
            # Use the NaN-filtered arrays for metric calculation
            mae, mse, rmse, r2, max_error = ('N/A', 'N/A', 'N/A', 'N/A', 'N/A')
            try:
                if actual_values_for_metrics is not None and len(actual_values_for_metrics) > 0:
                    mae = np.mean(np.abs(predicted_values_for_metrics - actual_values_for_metrics))
                    mse = np.mean((predicted_values_for_metrics - actual_values_for_metrics) ** 2)
                    rmse = np.sqrt(mse)
                    r2 = 1 - (np.sum((actual_values_for_metrics - predicted_values_for_metrics) ** 2) / np.sum((actual_values_for_metrics - np.mean(actual_values_for_metrics)) ** 2))
                    max_error = np.max(np.abs(predicted_values_for_metrics - actual_values_for_metrics))
            except Exception as e:
                self.progress.emit(f"  Warning: Could not calculate metrics: {e}")

            training_metrics = self._extract_training_metrics(task_path, task_info)
            num_params = task_info.get('model_parameters', hyperparams.get('NUM_LEARNABLE_PARAMS', 'N/A'))
            
            results_data = {
                'MAE': mae, 'RMSE': rmse, 'R²': r2, 'max_error': max_error,
                'predictions': predicted_values, # Send the full array with NaNs for plotting
                'actual_values': actual_values,   # Send the full actuals array for plotting
                'model_type': model_type, 'architecture': arch_name, 'task': task_name,
                'target_column': target_column, 'target_display': target_display, 'error_unit': error_unit,
                'model_file_path': model_file, 'test_data_file': self.test_data_path,
                'task_info': task_info, 'predictions_file': predictions_file,
                'inference_time': inference_time, 'training_info': training_metrics,
                'num_params': int(num_params) if isinstance(num_params, (int, float)) else 'N/A'
            }
            self.results_ready.emit(results_data)
            self.progress.emit(f"  ✓ Results: MAE={mae if isinstance(mae, str) else f'{mae:.4f}'}, RMSE={rmse if isinstance(rmse, str) else f'{rmse:.4f}'}, R²={r2 if isinstance(r2, str) else f'{r2:.4f}'}")
            
            # Display training metrics
            training_metrics = self._extract_training_metrics(task_path, task_info)
            self.progress.emit(f"  ✓ Training Info - Epochs: {training_metrics.get('epochs_trained', 'N/A')}, Best Train Loss: {training_metrics.get('best_train_loss', 'N/A')}, Best Val Loss: {training_metrics.get('best_val_loss', 'N/A')}")
            self.progress.emit(f"  ✓ Results saved to: {test_result_dir}")
            self.progress.emit(f"  ✓ Processing time: {time.time() - start_time:.2f}s")
            
            return True
            
        except Exception as e:
            self.progress.emit(f"  ✗ Error: {e}")
            return False
        finally:
            # Aggressively clean up memory after each model test
            del model
            if 'predictions_final' in locals(): del predictions_final
            if 'file_results' in locals(): del file_results
            if 'continuous_testing_service' in locals(): del continuous_testing_service
            if 'torch' in sys.modules and torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            task_name = task_info.get('task_name', 'Unknown')
            self.progress.emit(f"  ✓ Cleaned up memory for task {task_name}")

    def _apply_inference_filter_override(self, hyperparams: dict):
        """Apply optional GUI-selected inference filter override to task hyperparameters."""
        if not isinstance(hyperparams, dict):
            return

        override = self.inference_filter_override if isinstance(self.inference_filter_override, dict) else {}
        filter_type = str(override.get('INFERENCE_FILTER_TYPE', '')).strip()
        if not filter_type:
            return

        hyperparams['INFERENCE_FILTER_TYPE'] = filter_type

        if filter_type == 'Moving Average':
            hyperparams['INFERENCE_FILTER_WINDOW_SIZE'] = int(override.get('INFERENCE_FILTER_WINDOW_SIZE', 5))
            hyperparams.pop('INFERENCE_FILTER_ALPHA', None)
            hyperparams.pop('INFERENCE_FILTER_POLYORDER', None)
        elif filter_type == 'Exponential Moving Average':
            hyperparams['INFERENCE_FILTER_ALPHA'] = float(override.get('INFERENCE_FILTER_ALPHA', 0.2))
            hyperparams.pop('INFERENCE_FILTER_WINDOW_SIZE', None)
            hyperparams.pop('INFERENCE_FILTER_POLYORDER', None)
        elif filter_type == 'Savitzky-Golay':
            hyperparams['INFERENCE_FILTER_WINDOW_SIZE'] = int(override.get('INFERENCE_FILTER_WINDOW_SIZE', 9))
            hyperparams['INFERENCE_FILTER_POLYORDER'] = int(override.get('INFERENCE_FILTER_POLYORDER', 2))
            hyperparams.pop('INFERENCE_FILTER_ALPHA', None)
        elif filter_type == 'Median + Savitzky-Golay':
            hyperparams['INFERENCE_FILTER_WINDOW_SIZE'] = int(override.get('INFERENCE_FILTER_WINDOW_SIZE', 9))
            hyperparams['INFERENCE_FILTER_POLYORDER'] = int(override.get('INFERENCE_FILTER_POLYORDER', 2))
            hyperparams.pop('INFERENCE_FILTER_ALPHA', None)
        elif filter_type == 'Median + Butterworth (zero-phase)':
            hyperparams['INFERENCE_FILTER_WINDOW_SIZE'] = int(override.get('INFERENCE_FILTER_WINDOW_SIZE', 9))
            hyperparams['INFERENCE_FILTER_ALPHA'] = float(override.get('INFERENCE_FILTER_ALPHA', 0.08))
            hyperparams.pop('INFERENCE_FILTER_POLYORDER', None)
        else:
            hyperparams['INFERENCE_FILTER_TYPE'] = 'None'
            hyperparams.pop('INFERENCE_FILTER_WINDOW_SIZE', None)
            hyperparams.pop('INFERENCE_FILTER_ALPHA', None)
            hyperparams.pop('INFERENCE_FILTER_POLYORDER', None)

    def _extract_training_metrics(self, task_path, task_info):
        """Extract training metrics from training logs and task results."""
        training_metrics = {
            'epochs_trained': 'N/A',
            'best_train_loss': 'N/A',
            'best_val_loss': 'N/A',
            'final_train_loss': 'N/A',
            'final_val_loss': 'N/A',
            'training_time': 'N/A',
            'early_stopped': False
        }
        
        try:
            # First check task_info results
            results = task_info.get('results', {})
            if results.get('completed', False):
                training_metrics['best_val_loss'] = results.get('best_val_loss', 'N/A')
                training_metrics['best_epoch'] = results.get('best_epoch', 'N/A') 
                training_metrics['training_time'] = results.get('training_time', 'N/A')
                training_metrics['early_stopped'] = results.get('early_stopped', False)
            
            # Try to read training progress CSV for more detailed metrics
            logs_dir = os.path.join(task_path, 'logs')
            training_csv = os.path.join(logs_dir, 'training_progress.csv')
            
            if os.path.exists(training_csv):
                try:
                    df = pd.read_csv(training_csv, comment='#', on_bad_lines='warn', engine='python')
                    if not df.empty:
                        training_metrics['epochs_trained'] = int(df['epoch'].max())
                        training_metrics['final_train_loss'] = float(df['train_loss_norm'].iloc[-1])
                        training_metrics['final_val_loss'] = float(df['val_loss_norm'].iloc[-1])
                        training_metrics['best_val_loss'] = float(df['best_val_loss_norm'].min())
                        
                        # Find best epoch
                        best_idx = df['best_val_loss_norm'].idxmin()
                        training_metrics['best_train_loss'] = float(df.loc[best_idx, 'train_loss_norm'])
                        
                except Exception as e:
                    self.progress.emit(f"    Warning: Could not parse training CSV: {e}")
            
            return training_metrics
                        
        except Exception as e:
            self.progress.emit(f"    Warning: Could not extract training metrics: {e}")
            return training_metrics
    
    def _create_sequences(self, data, lookback):
        X = []
        for i in range(len(data) - lookback + 1):
            X.append(data[i:(i + lookback)])
        return np.array(X)