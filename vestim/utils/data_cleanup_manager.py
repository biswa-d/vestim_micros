# ---------------------------------------------------------------------------------
# Author: GitHub Copilot (Assistant)
# Date: 2025-07-08
# Version: 1.0.0
# Description: Utility functions for data cleanup and file reference management
# ---------------------------------------------------------------------------------

import os
import shutil
import json
import logging
from datetime import datetime

class DataCleanupManager:
    """
    Manages cleanup of training data folders and maintains file references for traceability.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def save_file_references_and_cleanup(self, job_folder_path: str) -> bool:
        """
        Simply cleans up data folders to save storage space.
        File references should already be saved during augmentation phase.
        
        :param job_folder_path: Path to the job folder containing train_data, val_data, test_data
        :return: True if cleanup was successful, False otherwise
        """
        try:
            self.logger.info(f"Starting data cleanup for job folder: {job_folder_path}")
            
            # Check if data reference file exists (created during augmentation)
            reference_file = os.path.join(job_folder_path, 'data_files_reference.txt')
            if os.path.exists(reference_file):
                self.logger.info("Data file references already saved during augmentation")
            else:
                self.logger.warning("No data file references found - they should have been saved during augmentation")
            
            # Simply clean up data folders
            cleanup_success = self._cleanup_data_folders(job_folder_path)
            
            if cleanup_success:
                # Create simple cleanup summary
                self._create_simple_cleanup_summary(job_folder_path)
                self.logger.info("Data cleanup completed successfully")
                return True
            else:
                self.logger.error("Data cleanup failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error during data cleanup: {e}", exc_info=True)
            return False
    
    def _create_basic_file_references(self, job_folder_path: str) -> dict:
        """Create basic file references without reading CSV content (fallback method)."""
        file_references = {
            'cleanup_info': {
                'timestamp': datetime.now().isoformat(),
                'job_folder': job_folder_path,
                'cleanup_performed': True,
                'note': 'Basic file references - no detailed CSV analysis performed'
            },
            'training_files': {'original_files': [], 'processed_files': [], 'total_samples': 'Unknown', 'file_count': 0},
            'validation_files': {'original_files': [], 'processed_files': [], 'total_samples': 'Unknown', 'file_count': 0},
            'test_files': {'original_files': [], 'processed_files': [], 'total_samples': 'Unknown', 'file_count': 0}
        }
        
        # Collect basic file info without reading CSV content
        for data_type in ['train', 'val', 'test']:
            data_key = f'{data_type}ing_files' if data_type == 'train' else f'{data_type}idation_files' if data_type == 'val' else 'test_files'
            
            # Original files from raw_data folder
            raw_folder = os.path.join(job_folder_path, f'{data_type}_data', 'raw_data')
            if os.path.exists(raw_folder):
                for file_name in os.listdir(raw_folder):
                    file_path = os.path.join(raw_folder, file_name)
                    if os.path.isfile(file_path):
                        file_info = {
                            'filename': file_name,
                            'original_path': file_path,
                            'file_size_mb': round(os.path.getsize(file_path) / (1024 * 1024), 2)
                        }
                        file_references[data_key]['original_files'].append(file_info)
                        file_references[data_key]['file_count'] += 1
            
            # Processed files from processed_data folder
            processed_folder = os.path.join(job_folder_path, f'{data_type}_data', 'processed_data')
            if os.path.exists(processed_folder):
                for file_name in os.listdir(processed_folder):
                    file_path = os.path.join(processed_folder, file_name)
                    if os.path.isfile(file_path):
                        file_info = {
                            'filename': file_name,
                            'processed_path': file_path,
                            'file_size_mb': round(os.path.getsize(file_path) / (1024 * 1024), 2)
                        }
                        file_references[data_key]['processed_files'].append(file_info)
        
        return file_references

    def _create_file_references(self, job_folder_path: str) -> dict:
        """Create a dictionary of file references with comprehensive details from all data folders."""
        file_references = {
            'cleanup_info': {
                'timestamp': datetime.now().isoformat(),
                'job_folder': job_folder_path,
                'cleanup_performed': True
            },
            'training_files': {
                'original_files': [],
                'processed_files': [],
                'total_samples': 0,
                'file_count': 0
            },
            'validation_files': {
                'original_files': [],
                'processed_files': [],
                'total_samples': 0,
                'file_count': 0
            },
            'test_files': {
                'original_files': [],
                'processed_files': [],
                'total_samples': 0,
                'file_count': 0
            }
        }
        
        # Collect file references for each data type
        for data_type in ['train', 'val', 'test']:
            data_key = f'{data_type}ing_files' if data_type == 'train' else f'{data_type}idation_files' if data_type == 'val' else 'test_files'
            
            # Original files from raw_data folder
            raw_folder = os.path.join(job_folder_path, f'{data_type}_data', 'raw_data')
            if os.path.exists(raw_folder):
                for file_name in os.listdir(raw_folder):
                    file_path = os.path.join(raw_folder, file_name)
                    if os.path.isfile(file_path) and file_name.endswith('.csv'):
                        file_info = {
                            'filename': file_name,
                            'original_path': file_path,
                            'file_size_mb': round(os.path.getsize(file_path) / (1024 * 1024), 2)
                        }
                        
                        # Try to extract additional details from CSV file
                        try:
                            import pandas as pd
                            df = pd.read_csv(file_path, nrows=5)  # Read just first few rows to get info
                            file_info['sample_count'] = len(pd.read_csv(file_path))  # Get full count
                            file_info['columns'] = list(df.columns)
                            file_info['column_count'] = len(df.columns)
                            
                            # Extract temperature and voltage info if present
                            temp_cols = [col for col in df.columns if 'temp' in col.lower() or 'temperature' in col.lower()]
                            voltage_cols = [col for col in df.columns if 'voltage' in col.lower() or 'volt' in col.lower()]
                            current_cols = [col for col in df.columns if 'current' in col.lower()]
                            soc_cols = [col for col in df.columns if 'soc' in col.lower()]
                            
                            if temp_cols:
                                file_info['temperature_columns'] = temp_cols
                            if voltage_cols:
                                file_info['voltage_columns'] = voltage_cols
                            if current_cols:
                                file_info['current_columns'] = current_cols
                            if soc_cols:
                                file_info['soc_columns'] = soc_cols
                            
                            # Try to determine drive cycle info from filename or data patterns
                            filename_lower = file_name.lower()
                            if any(cycle in filename_lower for cycle in ['udds', 'us06', 'hway', 'city', 'highway']):
                                file_info['likely_drive_cycle'] = True
                                file_info['cycle_indicators'] = [cycle for cycle in ['udds', 'us06', 'hway', 'city', 'highway'] if cycle in filename_lower]
                            
                            # Update totals
                            file_references[data_key]['total_samples'] += file_info['sample_count']
                            file_references[data_key]['file_count'] += 1
                            
                        except Exception as csv_error:
                            # If CSV reading fails, just save basic info
                            file_info['sample_count'] = 'Could not determine'
                            file_info['csv_read_error'] = str(csv_error)
                        
                        file_references[data_key]['original_files'].append(file_info)
            
            # Processed files from processed_data folder
            processed_folder = os.path.join(job_folder_path, f'{data_type}_data', 'processed_data')
            if os.path.exists(processed_folder):
                for file_name in os.listdir(processed_folder):
                    file_path = os.path.join(processed_folder, file_name)
                    if os.path.isfile(file_path):
                        file_info = {
                            'filename': file_name,
                            'processed_path': file_path,
                            'file_size_mb': round(os.path.getsize(file_path) / (1024 * 1024), 2)
                        }
                        
                        # For processed files, try to get sample count if it's CSV
                        if file_name.endswith('.csv'):
                            try:
                                import pandas as pd
                                df = pd.read_csv(file_path)
                                file_info['sample_count'] = len(df)
                                file_info['columns'] = list(df.columns)
                                file_info['processing_stage'] = 'normalized' if 'normalized' in file_name else 'processed'
                            except Exception:
                                file_info['sample_count'] = 'Could not determine'
                        
                        file_references[data_key]['processed_files'].append(file_info)
        
        return file_references
    
    def _create_readable_file_lists(self, job_folder_path: str, file_references: dict):
        """Create human-readable text files listing the original data files with comprehensive details."""
        try:
            # Training files list
            train_files_txt = os.path.join(job_folder_path, 'training_files_used.txt')
            with open(train_files_txt, 'w') as f:
                f.write("TRAINING DATA FILES USED\n")
                f.write("=" * 50 + "\n")
                f.write(f"Job Folder: {job_folder_path}\n")
                f.write(f"Cleanup Date: {file_references['cleanup_info']['timestamp']}\n")
                f.write(f"Total Files: {file_references['training_files']['file_count']}\n")
                f.write(f"Total Samples: {file_references['training_files']['total_samples']}\n\n")
                
                f.write("Original Training Files:\n")
                f.write("-" * 30 + "\n")
                for file_info in file_references['training_files']['original_files']:
                    f.write(f"• {file_info['filename']} ({file_info['file_size_mb']} MB)\n")
                    f.write(f"  Path: {file_info['original_path']}\n")
                    if 'sample_count' in file_info:
                        f.write(f"  Samples: {file_info['sample_count']}\n")
                    if 'column_count' in file_info:
                        f.write(f"  Columns: {file_info['column_count']}\n")
                    if 'temperature_columns' in file_info:
                        f.write(f"  Temperature Columns: {', '.join(file_info['temperature_columns'])}\n")
                    if 'voltage_columns' in file_info:
                        f.write(f"  Voltage Columns: {', '.join(file_info['voltage_columns'])}\n")
                    if 'current_columns' in file_info:
                        f.write(f"  Current Columns: {', '.join(file_info['current_columns'])}\n")
                    if 'soc_columns' in file_info:
                        f.write(f"  SOC Columns: {', '.join(file_info['soc_columns'])}\n")
                    if 'cycle_indicators' in file_info:
                        f.write(f"  Drive Cycle Indicators: {', '.join(file_info['cycle_indicators'])}\n")
                    f.write("\n")
                
                f.write("Processed Training Files:\n")
                f.write("-" * 30 + "\n")
                for file_info in file_references['training_files']['processed_files']:
                    f.write(f"• {file_info['filename']} ({file_info['file_size_mb']} MB)\n")
                    if 'sample_count' in file_info:
                        f.write(f"  Samples: {file_info['sample_count']}\n")
                    if 'processing_stage' in file_info:
                        f.write(f"  Processing Stage: {file_info['processing_stage']}\n")
                    f.write("\n")
            
            # Validation files list
            val_files_txt = os.path.join(job_folder_path, 'validation_files_used.txt')
            with open(val_files_txt, 'w') as f:
                f.write("VALIDATION DATA FILES USED\n")
                f.write("=" * 50 + "\n")
                f.write(f"Job Folder: {job_folder_path}\n")
                f.write(f"Cleanup Date: {file_references['cleanup_info']['timestamp']}\n")
                f.write(f"Total Files: {file_references['validation_files']['file_count']}\n")
                f.write(f"Total Samples: {file_references['validation_files']['total_samples']}\n\n")
                
                f.write("Original Validation Files:\n")
                f.write("-" * 30 + "\n")
                for file_info in file_references['validation_files']['original_files']:
                    f.write(f"• {file_info['filename']} ({file_info['file_size_mb']} MB)\n")
                    f.write(f"  Path: {file_info['original_path']}\n")
                    if 'sample_count' in file_info:
                        f.write(f"  Samples: {file_info['sample_count']}\n")
                    if 'column_count' in file_info:
                        f.write(f"  Columns: {file_info['column_count']}\n")
                    if 'temperature_columns' in file_info:
                        f.write(f"  Temperature Columns: {', '.join(file_info['temperature_columns'])}\n")
                    if 'voltage_columns' in file_info:
                        f.write(f"  Voltage Columns: {', '.join(file_info['voltage_columns'])}\n")
                    if 'current_columns' in file_info:
                        f.write(f"  Current Columns: {', '.join(file_info['current_columns'])}\n")
                    if 'soc_columns' in file_info:
                        f.write(f"  SOC Columns: {', '.join(file_info['soc_columns'])}\n")
                    if 'cycle_indicators' in file_info:
                        f.write(f"  Drive Cycle Indicators: {', '.join(file_info['cycle_indicators'])}\n")
                    f.write("\n")
                
                f.write("Processed Validation Files:\n")
                f.write("-" * 30 + "\n")
                for file_info in file_references['validation_files']['processed_files']:
                    f.write(f"• {file_info['filename']} ({file_info['file_size_mb']} MB)\n")
                    if 'sample_count' in file_info:
                        f.write(f"  Samples: {file_info['sample_count']}\n")
                    if 'processing_stage' in file_info:
                        f.write(f"  Processing Stage: {file_info['processing_stage']}\n")
                    f.write("\n")
            
            # Test files list
            test_files_txt = os.path.join(job_folder_path, 'test_files_used.txt')
            with open(test_files_txt, 'w') as f:
                f.write("TEST DATA FILES USED\n")
                f.write("=" * 50 + "\n")
                f.write(f"Job Folder: {job_folder_path}\n")
                f.write(f"Cleanup Date: {file_references['cleanup_info']['timestamp']}\n")
                f.write(f"Total Files: {file_references['test_files']['file_count']}\n")
                f.write(f"Total Samples: {file_references['test_files']['total_samples']}\n\n")
                
                f.write("Original Test Files:\n")
                f.write("-" * 30 + "\n")
                for file_info in file_references['test_files']['original_files']:
                    f.write(f"• {file_info['filename']} ({file_info['file_size_mb']} MB)\n")
                    f.write(f"  Path: {file_info['original_path']}\n")
                    if 'sample_count' in file_info:
                        f.write(f"  Samples: {file_info['sample_count']}\n")
                    if 'column_count' in file_info:
                        f.write(f"  Columns: {file_info['column_count']}\n")
                    if 'temperature_columns' in file_info:
                        f.write(f"  Temperature Columns: {', '.join(file_info['temperature_columns'])}\n")
                    if 'voltage_columns' in file_info:
                        f.write(f"  Voltage Columns: {', '.join(file_info['voltage_columns'])}\n")
                    if 'current_columns' in file_info:
                        f.write(f"  Current Columns: {', '.join(file_info['current_columns'])}\n")
                    if 'soc_columns' in file_info:
                        f.write(f"  SOC Columns: {', '.join(file_info['soc_columns'])}\n")
                    if 'cycle_indicators' in file_info:
                        f.write(f"  Drive Cycle Indicators: {', '.join(file_info['cycle_indicators'])}\n")
                    f.write("\n")
                
                f.write("Processed Test Files:\n")
                f.write("-" * 30 + "\n")
                for file_info in file_references['test_files']['processed_files']:
                    f.write(f"• {file_info['filename']} ({file_info['file_size_mb']} MB)\n")
                    if 'sample_count' in file_info:
                        f.write(f"  Samples: {file_info['sample_count']}\n")
                    if 'processing_stage' in file_info:
                        f.write(f"  Processing Stage: {file_info['processing_stage']}\n")
                    f.write("\n")
            
            self.logger.info("Created readable file lists with comprehensive details")
            
        except Exception as e:
            self.logger.error(f"Error creating readable file lists: {e}")
    
    def _create_readable_file_lists_from_details(self, job_folder_path: str, file_references: dict):
        """Create human-readable text files from pre-saved data details."""
        try:
            # Training files list
            train_files_txt = os.path.join(job_folder_path, 'training_files_used.txt')
            with open(train_files_txt, 'w') as f:
                f.write("TRAINING DATA FILES USED\n")
                f.write("=" * 50 + "\n")
                f.write(f"Job Folder: {job_folder_path}\n")
                f.write(f"Created: {file_references.get('timestamp', 'Unknown')}\n")
                
                training_info = file_references.get('training_files', {})
                f.write(f"Total Files: {training_info.get('file_count', 0)}\n")
                f.write(f"Total Samples: {training_info.get('total_samples', 0):,}\n")
                f.write(f"Total Size: {training_info.get('total_size_mb', 0):.2f} MB\n\n")
                
                f.write("File Details:\n")
                f.write("-" * 30 + "\n")
                for file_info in training_info.get('files', []):
                    f.write(f"• {file_info['filename']}\n")
                    f.write(f"  Samples: {file_info.get('sample_count', 'Unknown'):,}\n")
                    f.write(f"  Columns: {file_info.get('column_count', 'Unknown')}\n")
                    if 'temperature_columns' in file_info:
                        f.write(f"  Temperature: {', '.join(file_info['temperature_columns'])}\n")
                    if 'voltage_columns' in file_info:
                        f.write(f"  Voltage: {', '.join(file_info['voltage_columns'])}\n")
                    if 'current_columns' in file_info:
                        f.write(f"  Current: {', '.join(file_info['current_columns'])}\n")
                    if 'soc_columns' in file_info:
                        f.write(f"  SOC: {', '.join(file_info['soc_columns'])}\n")
                    if 'cycle_indicators' in file_info:
                        f.write(f"  Drive Cycles: {', '.join(file_info['cycle_indicators'])}\n")
                    f.write("\n")
            
            # Validation files list
            val_files_txt = os.path.join(job_folder_path, 'validation_files_used.txt')
            with open(val_files_txt, 'w') as f:
                f.write("VALIDATION DATA FILES USED\n")
                f.write("=" * 50 + "\n")
                f.write(f"Job Folder: {job_folder_path}\n")
                f.write(f"Created: {file_references.get('timestamp', 'Unknown')}\n")
                
                validation_info = file_references.get('validation_files', {})
                f.write(f"Total Files: {validation_info.get('file_count', 0)}\n")
                f.write(f"Total Samples: {validation_info.get('total_samples', 0):,}\n")
                f.write(f"Total Size: {validation_info.get('total_size_mb', 0):.2f} MB\n\n")
                
                f.write("File Details:\n")
                f.write("-" * 30 + "\n")
                for file_info in validation_info.get('files', []):
                    f.write(f"• {file_info['filename']}\n")
                    f.write(f"  Samples: {file_info.get('sample_count', 'Unknown'):,}\n")
                    f.write(f"  Columns: {file_info.get('column_count', 'Unknown')}\n")
                    if 'temperature_columns' in file_info:
                        f.write(f"  Temperature: {', '.join(file_info['temperature_columns'])}\n")
                    if 'voltage_columns' in file_info:
                        f.write(f"  Voltage: {', '.join(file_info['voltage_columns'])}\n")
                    if 'current_columns' in file_info:
                        f.write(f"  Current: {', '.join(file_info['current_columns'])}\n")
                    if 'soc_columns' in file_info:
                        f.write(f"  SOC: {', '.join(file_info['soc_columns'])}\n")
                    if 'cycle_indicators' in file_info:
                        f.write(f"  Drive Cycles: {', '.join(file_info['cycle_indicators'])}\n")
                    f.write("\n")
            
            # Test files list
            test_files_txt = os.path.join(job_folder_path, 'test_files_used.txt')
            with open(test_files_txt, 'w') as f:
                f.write("TEST DATA FILES USED\n")
                f.write("=" * 50 + "\n")
                f.write(f"Job Folder: {job_folder_path}\n")
                f.write(f"Created: {file_references.get('timestamp', 'Unknown')}\n")
                
                test_info = file_references.get('test_files', {})
                f.write(f"Total Files: {test_info.get('file_count', 0)}\n")
                f.write(f"Total Samples: {test_info.get('total_samples', 0):,}\n")
                f.write(f"Total Size: {test_info.get('total_size_mb', 0):.2f} MB\n\n")
                
                f.write("File Details:\n")
                f.write("-" * 30 + "\n")
                for file_info in test_info.get('files', []):
                    f.write(f"• {file_info['filename']}\n")
                    f.write(f"  Samples: {file_info.get('sample_count', 'Unknown'):,}\n")
                    f.write(f"  Columns: {file_info.get('column_count', 'Unknown')}\n")
                    if 'temperature_columns' in file_info:
                        f.write(f"  Temperature: {', '.join(file_info['temperature_columns'])}\n")
                    if 'voltage_columns' in file_info:
                        f.write(f"  Voltage: {', '.join(file_info['voltage_columns'])}\n")
                    if 'current_columns' in file_info:
                        f.write(f"  Current: {', '.join(file_info['current_columns'])}\n")
                    if 'soc_columns' in file_info:
                        f.write(f"  SOC: {', '.join(file_info['soc_columns'])}\n")
                    if 'cycle_indicators' in file_info:
                        f.write(f"  Drive Cycles: {', '.join(file_info['cycle_indicators'])}\n")
                    f.write("\n")
            
            self.logger.info("Created readable file lists from pre-saved details")
            
        except Exception as e:
            self.logger.error(f"Error creating readable file lists from details: {e}")

    def _cleanup_data_folders(self, job_folder_path: str) -> bool:
        """Remove the raw_data and processed_data folders to save space (cross-platform)."""
        import stat
        import platform
        
        def handle_readonly(func, path, exc_info):
            """Handle read-only file errors on Windows."""
            if os.path.exists(path):
                os.chmod(path, stat.S_IWRITE)
                func(path)
        
        try:
            folders_to_remove = []
            space_freed_mb = 0
            system_platform = platform.system()
            
            # Collect all data folders
            for data_type in ['train_data', 'val_data', 'test_data']:
                data_folder = os.path.join(job_folder_path, data_type)
                if os.path.exists(data_folder):
                    # Calculate space to be freed
                    for root, dirs, files in os.walk(data_folder):
                        for file in files:
                            file_path = os.path.join(root, file)
                            try:
                                space_freed_mb += os.path.getsize(file_path) / (1024 * 1024)
                            except OSError:
                                pass  # File might be locked, skip size calculation
                    
                    folders_to_remove.append(data_folder)
            
            # Remove the folders with cross-platform error handling
            for folder in folders_to_remove:
                try:
                    # First try standard removal
                    shutil.rmtree(folder, onerror=handle_readonly)
                    self.logger.info(f"Removed data folder: {folder}")
                except Exception as e:
                    # If standard removal fails, try alternative method
                    self.logger.warning(f"Standard removal failed for {folder}: {e}")
                    try:
                        # Alternative: Remove files individually
                        for root, dirs, files in os.walk(folder, topdown=False):
                            for file in files:
                                file_path = os.path.join(root, file)
                                try:
                                    os.chmod(file_path, stat.S_IWRITE)
                                    os.remove(file_path)
                                except OSError:
                                    pass  # Skip files that can't be removed
                            for dir in dirs:
                                try:
                                    os.rmdir(os.path.join(root, dir))
                                except OSError:
                                    pass  # Skip directories that can't be removed
                        
                        # Try to remove the main folder
                        try:
                            os.rmdir(folder)
                            self.logger.info(f"Removed data folder (alternative method): {folder}")
                        except OSError:
                            # Last resort: Use OS-specific commands for stubborn folders
                            try:
                                import subprocess
                                if system_platform == "Windows":
                                    # Use rmdir command with force options
                                    result = subprocess.run(
                                        ["rmdir", "/s", "/q", folder], 
                                        capture_output=True, 
                                        text=True, 
                                        shell=True
                                    )
                                    if result.returncode == 0:
                                        self.logger.info(f"Removed data folder (Windows rmdir): {folder}")
                                    else:
                                        self.logger.warning(f"Could not fully remove {folder}: {result.stderr}")
                                elif system_platform in ["Linux", "Darwin"]:  # Linux or macOS
                                    # Use rm command with force options
                                    result = subprocess.run(
                                        ["rm", "-rf", folder], 
                                        capture_output=True, 
                                        text=True
                                    )
                                    if result.returncode == 0:
                                        self.logger.info(f"Removed data folder (Unix rm): {folder}")
                                    else:
                                        self.logger.warning(f"Could not fully remove {folder}: {result.stderr}")
                                else:
                                    self.logger.warning(f"Could not fully remove {folder}, but most files were deleted (unsupported platform: {system_platform})")
                            except Exception as cmd_error:
                                self.logger.warning(f"OS command failed for {folder}: {cmd_error}")
                    except Exception as e2:
                        self.logger.warning(f"Alternative removal also failed for {folder}: {e2}")
            
            self.logger.info(f"Data cleanup freed approximately {space_freed_mb:.2f} MB of storage space")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during data cleanup: {e}")
            return False
    
    def _create_cleanup_summary(self, job_folder_path: str, file_references: dict):
        """Create a summary file about the cleanup operation."""
        try:
            summary_file = os.path.join(job_folder_path, 'data_cleanup_summary.txt')
            
            # Calculate totals
            total_train_files = len(file_references['training_files']['original_files'])
            total_val_files = len(file_references['validation_files']['original_files'])
            total_test_files = len(file_references['test_files']['original_files'])
            
            total_size_mb = 0
            for data_type in ['training_files', 'validation_files', 'test_files']:
                for file_info in file_references[data_type]['original_files']:
                    total_size_mb += file_info['file_size_mb']
                for file_info in file_references[data_type]['processed_files']:
                    total_size_mb += file_info['file_size_mb']
            
            with open(summary_file, 'w') as f:
                f.write("DATA CLEANUP SUMMARY\n")
                f.write("=" * 50 + "\n")
                f.write(f"Job Folder: {job_folder_path}\n")
                f.write(f"Cleanup Date: {file_references['cleanup_info']['timestamp']}\n\n")
                
                f.write("FILES PROCESSED:\n")
                f.write(f"• Training files: {total_train_files}\n")
                f.write(f"• Validation files: {total_val_files}\n") 
                f.write(f"• Test files: {total_test_files}\n")
                f.write(f"• Total storage freed: ~{total_size_mb:.2f} MB\n\n")
                
                f.write("FOLDERS REMOVED:\n")
                f.write("• train_data/ (raw_data and processed_data)\n")
                f.write("• val_data/ (raw_data and processed_data)\n")
                f.write("• test_data/ (raw_data and processed_data)\n\n")
                
                f.write("FILE REFERENCES PRESERVED IN:\n")
                f.write("• data_file_references.json (detailed JSON)\n")
                f.write("• training_files_used.txt (human-readable)\n")
                f.write("• validation_files_used.txt (human-readable)\n")
                f.write("• test_files_used.txt (human-readable)\n\n")
                
                f.write("NOTE: Original data files remain in their source locations.\n")
                f.write("This cleanup only removes the copied/processed data to save space.\n")
            
            self.logger.info(f"Created cleanup summary: {summary_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating cleanup summary: {e}")
    
    def _create_simple_cleanup_summary(self, job_folder_path: str):
        """Create a simple summary file about the cleanup operation."""
        try:
            summary_file = os.path.join(job_folder_path, 'data_cleanup_summary.txt')
            
            with open(summary_file, 'w') as f:
                f.write("DATA CLEANUP SUMMARY\n")
                f.write("=" * 50 + "\n")
                f.write(f"Job Folder: {job_folder_path}\n")
                f.write(f"Cleanup Date: {datetime.now().isoformat()}\n\n")
                
                f.write("FOLDERS REMOVED TO SAVE SPACE:\n")
                f.write("• train_data/ (raw_data and processed_data)\n")
                f.write("• val_data/ (raw_data and processed_data)\n")
                f.write("• test_data/ (raw_data and processed_data)\n\n")
                
                f.write("PRESERVED FILES:\n")
                f.write("• All models, predictions, and results\n")
                f.write("• data_files_reference.txt (file names and sample counts)\n")
                f.write("• Scaler files with min/max statistics (in scalers/ folder)\n")
                f.write("• All plots and training histories\n\n")
                
                f.write("NOTE:\n")
                f.write("• Original data files remain in their source locations\n")
                f.write("• This cleanup only removes copied/processed data to save space\n")
                f.write("• Full traceability is maintained through reference files\n")
            
            self.logger.info(f"Created cleanup summary: {summary_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating cleanup summary: {e}")
