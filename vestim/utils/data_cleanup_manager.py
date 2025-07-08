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
        Saves file references to text files and cleans up raw/processed data folders
        to save storage space while maintaining traceability.
        
        :param job_folder_path: Path to the job folder containing train_data, val_data, test_data
        :return: True if cleanup was successful, False otherwise
        """
        try:
            self.logger.info(f"Starting data cleanup for job folder: {job_folder_path}")
            
            # Create file references
            file_references = self._create_file_references(job_folder_path)
            
            if not file_references:
                self.logger.warning("No file references found. Skipping cleanup.")
                return False
            
            # Save file references to JSON file
            references_file = os.path.join(job_folder_path, 'data_file_references.json')
            with open(references_file, 'w') as f:
                json.dump(file_references, f, indent=4)
            
            self.logger.info(f"Saved file references to: {references_file}")
            
            # Create human-readable text files
            self._create_readable_file_lists(job_folder_path, file_references)
            
            # Clean up data folders
            cleanup_success = self._cleanup_data_folders(job_folder_path)
            
            if cleanup_success:
                # Create cleanup summary
                self._create_cleanup_summary(job_folder_path, file_references)
                self.logger.info("Data cleanup completed successfully")
                return True
            else:
                self.logger.error("Data cleanup failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error during data cleanup: {e}", exc_info=True)
            return False
    
    def _create_file_references(self, job_folder_path: str) -> dict:
        """Create a dictionary of file references from all data folders."""
        file_references = {
            'cleanup_info': {
                'timestamp': datetime.now().isoformat(),
                'job_folder': job_folder_path,
                'cleanup_performed': True
            },
            'training_files': {
                'original_files': [],
                'processed_files': []
            },
            'validation_files': {
                'original_files': [],
                'processed_files': []
            },
            'test_files': {
                'original_files': [],
                'processed_files': []
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
                    if os.path.isfile(file_path):
                        file_info = {
                            'filename': file_name,
                            'original_path': file_path,
                            'file_size_mb': round(os.path.getsize(file_path) / (1024 * 1024), 2)
                        }
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
                        file_references[data_key]['processed_files'].append(file_info)
        
        return file_references
    
    def _create_readable_file_lists(self, job_folder_path: str, file_references: dict):
        """Create human-readable text files listing the original data files."""
        try:
            # Training files list
            train_files_txt = os.path.join(job_folder_path, 'training_files_used.txt')
            with open(train_files_txt, 'w') as f:
                f.write("TRAINING DATA FILES USED\n")
                f.write("=" * 50 + "\n")
                f.write(f"Job Folder: {job_folder_path}\n")
                f.write(f"Cleanup Date: {file_references['cleanup_info']['timestamp']}\n\n")
                
                f.write("Original Training Files:\n")
                f.write("-" * 30 + "\n")
                for file_info in file_references['training_files']['original_files']:
                    f.write(f"• {file_info['filename']} ({file_info['file_size_mb']} MB)\n")
                    f.write(f"  Path: {file_info['original_path']}\n\n")
                
                f.write("Processed Training Files:\n")
                f.write("-" * 30 + "\n")
                for file_info in file_references['training_files']['processed_files']:
                    f.write(f"• {file_info['filename']} ({file_info['file_size_mb']} MB)\n\n")
            
            # Validation files list
            val_files_txt = os.path.join(job_folder_path, 'validation_files_used.txt')
            with open(val_files_txt, 'w') as f:
                f.write("VALIDATION DATA FILES USED\n")
                f.write("=" * 50 + "\n")
                f.write(f"Job Folder: {job_folder_path}\n")
                f.write(f"Cleanup Date: {file_references['cleanup_info']['timestamp']}\n\n")
                
                f.write("Original Validation Files:\n")
                f.write("-" * 30 + "\n")
                for file_info in file_references['validation_files']['original_files']:
                    f.write(f"• {file_info['filename']} ({file_info['file_size_mb']} MB)\n")
                    f.write(f"  Path: {file_info['original_path']}\n\n")
                
                f.write("Processed Validation Files:\n")
                f.write("-" * 30 + "\n")
                for file_info in file_references['validation_files']['processed_files']:
                    f.write(f"• {file_info['filename']} ({file_info['file_size_mb']} MB)\n\n")
            
            # Test files list
            test_files_txt = os.path.join(job_folder_path, 'test_files_used.txt')
            with open(test_files_txt, 'w') as f:
                f.write("TEST DATA FILES USED\n")
                f.write("=" * 50 + "\n")
                f.write(f"Job Folder: {job_folder_path}\n")
                f.write(f"Cleanup Date: {file_references['cleanup_info']['timestamp']}\n\n")
                
                f.write("Original Test Files:\n")
                f.write("-" * 30 + "\n")
                for file_info in file_references['test_files']['original_files']:
                    f.write(f"• {file_info['filename']} ({file_info['file_size_mb']} MB)\n")
                    f.write(f"  Path: {file_info['original_path']}\n\n")
                
                f.write("Processed Test Files:\n")
                f.write("-" * 30 + "\n")
                for file_info in file_references['test_files']['processed_files']:
                    f.write(f"• {file_info['filename']} ({file_info['file_size_mb']} MB)\n\n")
            
            self.logger.info("Created readable file lists")
            
        except Exception as e:
            self.logger.error(f"Error creating readable file lists: {e}")
    
    def _cleanup_data_folders(self, job_folder_path: str) -> bool:
        """Remove the raw_data and processed_data folders to save space."""
        try:
            folders_to_remove = []
            space_freed_mb = 0
            
            # Collect all data folders
            for data_type in ['train_data', 'val_data', 'test_data']:
                data_folder = os.path.join(job_folder_path, data_type)
                if os.path.exists(data_folder):
                    # Calculate space to be freed
                    for root, dirs, files in os.walk(data_folder):
                        for file in files:
                            file_path = os.path.join(root, file)
                            space_freed_mb += os.path.getsize(file_path) / (1024 * 1024)
                    
                    folders_to_remove.append(data_folder)
            
            # Remove the folders
            for folder in folders_to_remove:
                shutil.rmtree(folder)
                self.logger.info(f"Removed data folder: {folder}")
            
            self.logger.info(f"Data cleanup freed {space_freed_mb:.2f} MB of storage space")
            return True
            
        except Exception as e:
            self.logger.error(f"Error removing data folders: {e}")
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
