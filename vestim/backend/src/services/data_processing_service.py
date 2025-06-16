import os
import logging
import traceback
from datetime import datetime
import json

from vestim.backend.src.services.data_processor.src.data_processor_qt_arbin import DataProcessorArbin
from vestim.backend.src.services.data_processor.src.data_processor_qt_stla import DataProcessorSTLA
from vestim.backend.src.services.data_processor.src.data_processor_qt_digatron import DataProcessorDigatron

class DataProcessingService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.processors = {
            "Arbin": DataProcessorArbin(),
            "STLA": DataProcessorSTLA(),
            "Digatron": DataProcessorDigatron(),
        }

    def process_data(self, job_id: str, train_files: list, test_files: list, data_source: str):
        """
        Process and convert data for a given job.
        
        Args:
            job_id (str): The ID of the job
            train_files (list): List of paths to training data files
            test_files (list): List of paths to testing data files
            data_source (str): The data source type (e.g., 'Arbin', 'STLA', 'Digatron')
        
        Returns:
            tuple: (success, job_folder) where success is a boolean and job_folder is the path to the job folder
        """
        self.logger.info(f"Starting data processing for job {job_id} with source {data_source}")
        
        processor = self.processors.get(data_source)
        if not processor:
            self.logger.error(f"No data processor found for source: {data_source}")
            return False, None

        try:
            job_folder = processor.organize_and_convert_files(
                train_files,
                test_files,
                job_id=job_id
            )
            
            self.logger.info(f"Data processing completed for job {job_id}. Job folder: {job_folder}")
            
            # Save metadata
            try:
                metadata = {
                    "data_source": data_source,
                    "processed_at": datetime.now().isoformat(),
                    "train_files": [os.path.basename(f) for f in train_files],
                    "test_files": [os.path.basename(f) for f in test_files],
                }
                
                metadata_path = os.path.join(job_folder, "data_metadata.json")
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=4)
            except Exception as meta_error:
                self.logger.warning(f"Error saving metadata for job {job_id}: {meta_error}")
                # Continue despite metadata error
            
            return True, job_folder
            
        except Exception as e:
            self.logger.error(f"Error processing data for job {job_id}: {e}")
            self.logger.error(traceback.format_exc())
            return False, None
    
    def process_data_files(self, job_id, job_folder, train_files, test_files, data_source):
        """
        Process data files for a job. This is a wrapper around process_data for the new API.
        
        Args:
            job_id (str): The ID of the job
            job_folder (str): The path to the job folder (ignored, we use the one from the processor)
            train_files (list): List of paths to training data files
            test_files (list): List of paths to testing data files
            data_source (str): The data source type (e.g., 'Arbin', 'STLA', 'Digatron')
            
        Returns:
            bool: True if processing was successful, False otherwise
        """
        success, _ = self.process_data(job_id, train_files, test_files, data_source)
        return success