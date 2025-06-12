import os
import logging
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
        """
        self.logger.info(f"Starting data processing for job {job_id} with source {data_source}")
        
        processor = self.processors.get(data_source)
        if not processor:
            raise ValueError(f"No data processor found for source: {data_source}")

        try:
            job_folder = processor.organize_and_convert_files(
                train_files,
                test_files,
                job_id=job_id
            )
            self.logger.info(f"Data processing complete for job {job_id}. Output folder: {job_folder}")
            return job_folder
        except Exception as e:
            self.logger.error(f"Error during data processing for job {job_id}: {e}", exc_info=True)
            raise