import os
import shutil
import json
from datetime import datetime
from vestim.config import OUTPUT_DIR
from vestim.logger_config import configure_job_specific_logging # Import the new function
import logging # Import logging to potentially log the action

class JobManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(JobManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'job_id'):  # Ensure the attributes are initialized once
            self.job_id = None
            self._train_folder_path = None
            self._val_folder_path = None
            self._test_folder_path = None

    def set_data_paths(self, train_path=None, val_path=None, test_path=None):
        """Set the data paths for the current job."""
        if train_path:
            self._train_folder_path = train_path
        if val_path:
            self._val_folder_path = val_path
        if test_path:
            self._test_folder_path = test_path

    def create_new_job(self):
        """Generates a new job ID based on the current timestamp and initializes job directories."""
        self.job_id = f"job_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        job_folder = os.path.join(OUTPUT_DIR, self.job_id)
        os.makedirs(job_folder, exist_ok=True)
        
        # Configure logging to use a file within this new job folder
        try:
            configure_job_specific_logging(job_folder)
            logging.info(f"Job-specific logging configured for job: {self.job_id} in folder: {job_folder}")
        except Exception as e:
            logging.error(f"Failed to configure job-specific logging for {self.job_id}: {e}", exc_info=True)
            # Continue without job-specific logging if setup fails, default logging will be used.

        # Save job metadata, including original data paths
        metadata = {
            'job_id': self.job_id,
            'creation_date': datetime.now().isoformat(),
            'train_folder_path': self._train_folder_path,
            'val_folder_path': self._val_folder_path,
            'test_folder_path': self._test_folder_path
        }
        metadata_path = os.path.join(job_folder, 'job_metadata.json')
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            logging.info(f"Job metadata saved to {metadata_path}")
        except Exception as e:
            logging.error(f"Failed to save job metadata: {e}", exc_info=True)

        return self.job_id, job_folder

    def get_job_id(self):
        """Returns the current job ID."""
        return self.job_id

    def get_job_folder(self):
        """Returns the path to the current job folder."""
        if self.job_id:
            return os.path.join(OUTPUT_DIR, self.job_id)
        return None
    
    def get_train_folder(self):
        """Returns the path to the train processed data folder."""
        if self.job_id:
            return os.path.join(self.get_job_folder(), 'train_data', 'processed_data')
        return None

    def get_val_folder(self):
        """Returns the path to the validation processed data folder."""
        if self.job_id:
            return os.path.join(self.get_job_folder(), 'val_data', 'processed_data')
        return None

    def get_test_folder(self):
        """Returns the path to the test processed data folder."""
        if self.job_id:
            return os.path.join(self.get_job_folder(), 'test_data', 'processed_data')
        return None
    
    #Folder where test data will be stored
    def get_test_results_folder(self):
        """
        Returns the path to the test results folder.
        :return: Path to the test results folder within the job directory.
        """
        if self.job_id:
            results_folder = os.path.join(self.get_job_folder(), 'test', 'results')
            os.makedirs(results_folder, exist_ok=True)
            return results_folder
        return None

    def get_train_folder_path(self):
        return getattr(self, '_train_folder_path', '')

    def get_val_folder_path(self):
        return getattr(self, '_val_folder_path', '')

    def get_test_folder_path(self):
        return getattr(self, '_test_folder_path', '')

    def cleanup_job_data(self, job_folder=None):
        """
        Removes raw and processed data folders to save space, preserving models and results.
        Logs the action and the original data source paths to a summary file.
        """
        if job_folder is None:
            job_folder = self.get_job_folder()
        
        if not job_folder or not os.path.isdir(job_folder):
            logging.error(f"Invalid job folder for cleanup: {job_folder}")
            return

        # Read original data paths from job_metadata.json
        metadata_path = os.path.join(job_folder, 'job_metadata.json')
        original_train_path = "Unknown"
        original_val_path = "Unknown"
        original_test_path = "Unknown"
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                original_train_path = metadata.get('train_folder_path', 'Not specified')
                original_val_path = metadata.get('val_folder_path', 'Not specified')
                original_test_path = metadata.get('test_folder_path', 'Not specified')
        except Exception as e:
            logging.warning(f"Could not read job metadata to get original data paths: {e}")

        # Paths to remove
        paths_to_remove = [
            os.path.join(job_folder, 'train_data'),
            os.path.join(job_folder, 'val_data'),
            os.path.join(job_folder, 'test_data')
        ]

        summary_content = [
            "DATA CLEANUP SUMMARY",
            "=" * 50,
            f"Job Folder: {job_folder}",
            f"Cleanup Date: {datetime.now().isoformat()}\n",
            "ORIGINAL DATA SOURCES:",
            f"• Training: {original_train_path}",
            f"• Validation: {original_val_path}",
            f"• Testing: {original_test_path}\n",
            "FOLDERS REMOVED TO SAVE SPACE:"
        ]

        for path in paths_to_remove:
            if os.path.isdir(path):
                try:
                    shutil.rmtree(path)
                    logging.info(f"Removed data folder: {path}")
                    summary_content.append(f"• {os.path.relpath(path, job_folder)}/")
                except Exception as e:
                    logging.error(f"Failed to remove {path}: {e}")
                    summary_content.append(f"• FAILED to remove {os.path.relpath(path, job_folder)}/")
        
        summary_content.extend([
            "\nPRESERVED FILES:",
            "• All models, predictions, and results",
            "• data_files_reference.txt (file names and sample counts)",
            "• Scaler files with min/max statistics (in scalers/ folder)",
            "• All plots and training histories\n",
            "NOTE:",
            "• Original data files remain in their source locations",
            "• This cleanup only removes copied/processed data to save space",
            "• Full traceability is maintained through reference files"
        ])

        # Save the summary
        summary_path = os.path.join(job_folder, 'cleanup_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("\n".join(summary_content))
        
        logging.info(f"Data cleanup summary saved to {summary_path}")
    
    def load_existing_job(self, job_folder_path):
        """Load an existing job from a job folder path."""
        if not os.path.isdir(job_folder_path):
            raise ValueError(f"Job folder does not exist: {job_folder_path}")
        
        # Extract job_id from folder name
        self.job_id = os.path.basename(job_folder_path)
        
        # Verify this is a valid job folder by checking for essential files
        essential_files = ['hyperparams.json', 'job_metadata.json']
        for file in essential_files:
            if not os.path.exists(os.path.join(job_folder_path, file)):
                raise ValueError(f"Invalid job folder: missing {file}")
        
        # Configure logging for this job
        try:
            configure_job_specific_logging(job_folder_path)
            logging.info(f"Loaded existing job: {self.job_id} from folder: {job_folder_path}")
        except Exception as e:
            logging.error(f"Failed to configure logging for loaded job {self.job_id}: {e}", exc_info=True)
        
        return self.job_id, job_folder_path
