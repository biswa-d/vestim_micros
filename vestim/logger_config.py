import logging
from logging.handlers import RotatingFileHandler
import sys

def setup_logger(log_file='default.log'):
    logger = logging.getLogger()
    # If the root logger already has handlers, assume it's configured and return it.
    # This prevents adding duplicate handlers if setup_logger is called multiple times.
    if logger.hasHandlers():
        # Optionally, you could check if the level needs to be reset or if specific handlers are present,
        # but for now, just preventing duplicates is the main goal.
        # You might also want to ensure the level is at least INFO if it was set lower by another call.
        if logger.level > logging.INFO or logger.level == 0: # level 0 means NOTSET
             logger.setLevel(logging.INFO)
        return logger

    logger.setLevel(logging.INFO)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

    # Rotating File Handler (5 MB max, keep 3 backups)
    file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')) # Corrected back to %(message)s
    logger.addHandler(file_handler)

    return logger

def configure_job_specific_logging(job_folder_path, log_file_name='job.log'):
    """
    Reconfigures the root logger to use a job-specific log file.
    Removes existing file handlers and adds a new one for the job.
    """
    logger = logging.getLogger()
    
    # Remove existing file handlers to avoid duplicate logging or logging to old files
    for handler in logger.handlers[:]: # Iterate over a copy
        if isinstance(handler, logging.FileHandler): # Catches RotatingFileHandler too
            logger.removeHandler(handler)
            handler.close() # Important to close the file
            
    # Define the new job-specific log file path
    job_log_file = os.path.join(job_folder_path, log_file_name)
    
    # Create and add the new job-specific file handler
    # Ensure the directory for the job log file exists
    os.makedirs(os.path.dirname(job_log_file), exist_ok=True)
        
    job_file_handler = RotatingFileHandler(job_log_file, maxBytes=5*1024*1024, backupCount=3)
    job_file_handler.setLevel(logging.DEBUG) # Or INFO, as per requirements
    job_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')) # Added %(name)s
    logger.addHandler(job_file_handler)
    
    # Ensure console handler is still present if it was there or add it if not
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)

    # Ensure logger level is set (it might have been reset if all handlers were removed)
    if logger.level > logging.INFO or logger.level == 0:
        logger.setLevel(logging.INFO)
        
    logger.info(f"Logging reconfigured. Now logging to: {job_log_file}")
    return logger
