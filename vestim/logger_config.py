import logging
from logging.handlers import RotatingFileHandler
import sys
import os # Import the os module
import platform

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

    logger.setLevel(logging.INFO) # Set root logger level first

    # Set higher logging level for matplotlib to reduce verbosity
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

    # Rotating File Handler (5 MB max, keep 3 backups)
    # On Windows, RotatingFileHandler can fail when trying to rotate due to file locks
    # Use a custom handler that delays rotation to avoid PermissionError
    if platform.system() == 'Windows':
        # Use delayed=True to avoid file locking issues during rotation on Windows
        file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3, delay=True)
    else:
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
    
    # Forcefully remove ALL existing handlers from the root logger
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    
    # Reset root logger level (it might be set to a high level if all handlers were removed)
    logger.setLevel(logging.DEBUG) # Set to DEBUG to capture everything for the file

    # Define the new job-specific log file path
    job_log_file = os.path.join(job_folder_path, log_file_name)
    
    # Create and add the new job-specific file handler
    # Ensure the directory for the job log file exists
    os.makedirs(os.path.dirname(job_log_file), exist_ok=True)
        
    # On Windows, use delayed=True to avoid file locking issues during rotation
    if platform.system() == 'Windows':
        job_file_handler = RotatingFileHandler(job_log_file, maxBytes=5*1024*1024, backupCount=3, delay=True)
    else:
        job_file_handler = RotatingFileHandler(job_log_file, maxBytes=5*1024*1024, backupCount=3)
    job_file_handler.setLevel(logging.DEBUG) # Or INFO, as per requirements
    job_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')) # Moved %(name)s
    logger.addHandler(job_file_handler)
    
    # Add a new console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO) # Console shows INFO and above
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

    # Explicitly set matplotlib's logger level after our handlers are set up
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
        
    logger.info(f"Logging reconfigured. Root logger level: {logger.level}. File handler level: {job_file_handler.level}. Console handler level: {console_handler.level}. Matplotlib logger level: {logging.getLogger('matplotlib').level}. Now logging to: {job_log_file}")
    return logger
