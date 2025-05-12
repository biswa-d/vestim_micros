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
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    return logger
