import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(name, log_file=None, level=logging.INFO, 
                formatter_str='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                console_output=True):
    """
    Set up a logger with file and optional console handlers.
    
    Parameters:
    -----------
    name : str
        Name of the logger
    log_file : str or Path, optional
        Path to the log file. If None, no file handler is created.
    level : int, default=logging.INFO
        Logging level (e.g., logging.DEBUG, logging.INFO)
    formatter_str : str
        Format string for the log messages
    console_output : bool, default=True
        Whether to output logs to console
        
    Returns:
    --------
    logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear any existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
    
    formatter = logging.Formatter(formatter_str)
    
    # Add file handler if log_file is specified
    if log_file:
        # Ensure parent directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

def get_timestamped_log_filename(base_path, base_name, extension='.log'):
    """
    Create a timestamped log filename.
    
    Parameters:
    -----------
    base_path : str or Path
        Directory to store the log file
    base_name : str
        Base name for the log file
    extension : str, default='.log'
        File extension for the log file
        
    Returns:
    --------
    Path
        Path object for the timestamped log file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(base_path) / f"logs/{base_name}_{timestamp}{extension}"

def log_exception(logger, exc_info=True):
    """
    Log an exception with traceback.
    
    Parameters:
    -----------
    logger : logging.Logger
        Logger instance
    exc_info : bool or Exception, default=True
        Exception information to log
    """
    logger.error("Exception occurred", exc_info=exc_info)



def log_measurement_end(logger, duration_seconds):
    """
    Log the end of a measurement with duration and summary statistics.
    
    Parameters:
    -----------
    logger : logging.Logger
        Logger instance
    duration_seconds : float
        Duration of the measurement in seconds
    """
    logger.info("========== MEASUREMENT COMPLETED ==========")
    logger.info(f"Duration: {duration_seconds:.2f} seconds")
    if duration_seconds > 60:
        minutes = duration_seconds / 60
        logger.info(f"         {minutes:.2f} minutes")
    if duration_seconds > 3600:
        hours = duration_seconds / 3600
        logger.info(f"         {hours:.2f} hours")
    logger.info("==========================================")
