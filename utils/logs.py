import sys
import os
import logging
import inspect
import __main__
from datetime import datetime
from pathlib import Path


UTILS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = UTILS_DIR.parent
LOG_DIR = PROJECT_ROOT / 'logs'


def logger_writer(caller_name: str = 'MAIN', file_name: str = 'MAIN'):
    '''
    Configures and returns the logger with a dynamic name based on the caller.
    '''
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    
    # Create a logger with a unique name per function
    logger = logging.getLogger(caller_name)
    log_fname = LOG_DIR / f'{datetime.now().strftime("%Y_%m_%d")}_{file_name}.log'
    if not logger.hasHandlers():
        logging.basicConfig(
            level=logging.INFO,  # Default logging level
            format=LOG_FORMAT,
            handlers=[
                logging.FileHandler(log_fname),  # Log to a file
                logging.StreamHandler(sys.stdout),  # Log to console
            ]
        )
    return logger

def log_wrap(func):
    '''
    A decorator that applies logging to each function call
    using the logger_writer function with dynamic function name.
    '''
    def wrapper(*args, **kwargs):
        caller_name = func.__name__  # Get the caller function's name
        try:
            # This gets the filename of the currently running script (like streamlit_app.py)
            file_name = __main__.__file__.split('/')[-1].split('.')[0]
        except AttributeError:
            # Fallback in case __main__.__file__ is not available (e.g., interactive sessions)
            file_name = inspect.getfile(func)  # Get the file where the function is defined
            file_name = file_name.split('\\')[-1].split('.')[0]  # Extract the file name without the path and extension
        logger = logger_writer(caller_name, file_name)  # Get the logger for this function name
        logger.info(f'Executing function: {caller_name}')  # Log before the function execution
        result = func(*args, **kwargs, logger=logger)  # Execute the function passing the logger
        logger.info(f'Completed function: {caller_name}')  # Log after the function execution

        return result
    return wrapper

@log_wrap
def logls(lst, logger=None):
    # Join the list into a single string separated by commas
    s = ', '.join(map(str, lst))
    # Split on every 5th comma
    parts = []
    count = 0
    last = 0
    for i, c in enumerate(s):
        if c == ',':
            count += 1
            if count % 5 == 0:
                parts.append(s[last:i])
                last = i + 2  # skip comma and space
    parts.append(s[last:])
    # Print each part as a row
    for part in parts:
        logger.info(part)

@log_wrap
def logdf(df, logger=None):
    '''
    Logs the DataFrame's head (first few rows) at the INFO level.
    '''
    logger.info(', '.join(df.columns))
    for idx, row in df.head(50).iterrows():
        logger.info(', '.join(map(str, row.tolist())))
