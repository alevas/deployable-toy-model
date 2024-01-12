import logging
import os

"""
This config file, inspired by Flask, stores variables that need to be accessed
throughout the project.
"""

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

logging_datetime_format = '%Y%m%d__%H%M%S'
logging_time_format = '%H:%M:%S'
logging.basicConfig(level=logging.INFO, datefmt=logging_datetime_format)
