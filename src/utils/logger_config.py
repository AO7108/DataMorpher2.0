# src/utils/logger_config.py

import logging
import sys

def setup_logging():
    """
    Configures the root logger for the entire application.
    """
    # Define the format for our log messages
    log_format = "%(asctime)s - %(levelname)s - [%(name)s] - %(message)s"
    
    # Configure the logging system
    logging.basicConfig(
        level=logging.INFO,             # Set the minimum level of messages to log
        format=log_format,              # Use our defined format
        stream=sys.stdout               # Log messages to the console (standard output)
    )

    # You could also add a file handler here to log to a file
    # file_handler = logging.FileHandler("app.log")
    # file_handler.setFormatter(logging.Formatter(log_format))
    # logging.getLogger().addHandler(file_handler)

    logging.info("Logging configured successfully.")