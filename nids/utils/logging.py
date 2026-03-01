"""Logging utilities for NIDS."""

import logging
import sys
from pathlib import Path
from typing import Optional
from pythonjsonlogger import jsonlogger

def setup_logger(
    name: str = "nids.core",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logger with JSON SOC-ready console and optional file output.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    if not logger.handlers:
        # Default format for JSON
        if format_string is None:
            format_string = '%(asctime)s %(levelname)s %(name)s %(message)s'
        
        # SOC-ready JSON formatter
        formatter = jsonlogger.JsonFormatter(format_string, timestamp=True)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler (optional)
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
        logger.propagate = False
    
    return logger
