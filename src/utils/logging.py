"""Logging utilities for cognitive flexibility experiments."""
import logging
from datetime import datetime
from pathlib import Path

def setup_logger(model_name: str, test_name: str, log_dir: str = "logs") -> logging.Logger:
    """Setup experiment logger with standardized formatting."""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    logger = logging.getLogger(f"{test_name}_{model_name}")
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(
        log_path / f"{test_name}_{model_name}_{timestamp}.log"
    )
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger