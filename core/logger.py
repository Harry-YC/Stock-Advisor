"""
Centralized logging for Literature Review Platform

Provides consistent logging across all modules with configurable
output to console and optional file logging.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# Module-level cache for loggers to prevent duplicate handlers
_loggers = {}


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Configure and return a logger instance.

    Args:
        name: Logger name (typically module name)
        log_file: Optional path to log file
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    # Return cached logger if exists
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler (stderr for visibility in Streamlit)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    _loggers[name] = logger
    return logger


def get_logger(module_name: str) -> logging.Logger:
    """
    Get a logger for a module.

    Args:
        module_name: Short module name (e.g., 'pubmed_client', 'ai_screener')

    Returns:
        Logger instance with 'literature_review.' prefix
    """
    return setup_logger(f"literature_review.{module_name}")
