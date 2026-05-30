"""Logging configuration utilities for NCAD."""

import logging
import sys


def setup_logging(level_name: str = "INFO") -> logging.Logger:
    """Set up the root logger configuration for NCAD."""
    numeric_level = getattr(logging, level_name.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    # Define simple log formatting
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Configure the root logger
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Disable excessive log spam from third-party libraries if they are set to DEBUG
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    logger = logging.getLogger("NCAD")
    logger.debug(f"Logging initialized with level: {level_name.upper()}")
    return logger
