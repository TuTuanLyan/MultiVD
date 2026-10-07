"""Shared console logging configuration for all experiment entry points."""

import logging
import sys


LOGGER_NAME = "multivd"


def configure_logging():
    """Log to stdout without logger/function names so shell tee can persist it."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return logger


def get_logger():
    return logging.getLogger(LOGGER_NAME)
