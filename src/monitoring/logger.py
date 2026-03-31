"""Structured logging configuration."""

from __future__ import annotations

import logging
import sys
from typing import Literal


def setup_logging(
    level: str = "INFO",
    format_type: Literal["json", "text"] = "text",
) -> logging.Logger:
    """Configure and return the application logger.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        format_type: Output format — 'text' for human-readable, 'json' for structured.
    """
    logger = logging.getLogger("llm_router")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper(), logging.INFO))

    if format_type == "json":
        fmt = logging.Formatter(
            '{"timestamp":"%(asctime)s","level":"%(levelname)s",'
            '"module":"%(module)s","message":"%(message)s"}'
        )
    else:
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(module)s:%(funcName)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    handler.setFormatter(fmt)
    logger.addHandler(handler)
    return logger


def get_logger() -> logging.Logger:
    """Get the application logger."""
    return logging.getLogger("llm_router")
