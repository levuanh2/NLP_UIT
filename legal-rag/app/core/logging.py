"""Logging configuration."""

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure process logging without side effects at import time."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
