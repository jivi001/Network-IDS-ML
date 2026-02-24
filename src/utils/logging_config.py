from __future__ import annotations

import logging
import os


def configure_logging(name: str) -> logging.Logger:
    """Configure a structured logger with runtime log level from environment."""
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)
