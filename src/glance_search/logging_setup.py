"""Logging configuration. Idempotent."""

from __future__ import annotations

import logging
import os
import sys

_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
_CONFIGURED = False


def configure_logging(level: str | None = None) -> None:
    """Configure root logger. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        logging.getLogger().setLevel((level or os.environ.get("LOG_LEVEL") or "INFO").upper())
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel((level or os.environ.get("LOG_LEVEL") or "INFO").upper())
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
