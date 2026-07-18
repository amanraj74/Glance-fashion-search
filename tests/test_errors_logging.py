"""Tests for glance_search.errors and glance_search.logging_setup."""

from __future__ import annotations

import logging

from glance_search.errors import (
    CaptionError,
    EmbeddingError,
    GlanceError,
    IndexNotFoundError,
    ModelLoadError,
    RerankError,
)
from glance_search.logging_setup import configure_logging, get_logger


def test_exception_hierarchy() -> None:
    assert issubclass(ModelLoadError, GlanceError)
    assert issubclass(IndexNotFoundError, GlanceError)
    assert issubclass(EmbeddingError, GlanceError)
    assert issubclass(CaptionError, GlanceError)
    assert issubclass(RerankError, GlanceError)


def test_can_raise_and_catch() -> None:
    with __import__("pytest").raises(IndexNotFoundError):
        raise IndexNotFoundError("no index")


def test_configure_logging_idempotent(capsys) -> None:
    configure_logging("DEBUG")
    configure_logging("INFO")
    log = get_logger("test_logger")
    log.info("hi")
    captured = capsys.readouterr()
    assert "hi" in captured.err or "hi" in captured.out


def test_logger_returns_named_logger() -> None:
    lg = get_logger("named")
    assert isinstance(lg, logging.Logger)
    assert lg.name == "named"
