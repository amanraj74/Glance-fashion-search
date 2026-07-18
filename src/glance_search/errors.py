"""Domain exceptions. Catch `GlanceError` for any package error."""

from __future__ import annotations


class GlanceError(Exception):
    """Base exception for the glance_search package."""


class ConfigError(GlanceError):
    """Invalid or missing configuration."""


class ModelLoadError(GlanceError):
    """Failed to load a model checkpoint."""


class IndexNotFoundError(GlanceError):
    """FAISS index or metadata missing on disk."""


class EmbeddingError(GlanceError):
    """Image embedding pipeline failure."""


class CaptionError(GlanceError):
    """Caption generation failure."""


class RerankError(GlanceError):
    """Re-ranking failure."""
