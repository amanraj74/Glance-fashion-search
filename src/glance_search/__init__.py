"""Glance Fashion Search - multimodal retrieval package."""

from glance_search.attributes import QueryAttributes, parse_query
from glance_search.config import (
    Config,
    IndexConfig,
    ModelConfig,
    RetrievalConfig,
    RerankConfig,
    CaptionsConfig,
    load_config,
)
from glance_search.errors import (
    CaptionError,
    EmbeddingError,
    GlanceError,
    IndexNotFoundError,
    ModelLoadError,
    RerankError,
)
from glance_search.index_store import FaissStore, SearchHit
from glance_search.metrics import (
    aggregate_metrics,
    margin_at_k,
    rrf,
    score_entropy,
    score_gap,
    topk_diversity,
)
from glance_search.model import ClipModel
from glance_search.pipeline import (
    LoadedIndexes,
    SearchResult,
    build_caption_index,
    build_image_index,
    load_indexes,
    search,
    search_with_breakdown,
)

__all__ = [
    "Config",
    "CaptionsConfig",
    "IndexConfig",
    "ModelConfig",
    "RetrievalConfig",
    "RerankConfig",
    "load_config",
    "QueryAttributes",
    "parse_query",
    "aggregate_metrics",
    "margin_at_k",
    "rrf",
    "score_entropy",
    "score_gap",
    "topk_diversity",
    "ClipModel",
    "FaissStore",
    "SearchHit",
    "SearchResult",
    "LoadedIndexes",
    "build_image_index",
    "build_caption_index",
    "load_indexes",
    "search",
    "search_with_breakdown",
    "GlanceError",
    "IndexNotFoundError",
    "ModelLoadError",
    "EmbeddingError",
    "CaptionError",
    "RerankError",
]
