"""Cross-encoder re-ranker. Operates on (query, caption_text) pairs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from glance_search.config import RerankConfig
from glance_search.errors import RerankError
from glance_search.logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class RerankHit:
    key: int
    score: float


_MODEL_CACHE: dict[str, object] = {}


def _load_cross_encoder(model_name: str):
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RerankError(
            "sentence-transformers not installed; install with `pip install sentence-transformers`"
        ) from exc
    log.info("loading reranker model=%s (cached after first load)", model_name)
    try:
        model = CrossEncoder(model_name)
    except Exception as exc:
        raise RerankError(f"failed to load reranker {model_name}: {exc}") from exc
    _MODEL_CACHE[model_name] = model
    return model


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def rerank(
    query: str,
    candidates: Sequence[tuple[int, str]],
    cfg: RerankConfig,
    top_k: int,
) -> list[RerankHit]:
    """Re-rank `(image_idx, caption_text)` candidates by cross-encoder relevance to the query.

    Cross-encoders emit raw logits; we squash to [0, 1] via sigmoid so that scores
    combine meaningfully with the cosine-based hybrid score.
    """
    if not candidates:
        return []
    model = _load_cross_encoder(cfg.model)

    keys = [c[0] for c in candidates]
    texts = [c[1] for c in candidates]
    pairs = [[query, t] for t in texts]
    log.info("reranking %d candidates with %s", len(pairs), cfg.model)
    raw = model.predict(pairs, batch_size=cfg.batch_size, show_progress_bar=False)
    raw = raw.tolist() if hasattr(raw, "tolist") else list(raw)
    probs = _sigmoid(raw)
    scored = sorted(zip(keys, probs.tolist()), key=lambda x: x[1], reverse=True)
    return [RerankHit(key=k, score=float(s)) for k, s in scored[:top_k]]
