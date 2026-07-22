"""Semantic attribute scoring via CLIP text-encoder.

For each query attribute (color, garment), we encode a small set of natural
language phrasings ("a photo of a yellow raincoat", "yellow outerwear", ...) and
average their CLIP embeddings into a single attribute vector. We then compare
that vector against the image embedding directly — this is a fashion-aware
semantic similarity that beats lexical overlap when BLIP's captions are sparse.

Used in ``pipeline.search`` to add a small "semantic attribute bonus" on top of
the existing RRF and re-ranker scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from glance_search.attributes import (
    _COLOR_TERMS,
    _GARMENT_TERMS,
    _SCENE_TERMS,
    _STYLE_TERMS,
)
from glance_search.attributes import QueryAttributes
from glance_search.logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class SemanticAttributeScores:
    color: float = 0.0
    garment: float = 0.0
    scene: float = 0.0
    style: float = 0.0
    overall: float = 0.0


_ATTRIBUTE_PROMPTS: dict[str, tuple[str, ...]] = {
    "color": (
        "a fashion photo in {} color",
        "an outfit that is {}",
        "clothing in shade of {}",
    ),
    "garment": (
        "a photo of a person wearing a {}",
        "a fashion product photo of a {}",
        "an image showing a {} garment",
    ),
    "scene": (
        "a fashion photo taken in a {}",
        "an image set in a {}",
        "a photo of a {} scene",
    ),
    "style": (
        "an outfit in {} style",
        "a photo of a {} look",
        "fashion photography in {} style",
    ),
}


def _encode_attribute_prompts(
    term: str,
    axis: str,
    model,
) -> np.ndarray | None:
    """Return one L2-normalised embedding for the given attribute term."""
    templates = _ATTRIBUTE_PROMPTS.get(axis, ())
    if not templates:
        return None
    prompts = [tpl.format(term) for tpl in templates]
    try:
        feats = model.encode_text(prompts)
    except Exception as exc:
        log.warning("attribute encode failed for %s/%s: %s", axis, term, exc)
        return None
    feats = feats.mean(dim=0, keepdim=True)
    feats = torch.nn.functional.normalize(feats, dim=-1)
    return feats.cpu().numpy().astype("float32")[0]


def build_attribute_vectors(
    attrs: QueryAttributes,
    model,
) -> dict[str, np.ndarray]:
    """Pre-compute one vector per attribute term (averaged across templates)."""
    out: dict[str, np.ndarray] = {}
    mapping = (
        ("color", attrs.colors),
        ("garment", attrs.garments),
        ("scene", attrs.scenes),
        ("style", attrs.styles),
    )
    for axis, terms in mapping:
        for term in terms:
            v = _encode_attribute_prompts(term, axis, model)
            if v is not None:
                out[f"{axis}:{term}"] = v
    return out


def semantic_attribute_score(
    image_embedding: np.ndarray,
    attr_vectors: dict[str, np.ndarray],
) -> SemanticAttributeScores:
    """Compare a stored image embedding to each query attribute vector.

    Returns per-axis scores (the maximum cosine over the terms in that axis) and
    an overall score (weighted average).
    """
    if not attr_vectors or image_embedding is None:
        return SemanticAttributeScores()
    n = float(np.linalg.norm(image_embedding))
    if n == 0:
        return SemanticAttributeScores()
    img = image_embedding / n

    per_axis: dict[str, list[float]] = {"color": [], "garment": [], "scene": [], "style": []}
    for key, vec in attr_vectors.items():
        axis, _term = key.split(":", 1)
        if axis not in per_axis:
            continue
        m = float(np.linalg.norm(vec))
        if m == 0:
            continue
        cos = float(np.dot(img, vec / m))
        per_axis[axis].append(cos)

    out = SemanticAttributeScores()
    weight = {"color": 0.30, "garment": 0.30, "scene": 0.25, "style": 0.15}
    total = 0.0
    total_w = 0.0
    for axis, vals in per_axis.items():
        if not vals:
            continue
        max_v = max(vals)
        setattr(out, axis, max_v)
        total += weight[axis] * max_v
        total_w += weight[axis]
    out.overall = (total / total_w) if total_w else 0.0
    return out