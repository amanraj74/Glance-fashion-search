"""Per-image attribute tags, derived from the BLIP caption cache.

We extract structured tags (color, garment, scene, style, material) for each
catalogue image from its caption text. The result is a side-table that the
search pipeline uses for hard-negative penalties: if the query asks for
"red tie" and a candidate caption mentions a "white tie", that candidate
gets demoted.

This is the cheapest way to add compositional binding without a larger
neural model — colour / garment / scene words are largely shared between
caption and query.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from glance_search.attributes import (
    _COLOR_TERMS,
    _GARMENT_TERMS,
    _MATERIAL_TERMS,
    _SCENE_TERMS,
    _STYLE_TERMS,
    attribute_overlap_score,
    parse_query,
)
from glance_search.logging_setup import get_logger

log = get_logger(__name__)


def extract_image_tags(caption: str | None) -> dict[str, tuple[str, ...]]:
    """Return {axis -> tuple(terms)} present in ``caption``."""
    if not caption:
        return {"colors": (), "garments": (), "scenes": (), "styles": (), "materials": ()}
    text = caption.lower()
    out: dict[str, tuple[str, ...]] = {}
    for axis, vocab in (
        ("colors", _COLOR_TERMS),
        ("garments", _GARMENT_TERMS),
        ("scenes", _SCENE_TERMS),
        ("styles", _STYLE_TERMS),
        ("materials", _MATERIAL_TERMS),
    ):
        tokens = re.findall(r"[A-Za-z][A-Za-z\-]+", text)
        hits = []
        for tok in tokens:
            stems = {tok, tok.rstrip("s"), tok.rstrip("es")}
            if stems & vocab and tok not in hits:
                hits.append(tok)
        out[axis] = tuple(dict.fromkeys(hits))
    return out


def build_image_attribute_cache(
    captions: dict[str, str],
    out_path: Path,
) -> dict[str, dict[str, tuple[str, ...]]]:
    """Persist image -> {axis -> tuple(tags)} side-table for fast lookup."""
    cache = {path: extract_image_tags(cap) for path, cap in captions.items()}
    serializable = {
        path: {axis: list(tags) for axis, tags in tags_dict.items()}
        for path, tags_dict in cache.items()
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    log.info("wrote image-attribute cache (%d entries) to %s", len(cache), out_path)
    return cache


def load_image_attribute_cache(path: Path) -> dict[str, dict[str, tuple[str, ...]]]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        image_path: {axis: tuple(tags) for axis, tags in tags_dict.items()}
        for image_path, tags_dict in raw.items()
    }


def composite_attribute_score(
    query_attrs,
    image_tags: dict[str, tuple[str, ...]],
) -> tuple[float, dict[str, float]]:
    """Return (overall_score, per_axis_score_dict).

    Score design:
    - For each axis the query touches, look at the candidate's tags.
    - If candidate MATCHES the query axis terms: +1.0
    - If candidate has the *opposite axis* but with a *different* term
      (e.g., "red" vs "white"): -1.0 (hard negative)
    - If candidate has *no* terms from that axis at all: 0.0 (neutral)
    - Weighted average across axes the query touches.
    """
    per_axis: dict[str, float] = {}
    if query_attrs.total_hits == 0:
        return 0.0, per_axis

    axes_to_check = []
    if query_attrs.colors:
        axes_to_check.append(("colors", query_attrs.colors, image_tags.get("colors", ())))
    if query_attrs.garments:
        axes_to_check.append(("garments", query_attrs.garments, image_tags.get("garments", ())))
    if query_attrs.scenes:
        axes_to_check.append(("scenes", query_attrs.scenes, image_tags.get("scenes", ())))
    if query_attrs.styles:
        axes_to_check.append(("styles", query_attrs.styles, image_tags.get("styles", ())))
    if query_attrs.materials:
        axes_to_check.append(("materials", query_attrs.materials, image_tags.get("materials", ())))

    if not axes_to_check:
        return 0.0, per_axis

    weight = {"colors": 0.45, "garments": 0.45, "scenes": 0.30, "styles": 0.20, "materials": 0.20}

    total = 0.0
    total_w = 0.0
    for axis, query_terms, candidate_terms in axes_to_check:
        w = weight[axis]
        if not candidate_terms:
            per_axis[axis] = 0.0
            total_w += w
            continue
        if any(t in candidate_terms for t in query_terms):
            per_axis[axis] = 1.0
            total += w
        elif any(t in candidate_terms for t in _FASHION_NEGATIVES.get(axis, ())):
            per_axis[axis] = -1.0
            total -= w
        else:
            per_axis[axis] = 0.0
        total_w += w

    return (total / total_w) if total_w else 0.0, per_axis


_FASHION_NEGATIVES: dict[str, tuple[str, ...]] = {
    "colors": (
        "red", "blue", "green", "yellow", "black", "white", "pink", "gray",
        "grey", "brown", "beige", "orange", "purple", "navy", "tan", "cream",
        "khaki", "burgundy", "maroon", "olive",
    ),
    "garments": (
        "shirt", "dress", "skirt", "pants", "jeans", "jacket", "coat",
        "raincoat", "sweater", "hoodie", "tie", "vest", "blazer", "suit",
    ),
    "scenes": (
        "office", "park", "beach", "garden", "studio", "runway", "stage",
        "wedding", "formal", "casual", "weekend",
    ),
    "styles": ("casual", "formal", "vintage", "sporty", "elegant"),
}


def collect_axis_inventory(
    image_tags: dict[str, dict[str, tuple[str, ...]]],
) -> dict[str, dict[str, int]]:
    """Build catalogue-wide term counts per axis (used to find rare/distinctive terms)."""
    inv: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for tags in image_tags.values():
        for axis, terms in tags.items():
            for t in terms:
                inv[axis][t] += 1
    return {axis: dict(counts) for axis, counts in inv.items()}


def find_hard_negatives_for_query(
    query_attrs,
    image_tags: dict[str, dict[str, tuple[str, ...]]],
    top_k: int = 20,
) -> list[tuple[str, float]]:
    """For each image, compute a hard-negative penalty if it has the wrong attributes.

    Returns a sorted list of (image_path, penalty) where penalty < 0 means the
    candidate is a likely hard-negative (has *a* term on a queried axis but not
    the one the query asked for).
    """
    if query_attrs.total_hits == 0:
        return []
    flagged: list[tuple[str, float]] = []
    for path, tags in image_tags.items():
        score, _ = composite_attribute_score(query_attrs, tags)
        if score < 0:
            flagged.append((path, score))
    flagged.sort(key=lambda x: x[1])
    return flagged[:top_k]
