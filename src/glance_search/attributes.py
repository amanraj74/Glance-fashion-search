"""Lightweight query-attribute parsing for fashion retrieval.

We extract color, garment, scene, style, and material hints from a free-text
query using lexical lookups. This is a small, deterministic step that lifts
compositionality without a heavy NER model.

Used by ``pipeline.py`` to apply an attribute-matching bonus during re-ranking.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

_COLOR_TERMS: set[str] = {
    "red", "blue", "green", "yellow", "black", "white", "pink", "gray", "grey",
    "brown", "beige", "orange", "purple", "navy", "tan", "cream", "khaki",
    "burgundy", "maroon", "olive", "gold", "silver", "violet", "teal",
    "turquoise", "ivory", "plaid", "striped", "floral", "denim", "leather",
    "silk", "satin", "velvet",
}

_GARMENT_TERMS: set[str] = {
    "shirt", "tshirt", "t-shirt", "top", "blouse", "tank", "dress", "skirt",
    "pants", "jeans", "trousers", "shorts", "jacket", "coat", "raincoat",
    "sweater", "hoodie", "cardigan", "vest", "blazer", "suit", "tie", "bowtie",
    "shoes", "boots", "heels", "sneakers", "socks", "hat", "cap", "scarf",
    "gloves", "bag", "handbag", "backpack", "belt", "sunglasses",
    "pajamas", "robe", "swimsuit", "bikini",
}

_SCENE_TERMS: set[str] = {
    "office", "park", "city", "street", "beach", "garden", "indoor", "outdoor",
    "home", "restaurant", "cafe", "gym", "studio", "runway", "stage",
    "wedding", "party", "formal", "casual", "winter", "summer", "spring",
    "autumn", "fall", "snow", "rain", "sunny", "rainy", "snowy", "overcast",
    "modern", "vintage", "traditional", "minimalist", "luxury", "elegant",
    "sporty", "professional", "weekend", "evening", "morning", "night",
    "forest", "desert", "urban", "rural", "rooftop", "bench",
}

_STYLE_TERMS: set[str] = {
    "casual", "formal", "elegant", "sporty", "professional", "business",
    "edgy", "bohemian", "minimalist", "vintage", "retro", "classic",
    "modern", "luxurious", "streetwear", "athleisure", "chic", "trendy",
    "stylish", "smart", "preppy", "grunge",
}

_MATERIAL_TERMS: set[str] = {
    "cotton", "silk", "denim", "leather", "wool", "linen", "cashmere",
    "polyester", "nylon", "velvet", "satin", "lace", "chiffon", "suede",
    "flannel", "corduroy", "tweed", "knit",
}


@dataclass(frozen=True)
class QueryAttributes:
    colors: tuple[str, ...] = ()
    garments: tuple[str, ...] = ()
    scenes: tuple[str, ...] = ()
    styles: tuple[str, ...] = ()
    materials: tuple[str, ...] = ()
    raw_tokens: tuple[str, ...] = ()

    @property
    def total_hits(self) -> int:
        return sum(len(getattr(self, k)) for k in ("colors", "garments", "scenes", "styles", "materials"))

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "colors": list(self.colors),
            "garments": list(self.garments),
            "scenes": list(self.scenes),
            "styles": list(self.styles),
            "materials": list(self.materials),
        }


def _scan(text: str, vocab: set[str]) -> tuple[str, ...]:
    """Return members of ``vocab`` that appear in ``text`` as whole words."""
    tokens = re.findall(r"[A-Za-z][A-Za-z\-]+", text.lower())
    found = []
    for tok in tokens:
        # handle plural/plural-y stripping
        stems = {tok, tok.rstrip("s"), tok.rstrip("es")}
        if stems & vocab:
            found.append(tok)
    return tuple(dict.fromkeys(found))


def parse_query(query: str) -> QueryAttributes:
    """Extract color / garment / scene / style / material hints from ``query``."""
    return QueryAttributes(
        colors=_scan(query, _COLOR_TERMS),
        garments=_scan(query, _GARMENT_TERMS),
        scenes=_scan(query, _SCENE_TERMS),
        styles=_scan(query, _STYLE_TERMS),
        materials=_scan(query, _MATERIAL_TERMS),
        raw_tokens=tuple(re.findall(r"[A-Za-z][A-Za-z\-]+", query.lower())),
    )


def attribute_overlap_score(attrs: QueryAttributes, candidate_text: str | None) -> float:
    """Return 0..1 weighted score of how well ``candidate_text`` matches ``attrs``.

    Colors and garments weight more than scene/style because they are
    visually verifiable; materials weight slightly less because BLIP-base often
    mislabels them. Returns 0.0 when ``candidate_text`` is empty/None.
    """
    if not candidate_text or attrs.total_hits == 0:
        return 0.0
    text = candidate_text.lower()
    score = 0.0
    weight = {"colors": 0.30, "garments": 0.30, "scenes": 0.20, "styles": 0.10, "materials": 0.10}
    for kind, terms in attrs.to_dict().items():
        if not terms:
            continue
        hits = sum(1 for t in terms if t in text)
        if hits:
            score += weight[kind] * (hits / len(terms))
    return min(1.0, score)


def missing_attributes(attrs: QueryAttributes, candidate_text: str | None) -> tuple[str, ...]:
    """Return the attributes from ``attrs`` that do NOT appear in ``candidate_text``."""
    if not candidate_text:
        return ()
    text = candidate_text.lower()
    missing: list[str] = []
    for terms in attrs.to_dict().values():
        for t in terms:
            if t not in text:
                missing.append(t)
    return tuple(missing)


def query_axis_tags(attrs: QueryAttributes) -> list[str]:
    """Return one-word axis tags the query touches, used for the eval axis manifest."""
    tags: list[str] = []
    if attrs.colors:
        tags.append("color")
    if attrs.garments:
        tags.append("garment")
    if attrs.scenes:
        tags.append("scene")
    if attrs.styles:
        tags.append("style")
    if attrs.materials:
        tags.append("material")
    return tags or ["free-form"]