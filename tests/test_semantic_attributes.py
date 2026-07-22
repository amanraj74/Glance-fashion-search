"""Tests for glance_search.semantic_attributes — no model download."""

from __future__ import annotations

import numpy as np

from glance_search.attributes import parse_query
from glance_search.semantic_attributes import (
    SemanticAttributeScores,
    semantic_attribute_score,
)


class _StubModel:
    """Tiny encoder that returns a fixed unit vector for any input."""

    def encode_text(self, text):
        import torch
        out = []
        for s in text:
            v = np.zeros(4, dtype="float32")
            for i, c in enumerate(s):
                v[i % 4] += ord(c) % 7
            out.append(v)
        t = torch.tensor(np.stack(out), dtype=torch.float32)
        return torch.nn.functional.normalize(t, dim=-1)


def test_semantic_attribute_score_empty() -> None:
    s = semantic_attribute_score(np.array([0.0, 0.0, 0.0], dtype="float32"), {})
    assert s.overall == 0.0


def test_semantic_attribute_score_match() -> None:
    vec = np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")
    s = semantic_attribute_score(vec, {"color:yellow": vec.copy()})
    assert s.color > 0.99


def test_semantic_attribute_score_no_match() -> None:
    vec = np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")
    other = np.array([0.0, 0.0, 1.0, 0.0], dtype="float32")
    s = semantic_attribute_score(vec, {"color:yellow": other})
    assert s.color < 0.01


def test_parse_query_then_semantic_no_terms() -> None:
    s = semantic_attribute_score(
        np.array([1.0, 0.0], dtype="float32"),
        {},
    )
    assert s == SemanticAttributeScores()


def test_build_attribute_vectors() -> None:
    from glance_search.semantic_attributes import build_attribute_vectors
    attrs = parse_query("yellow raincoat in a park")
    vectors = build_attribute_vectors(attrs, _StubModel())
    keys = set(vectors.keys())
    assert any(k.startswith("color:") for k in keys)
    assert any(k.startswith("garment:") for k in keys)
    assert any(k.startswith("scene:") for k in keys)
    for v in vectors.values():
        assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-3
