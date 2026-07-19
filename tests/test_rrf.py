"""Tests for glance_search.pipeline._rrf_fuse — no model download required."""

from __future__ import annotations

from glance_search.pipeline import _rrf_fuse


def test_rrf_fuses_basic() -> None:
    img_ranks = {0: 1, 1: 2, 2: 3}
    cap_ranks = {1: 1, 2: 2, 3: 3}
    fused = _rrf_fuse(img_ranks, cap_ranks)
    assert fused[0] > 0
    assert fused[1] > fused[2]  # 1 in caption → higher


def test_rrf_fuses_disjoint() -> None:
    img = {0: 1, 1: 2}
    cap = {2: 1, 3: 2}
    fused = _rrf_fuse(img, cap)
    assert set(fused.keys()) == {0, 1, 2, 3}


def test_rrf_weighted() -> None:
    img = {0: 1, 1: 5}
    cap = {1: 1, 2: 5}
    fused_a = _rrf_fuse(img, cap, image_w=2.0, caption_w=0.5)
    fused_b = _rrf_fuse(img, cap, image_w=0.5, caption_w=2.0)
    # different weights → different scores
    assert fused_a != fused_b


def test_rrf_k_constant_changes_score() -> None:
    img = {0: 1}
    cap = {0: 2}
    a = _rrf_fuse(img, cap, k_const=10)
    b = _rrf_fuse(img, cap, k_const=100)
    assert a != b