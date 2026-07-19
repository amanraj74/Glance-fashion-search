"""Tests for glance_search.metrics — no model download required."""

from __future__ import annotations

import pytest

from glance_search.metrics import (
    aggregate_metrics,
    margin_at_k,
    rrf,
    score_entropy,
    score_gap,
    topk_diversity,
)


def test_score_gap_basic() -> None:
    assert score_gap([0.9, 0.7, 0.5]) == pytest.approx(0.2)
    assert score_gap([0.5]) == 0.0


def test_score_gap_single() -> None:
    assert score_gap([0.4]) == 0.0
    assert score_gap([]) == 0.0


def test_topk_diversity_full() -> None:
    assert topk_diversity(["a", "b", "c", "d", "e"], 5) == 1.0


def test_topk_diversity_dups() -> None:
    assert topk_diversity(["a", "a", "a", "a"], 4) == 0.25


def test_topk_diversity_empty() -> None:
    assert topk_diversity([], 5) == 0.0


def test_score_entropy_higher_for_uniform() -> None:
    e_uniform = score_entropy([0.5, 0.5, 0.5, 0.5])
    e_peaked = score_entropy([1.0, 0.0, 0.0, 0.0])
    assert e_uniform > e_peaked


def test_score_entropy_empty() -> None:
    assert score_entropy([]) == 0.0


def test_margin_at_k() -> None:
    s = [0.9, 0.8, 0.7, 0.4, 0.3, 0.2]
    assert margin_at_k(s, 3) == pytest.approx(0.3)  # 0.7 - 0.4


def test_rrf_first_rank() -> None:
    assert rrf(1, k_const=60) == 1.0 / 61
    assert rrf(1, k_const=10) == 1.0 / 11


def test_rrf_zero_or_negative() -> None:
    assert rrf(0) == 0.0
    assert rrf(-1) == 0.0


def test_rrf_monotonic_decreasing() -> None:
    assert rrf(1) > rrf(2) > rrf(10)


def test_aggregate_metrics_keys() -> None:
    rows = [{"score": 0.9, "path": "a"}, {"score": 0.5, "path": "b"}]
    m = aggregate_metrics(rows, top_k=2)
    assert m["top1_score"] == 0.9
    assert m["score_gap"] == 0.4
    assert m["topk_diversity"] == 1.0
    assert 0.0 <= m["score_entropy"] <= 3.0


def test_aggregate_metrics_empty() -> None:
    m = aggregate_metrics([], top_k=5)
    assert m["top1_score"] == 0.0
    assert m["topk_diversity"] == 0.0