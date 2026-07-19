"""Rank-based metrics that don't need labelled ground truth.

These are useful proxies when we can't manually label every relevant image:

- ``score_gap`` — margin between the top-1 score and the runner-up. Higher
  means the model is more confident in its top pick.
- ``topk_diversity`` — fraction of *unique* image paths among the top-k. Higher
  means the model is not degenerate (returning near-duplicates).
- ``score_entropy`` — Shannon entropy of the softmaxed top-k scores. Higher
  means the model distributes mass across candidates; lower means it has a
  clear winner.
- ``margin_at_k`` — relative score gap between rank-k and rank-(k+1). Higher
  means there is a clean cut between the top-k and the rest.
- ``rrf`` — classic Reciprocal Rank Fusion weight = 1 / (k_const + rank).

These metrics are surfaced via ``scripts/run_metrics.py`` and stored alongside
the existing ablation CSV.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np


def score_gap(scores: Sequence[float]) -> float:
    if len(scores) < 2:
        return 0.0
    sorted_s = sorted(scores, reverse=True)
    return float(sorted_s[0] - sorted_s[1])


def topk_diversity(paths: Sequence[str], top_k: int) -> float:
    if not paths:
        return 0.0
    head = paths[:top_k]
    return len(set(head)) / max(len(head), 1)


def score_entropy(scores: Sequence[float]) -> float:
    if not scores:
        return 0.0
    arr = np.asarray(scores, dtype=np.float64)
    arr = arr - arr.max()
    p = np.exp(arr)
    p = p / max(p.sum(), 1e-12)
    return float(-(p * np.log(np.clip(p, 1e-12, 1.0))).sum())


def margin_at_k(scores: Sequence[float], k: int) -> float:
    arr = sorted(scores, reverse=True)
    if k >= len(arr) - 1:
        return 0.0
    return float(arr[k - 1] - arr[k])


def rrf(rank: int, k_const: int = 60) -> float:
    """Reciprocal Rank Fusion weight for a given rank (1-indexed)."""
    if rank < 1:
        return 0.0
    return 1.0 / (k_const + rank)


def aggregate_metrics(results: Iterable[dict], top_k: int = 5) -> dict[str, float]:
    """Aggregate the rank-based metrics over a list of ``SearchResult``-like dicts."""
    scores = [float(r.get("score", 0.0)) for r in results]
    paths = [str(r.get("path", "")) for r in results]
    if not scores:
        return {
            "top1_score": 0.0,
            "score_gap": 0.0,
            "topk_diversity": 0.0,
            "score_entropy": 0.0,
            "margin_at_k": 0.0,
        }
    return {
        "top1_score": scores[0],
        "score_gap": score_gap(scores),
        "topk_diversity": topk_diversity(paths, top_k),
        "score_entropy": score_entropy(scores),
        "margin_at_k": margin_at_k(scores, top_k),
    }
