"""Tests for glance_search.reranker using a stub cross-encoder (no download)."""

from __future__ import annotations

import sys
from dataclasses import dataclass

from glance_search.config import RerankConfig


@dataclass
class StubCrossEncoder:
    def predict(self, pairs, batch_size: int = 16, show_progress_bar: bool = False):
        return [float(len(c[1])) for c in pairs]


def test_rerank_orders_by_predicted(monkeypatch) -> None:
    fake_module = type(sys)("sentence_transformers")
    fake_module.CrossEncoder = staticmethod(lambda model_name: StubCrossEncoder())
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    import importlib
    import glance_search.reranker as reranker_mod
    importlib.reload(reranker_mod)

    candidates = [("a", "short"), ("b", "a longer caption text here"), ("c", "medium length")]
    cfg = RerankConfig(model="stub", batch_size=4)
    out = reranker_mod.rerank("q", candidates, cfg, top_k=3)
    keys = [h.key for h in out]
    assert keys == ["b", "c", "a"]
