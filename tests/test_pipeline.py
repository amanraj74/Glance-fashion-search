"""Tests for glance_search.pipeline (composition only, uses stub model)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from glance_search.config import Config, IndexConfig, RetrievalConfig
from glance_search.index_store import FaissStore
from glance_search.pipeline import (
    LoadedIndexes,
    SearchResult,
    search,
)


class _Arr:
    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def cpu(self):
        return _CpuView(self._arr)


class _CpuView:
    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def numpy(self):
        return self._arr


def _to_t(arr: np.ndarray):
    return _Arr(arr)


class StubModel:
    dim = 16

    def encode_images(self, images):
        return _to_t(_ones(len(images)))

    def encode_text(self, text):
        return _to_t(_ones(1))


def _ones(n: int) -> np.ndarray:
    arr = np.zeros((n, StubModel.dim), dtype=np.float32)
    arr[:, 0] = 1.0
    return arr


def test_build_image_index_writes_artifacts(tmp_path: Path, monkeypatch) -> None:
    index_cfg = IndexConfig(
        image_dir=str(tmp_path),
        output_dir=str(tmp_path),
        index_path=str(tmp_path / "f.index"),
        metadata_path=str(tmp_path / "f.json"),
    )
    retrieval_cfg = RetrievalConfig(use_captions=False, use_reranker=False)
    cfg = Config(index=index_cfg, retrieval=retrieval_cfg)

    rng = np.random.default_rng(0)
    fake_embeddings = rng.standard_normal((5, StubModel.dim)).astype("float32")
    fake_embeddings /= np.linalg.norm(fake_embeddings, axis=1, keepdims=True)

    paths = [tmp_path / f"img_{i}.jpg" for i in range(5)]

    from glance_search import pipeline as pipeline_mod
    monkeypatch.setattr(pipeline_mod, "list_images", lambda root, exts=None: paths)
    monkeypatch.setattr(pipeline_mod, "embed_corpus", lambda p, m, batch_size=16: (fake_embeddings, paths))
    monkeypatch.setattr(pipeline_mod.ClipModel, "get", classmethod(lambda cls, c: StubModel()))

    store, kept = pipeline_mod.build_image_index(cfg)
    assert store.ntotal == 5
    assert len(kept) == 5


def test_search_returns_ranked_results(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    emb = rng.standard_normal((20, 16)).astype("float32")
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    paths = [tmp_path / f"img_{i}.jpg" for i in range(20)]

    index_cfg = IndexConfig(
        image_dir=str(tmp_path),
        output_dir=str(tmp_path),
        index_path=str(tmp_path / "f.index"),
        metadata_path=str(tmp_path / "f.json"),
    )
    retrieval_cfg = RetrievalConfig(use_captions=False, use_reranker=False)
    cfg = Config(index=index_cfg, retrieval=retrieval_cfg)

    store = FaissStore(dim=16, cfg=index_cfg)
    store.build(emb)
    store.save(paths, index_cfg.index_path_obj, index_cfg.metadata_path_obj)
    loaded = LoadedIndexes(image_store=store, image_paths=paths, caption_store=None)

    results = search("x", cfg, loaded=loaded, model=StubModel())
    assert isinstance(results, list)
    assert len(results) <= cfg.retrieval.top_k
    assert all(isinstance(r, SearchResult) for r in results)
