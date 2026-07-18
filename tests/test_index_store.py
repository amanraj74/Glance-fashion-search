"""Tests for glance_search.index_store. Uses tiny synthetic embeddings (no model)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from glance_search.config import IndexConfig
from glance_search.errors import IndexNotFoundError
from glance_search.index_store import FaissStore


def test_build_and_save_flat(tmp_path: Path, fake_embeddings: np.ndarray) -> None:
    cfg = IndexConfig(output_dir=str(tmp_path), index_path=str(tmp_path / "x.index"), metadata_path=str(tmp_path / "x.json"))
    store = FaissStore(dim=fake_embeddings.shape[1], cfg=cfg)
    store.build(fake_embeddings)
    assert store.ntotal == len(fake_embeddings)
    paths = [tmp_path / f"img_{i}.jpg" for i in range(len(fake_embeddings))]
    store.save(paths, cfg.index_path_obj, cfg.metadata_path_obj)
    assert cfg.index_path_obj.exists()
    assert cfg.metadata_path_obj.exists()


def test_load_roundtrip(tmp_path: Path, fake_embeddings: np.ndarray) -> None:
    cfg = IndexConfig(output_dir=str(tmp_path), index_path=str(tmp_path / "x.index"), metadata_path=str(tmp_path / "x.json"))
    store = FaissStore(dim=fake_embeddings.shape[1], cfg=cfg)
    store.build(fake_embeddings)
    paths = [tmp_path / f"img_{i}.jpg" for i in range(len(fake_embeddings))]
    store.save(paths, cfg.index_path_obj, cfg.metadata_path_obj)

    loaded, loaded_paths = FaissStore.load(cfg.index_path_obj, cfg.metadata_path_obj, cfg)
    assert loaded.ntotal == len(fake_embeddings)
    assert [str(p) for p in loaded_paths] == [str(p) for p in paths]


def test_search_returns_known_neighbors(tmp_path: Path, fake_embeddings: np.ndarray) -> None:
    cfg = IndexConfig(output_dir=str(tmp_path), index_path=str(tmp_path / "x.index"), metadata_path=str(tmp_path / "x.json"))
    store = FaissStore(dim=fake_embeddings.shape[1], cfg=cfg)
    store.build(fake_embeddings)
    query = fake_embeddings[0:1]
    scores, indices = store.search(query, top_k=3)
    assert indices.shape == (1, 3)
    assert int(indices[0, 0]) == 0
    assert float(scores[0, 0]) == pytest.approx(1.0, abs=1e-4)


def test_missing_artifacts_raises(tmp_path: Path) -> None:
    cfg = IndexConfig(output_dir=str(tmp_path), index_path=str(tmp_path / "x.index"), metadata_path=str(tmp_path / "x.json"))
    with pytest.raises(IndexNotFoundError):
        FaissStore.load(cfg.index_path_obj, cfg.metadata_path_obj, cfg)


def test_ivfflat_builds(tmp_path: Path, fake_embeddings: np.ndarray) -> None:
    cfg = IndexConfig(output_dir=str(tmp_path), index_path=str(tmp_path / "y.index"), metadata_path=str(tmp_path / "y.json"), backend="ivfflat", ivf_nlist=8, ivf_nprobe=2)
    store = FaissStore(dim=fake_embeddings.shape[1], cfg=cfg)
    store.build(fake_embeddings)
    assert store.ntotal == len(fake_embeddings)
