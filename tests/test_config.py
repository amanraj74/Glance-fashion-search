"""Tests for glance_search.config."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from glance_search.config import (
    CaptionsConfig,
    Config,
    IndexConfig,
    ModelConfig,
    RetrievalConfig,
    RerankConfig,
    load_config,
)


def test_defaults_load(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert isinstance(cfg.model, ModelConfig)
    assert isinstance(cfg.index, IndexConfig)
    assert isinstance(cfg.retrieval, RetrievalConfig)
    assert isinstance(cfg.captions, CaptionsConfig)
    assert isinstance(cfg.rerank, RerankConfig)


def test_yaml_overrides(tmp_path: Path, monkeypatch) -> None:
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "model:\n  name: ViT-B-32\n  pretrained: laion2b_s34b_b79k\n"
        "retrieval:\n  top_k: 9\n  use_reranker: false\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    cfg = load_config(cfg_file)
    assert cfg.model.name == "ViT-B-32"
    assert cfg.model.pretrained == "laion2b_s34b_b79k"
    assert cfg.retrieval.top_k == 9
    assert cfg.retrieval.use_reranker is False


def test_env_overrides(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GLANCE_MODEL__NAME", "ViT-L-14")
    monkeypatch.setenv("GLANCE_RETRIEVAL__TOP_K", "13")
    monkeypatch.setenv("GLANCE_RETRIEVAL__USE_RERANKER", "false")
    cfg = load_config()
    assert cfg.model.name == "ViT-L-14"
    assert cfg.retrieval.top_k == 13
    assert cfg.retrieval.use_reranker is False


def test_path_helpers(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert isinstance(cfg.image_dir_path, Path)
    assert isinstance(cfg.index_path_obj, Path)
    assert isinstance(cfg.metadata_path_obj, Path)
    assert isinstance(cfg.caption_path_obj, Path)
    assert isinstance(cfg.caption_index_path_obj, Path)


def test_missing_yaml_returns_defaults(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = load_config("nonexistent.yaml")
    assert cfg.model.name == ModelConfig().name


def test_frozen_dataclass() -> None:
    cfg = Config()
    with pytest.raises(Exception):
        cfg.model = ModelConfig(name="ViT-L-14")
