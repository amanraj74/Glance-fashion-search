"""pytest fixtures shared across test modules."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def tmp_config(tmp_path: Path, repo_root: Path):
    """Write a config.yaml into a temp dir, return its path and a chdir wrapper."""
    cfg = repo_root / "config.yaml"
    backup = None
    if cfg.exists():
        backup = cfg.read_bytes()
    try:
        new_path = tmp_path / "config.yaml"
        new_path.write_text(
            "model:\n  name: ViT-B-32\n  pretrained: openai\n  device: cpu\n"
            "index:\n  image_dir: dataset/images\n  output_dir: output\n"
            "  index_path: output/faiss.index\n  metadata_path: output/metadata.json\n"
            "  caption_path: output/captions.json\n  caption_index_path: output/captions.index\n"
            "  backend: flat\n  ivf_nlist: 64\n  ivf_nprobe: 8\n"
            "retrieval:\n  top_k: 5\n  rerank_top_n: 50\n  caption_weight: 0.4\n"
            "  image_weight: 0.6\n  use_captions: true\n  use_reranker: true\n"
            "captions:\n  enabled: true\n  model: Salesforce/blip-image-captioning-base\n"
            "  batch_size: 8\n  max_new_tokens: 30\n  num_beams: 3\n"
            "rerank:\n  enabled: true\n  model: cross-encoder/ms-marco-MiniLM-L-2-v2\n"
            "  batch_size: 16\nlog_level: INFO\n",
            encoding="utf-8",
        )
        yield new_path
    finally:
        if backup is not None:
            cfg.write_bytes(backup)
        elif cfg.exists():
            cfg.unlink()


@pytest.fixture
def fake_embeddings() -> np.ndarray:
    rng = np.random.default_rng(42)
    arr = rng.standard_normal((50, 64)).astype("float32")
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    return arr


@pytest.fixture
def fake_image_list(tmp_path: Path) -> list[Path]:
    paths = []
    for i in range(5):
        p = tmp_path / f"img_{i}.jpg"
        p.write_bytes(b"\xff\xd8\xff\xe0fake")
        paths.append(p)
    return paths
