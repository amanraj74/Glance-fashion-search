"""FAISS index wrappers - flat (exact) and IVFFlat (scalable)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from glance_search.config import IndexConfig
from glance_search.errors import IndexNotFoundError
from glance_search.logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class SearchHit:
    path: Path
    score: float
    rank: int


class FaissStore:
    """Thin wrapper around FAISS with persistence + metadata."""

    def __init__(self, dim: int, cfg: IndexConfig):
        self.dim = dim
        self.cfg = cfg
        self._index: faiss.Index | None = None

    @property
    def ntotal(self) -> int:
        return int(self._index.ntotal) if self._index is not None else 0

    def build(self, embeddings: np.ndarray) -> None:
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype("float32")
        n, d = embeddings.shape
        if d != self.dim:
            raise ValueError(f"dim mismatch: index dim={self.dim}, got {d}")
        if self.cfg.backend == "flat":
            log.info("building IndexFlatIP n=%d d=%d", n, d)
            self._index = faiss.IndexFlatIP(d)
        elif self.cfg.backend == "ivfflat":
            nlist = min(self.cfg.ivf_nlist, max(1, n // 10))
            log.info("building IndexIVFFlat n=%d d=%d nlist=%d nprobe=%d", n, d, nlist, self.cfg.ivf_nprobe)
            quantizer = faiss.IndexFlatIP(d)
            self._index = faiss.IndexIVFFlat(quantizer, d, nlist)
            self._index.train(embeddings)
            self._index.nprobe = self.cfg.ivf_nprobe
        else:
            raise ValueError(f"unknown index backend: {self.cfg.backend}")
        self._index.add(embeddings)

    def search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if self._index is None:
            raise RuntimeError("index not built or loaded")
        if query.dtype != np.float32:
            query = query.astype("float32")
        if query.ndim == 1:
            query = query[np.newaxis, :]
        return self._index.search(query, top_k)

    def save(self, paths: list[Path], index_path: Path, metadata_path: Path) -> None:
        if self._index is None:
            raise RuntimeError("index not built")
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(index_path))
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump([str(p) for p in paths], f, indent=2)
        log.info("wrote index=%s ntotal=%d metadata=%s rows=%d", index_path, self.ntotal, metadata_path, len(paths))

    @classmethod
    def load(cls, index_path: Path, metadata_path: Path, cfg: IndexConfig) -> tuple["FaissStore", list[Path]]:
        if not index_path.exists() or not metadata_path.exists():
            raise IndexNotFoundError(
                f"missing index artifacts: expected {index_path} and {metadata_path}; run build_index first"
            )
        raw_index = faiss.read_index(str(index_path))
        with open(metadata_path, "r", encoding="utf-8") as f:
            path_strs = json.load(f)
        store = cls(dim=raw_index.d, cfg=cfg)
        store._index = raw_index
        log.info("loaded index type=%s dim=%d ntotal=%d", type(raw_index).__name__, raw_index.d, raw_index.ntotal)
        return store, [Path(p) for p in path_strs]
