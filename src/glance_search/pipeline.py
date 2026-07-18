"""High-level retrieval pipeline. Wires model, indexes, captions, and re-ranker.

Design: hybrid late-interaction over image and caption embeddings, optionally
re-ranked by a cross-encoder over caption text. All heavy lifting lives in
modules; this file is orchestration only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from glance_search.captions import caption_corpus
from glance_search.config import Config
from glance_search.embedder import embed_corpus, list_images
from glance_search.index_store import FaissStore
from glance_search.logging_setup import get_logger
from glance_search.model import ClipModel
from glance_search.reranker import rerank

log = get_logger(__name__)


@dataclass
class SearchResult:
    path: Path
    score: float
    rank: int
    image_score: float = 0.0
    caption_score: float = 0.0
    rerank_score: float | None = None
    caption: str | None = None


@dataclass
class LoadedIndexes:
    image_store: FaissStore
    image_paths: list[Path]
    caption_store: FaissStore | None
    captions: dict[str, str] = field(default_factory=dict)


def build_image_index(cfg: Config) -> tuple[FaissStore, list[Path]]:
    """Embed every image in `cfg.index.image_dir` and persist the FAISS index."""
    paths = list_images(cfg.index.image_dir)
    log.info("discovered %d images under %s", len(paths), cfg.index.image_dir)
    model = ClipModel.get(cfg.model)
    embeddings, kept = embed_corpus(paths, model, batch_size=16)
    store = FaissStore(dim=embeddings.shape[1], cfg=cfg.index)
    store.build(embeddings)
    store.save(kept, cfg.index_path_obj, cfg.metadata_path_obj)
    return store, kept


def build_caption_index(cfg: Config) -> tuple[FaissStore, dict[str, str]]:
    """Generate captions (if missing), embed with the same CLIP text encoder, persist."""
    paths = list_images(cfg.index.image_dir)
    captions = caption_corpus(paths, cfg.captions, cfg.caption_path_obj)

    model = ClipModel.get(cfg.model)
    valid = [(p, captions[str(p)]) for p in paths if captions.get(str(p))]
    if not valid:
        log.warning("no captions available; skipping caption index build")
        empty_store = FaissStore(dim=model.dim, cfg=cfg.index)
        return empty_store, captions

    with __import__("torch").no_grad():
        feats = model.encode_text([t for _, t in valid])
    embeddings = feats.cpu().numpy().astype("float32")

    store = FaissStore(dim=embeddings.shape[1], cfg=cfg.index)
    store.build(embeddings)

    meta_path = cfg.caption_index_path_obj.with_name("caption_meta.json")
    store.save([p for p, _ in valid], cfg.caption_index_path_obj, meta_path)
    return store, captions


def load_indexes(cfg: Config) -> LoadedIndexes:
    """Load persisted image + caption indexes. Caption parts are optional."""
    img_store, img_paths = FaissStore.load(cfg.index_path_obj, cfg.metadata_path_obj, cfg.index)
    cap_store: FaissStore | None = None
    captions: dict[str, str] = {}
    if cfg.retrieval.use_captions and cfg.caption_index_path_obj.exists():
        meta = cfg.caption_index_path_obj.with_name("caption_meta.json")
        if meta.exists():
            try:
                cap_store, cap_paths = FaissStore.load(cfg.caption_index_path_obj, meta, cfg.index)
            except Exception as exc:
                log.warning("could not load caption index: %s", exc)
        if cfg.caption_path_obj.exists():
            try:
                captions = json.loads(cfg.caption_path_obj.read_text(encoding="utf-8"))
            except Exception as exc:
                log.warning("could not read captions cache: %s", exc)
    return LoadedIndexes(
        image_store=img_store,
        image_paths=img_paths,
        caption_store=cap_store,
        captions=captions,
    )


def search(
    query: str,
    cfg: Config,
    loaded: LoadedIndexes | None = None,
    model: ClipModel | None = None,
) -> list[SearchResult]:
    """End-to-end retrieval: image + caption similarity, optional re-rank."""
    if loaded is None:
        loaded = load_indexes(cfg)
    if model is None:
        model = ClipModel.get(cfg.model)

    query_emb = model.encode_text(query).cpu().numpy().astype("float32")
    top_n = max(cfg.retrieval.rerank_top_n, cfg.retrieval.top_k)

    img_scores, img_indices = loaded.image_store.search(query_emb, top_n)

    candidates: dict[int, dict[str, float]] = {}
    for s, i in zip(img_scores[0], img_indices[0]):
        i_int = int(i)
        if i_int < 0 or i_int >= len(loaded.image_paths):
            continue
        entry = candidates.get(i_int)
        if entry is None:
            candidates[i_int] = {"image": float(s), "caption": 0.0}
        else:
            entry["image"] = float(s)

    if loaded.caption_store is not None:
        cap_scores, cap_indices = loaded.caption_store.search(query_emb, top_n)
        for s, ci in zip(cap_scores[0], cap_indices[0]):
            ci_int = int(ci)
            if ci_int < 0 or ci_int >= len(loaded.image_paths):
                continue
            entry = candidates.get(ci_int)
            if entry is None:
                candidates[ci_int] = {"image": 0.0, "caption": float(s)}
            else:
                entry["caption"] = max(entry["caption"], float(s))

    if not candidates:
        return []

    hybrid: dict[int, float] = {
        i: cfg.retrieval.image_weight * v["image"] + cfg.retrieval.caption_weight * v["caption"]
        for i, v in candidates.items()
    }

    rerank_scores: dict[int, float] = {}
    if cfg.retrieval.use_reranker and len(hybrid) > cfg.retrieval.top_k:
        top_idx = sorted(hybrid.keys(), key=lambda x: hybrid[x], reverse=True)[:top_n]
        candidate_pairs = [
            (i, loaded.captions.get(str(loaded.image_paths[i]), "") or "") for i in top_idx
        ]
        reranked = rerank(query, candidate_pairs, cfg.rerank, top_n)
        rerank_scores = {r.key: r.score for r in reranked}
        final = {
            i: 0.5 * hybrid[i] + 0.5 * rerank_scores.get(i, hybrid[i])
            for i in hybrid
        }
    else:
        final = hybrid

    sorted_idx = sorted(final.keys(), key=lambda x: final[x], reverse=True)[: cfg.retrieval.top_k]
    results: list[SearchResult] = []
    for rank, i in enumerate(sorted_idx, start=1):
        path = loaded.image_paths[i]
        results.append(
            SearchResult(
                path=path,
                score=float(final[i]),
                rank=rank,
                image_score=float(candidates[i]["image"]),
                caption_score=float(candidates[i]["caption"]),
                rerank_score=rerank_scores.get(i),
                caption=loaded.captions.get(str(path)),
            )
        )
    return results


def search_with_breakdown(
    query: str,
    cfg: Config,
    loaded: LoadedIndexes | None = None,
    model: ClipModel | None = None,
) -> dict[str, Any]:
    """Search and return both top-k results and full score breakdown for analysis."""
    if loaded is None:
        loaded = load_indexes(cfg)
    if model is None:
        model = ClipModel.get(cfg.model)
    results = search(query, cfg, loaded=loaded, model=model)
    return {
        "query": query,
        "results": [
            {
                "rank": r.rank,
                "path": str(r.path),
                "score": r.score,
                "image_score": r.image_score,
                "caption_score": r.caption_score,
                "rerank_score": r.rerank_score,
                "caption": r.caption,
            }
            for r in results
        ],
    }
