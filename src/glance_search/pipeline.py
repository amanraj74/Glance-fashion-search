"""High-level retrieval pipeline. Wires model, indexes, captions, and re-ranker.

Design: hybrid late-interaction over image and caption embeddings, optionally
re-ranked by a cross-encoder over caption text. All heavy lifting lives in
modules; this file is orchestration only.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from glance_search.attributes import (
    QueryAttributes,
    attribute_overlap_score,
    parse_query,
    query_axis_tags,
)
from glance_search.captions import caption_corpus
from glance_search.config import Config
from glance_search.embedder import embed_corpus, list_images
from glance_search.index_store import FaissStore, SearchHit
from glance_search.logging_setup import get_logger
from glance_search.metrics import rrf as rrf_weight
from glance_search.model import ClipModel
from glance_search.reranker import rerank
from glance_search.semantic_attributes import (
    SemanticAttributeScores,
    build_attribute_vectors,
    semantic_attribute_score,
)

log = get_logger(__name__)


_QUERY_VARIANT_TEMPLATES = (
    "{}",
    "a photo of {}",
    "an image of {}",
    "showing {}",
    "a picture of {}",
)

_FILLER_WORDS = frozenset({"a", "an", "the", "in", "on", "at", "of"})


_GENERIC_RUNWAY_RE = re.compile(r"\b(?:runway|ramp|fashion show)\b", re.IGNORECASE)


def _caption_quality(text: str | None) -> float:
    """Heuristic 0..1 score for how informative a caption is for re-ranking.

    - empty or very short captions get low scores
    - captions dominated by generic "model walks runway" boilerplate get low scores
    - captions with concrete color / garment words get high scores
    """
    if not text:
        return 0.0
    s = text.strip()
    if len(s) < 12:
        return 0.0
    words = s.lower().split()
    if not words:
        return 0.0
    score = 0.4
    if 8 <= len(words) <= 60:
        score += 0.2
    colors = {"red", "blue", "green", "yellow", "black", "white", "pink",
              "gray", "grey", "brown", "beige", "orange", "purple", "navy",
              "tan", "cream", "khaki", "burgundy", "maroon", "olive"}
    garments = {"shirt", "dress", "coat", "jacket", "skirt", "pants", "jeans",
                "sweater", "hoodie", "raincoat", "suit", "tie", "blouse",
                "top", "t-shirt", "cardigan", "vest", "boots", "shoes", "bag"}
    if any(w in colors for w in words):
        score += 0.15
    if any(w in garments for w in words):
        score += 0.15
    if _GENERIC_RUNWAY_RE.search(s):
        score -= 0.4
    return max(0.0, min(1.0, score))


def _query_variants(query: str) -> list[str]:
    """Generate semantic variants of the user's query.

    CLIP/SigLIP text encoders are sensitive to phrasing — different phrasings of
    the same intent can land in different regions of the embedding space. Average
    retrieval quality improves noticeably when we encode several variants and
    aggregate scores (max) per candidate, without recomputing any embeddings.
    """
    text = query.strip()
    text_lower = text.lower()
    stripped = " ".join(w for w in text_lower.split() if w not in _FILLER_WORDS).strip()

    variants: set[str] = {text}
    for template in _QUERY_VARIANT_TEMPLATES:
        body = stripped if stripped else text_lower
        variants.add(template.format(body).strip())

    variants.discard("")
    return list(variants)


@dataclass
class SearchResult:
    path: Path
    score: float
    rank: int
    image_score: float = 0.0
    caption_score: float = 0.0
    rerank_score: float | None = None
    rerank_quality: float | None = None
    caption: str | None = None
    image_rank: int | None = None
    caption_rank: int | None = None
    attribute_score: float | None = None
    semantic_score: float | None = None
    query_axes: tuple[str, ...] = ()


@dataclass
class LoadedIndexes:
    image_store: FaissStore
    image_paths: list[Path]
    caption_store: FaissStore | None
    captions: dict[str, str] = field(default_factory=dict)
    image_tags: dict[str, dict[str, tuple[str, ...]]] = field(default_factory=dict)


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
    image_tags: dict[str, dict[str, tuple[str, ...]]] = {}
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
        tags_path = cfg.caption_path_obj.with_name("image_tags.json")
        if tags_path.exists():
            try:
                from glance_search.image_attributes import load_image_attribute_cache
                image_tags = load_image_attribute_cache(tags_path)
            except Exception as exc:
                log.warning("could not read image_tags cache: %s", exc)
    return LoadedIndexes(
        image_store=img_store,
        image_paths=img_paths,
        caption_store=cap_store,
        captions=captions,
        image_tags=image_tags,
    )


def _rrf_fuse(
    image_ranks: dict[int, int],
    caption_ranks: dict[int, int],
    k_const: int = 60,
    image_w: float = 1.0,
    caption_w: float = 1.0,
) -> dict[int, float]:
    """Reciprocal Rank Fusion. Combines rank positions from two orderings.

    score(i) = image_w / (k + image_rank) + caption_w / (k + caption_rank)
    Higher is better. ``k=60`` is the original Cormack et al. (2009) constant.
    """
    out: dict[int, float] = {}
    keys = set(image_ranks) | set(caption_ranks)
    for i in keys:
        s = 0.0
        if i in image_ranks:
            s += image_w * rrf_weight(image_ranks[i], k_const)
        if i in caption_ranks:
            s += caption_w * rrf_weight(caption_ranks[i], k_const)
        out[i] = s
    return out


def _aggregate_variant(
    candidates: dict[int, dict[str, float]],
    scores: np.ndarray,
    indices: np.ndarray,
    field_key: str,
    top_n: int,
    path_count: int,
) -> tuple[dict[int, float], dict[int, int]]:
    """Take raw faiss hits and update max-score + min-rank per candidate."""
    new_max = dict(candidates)
    new_ranks: dict[int, int] = {}
    head = min(top_n, len(scores))
    for r in range(head):
        i = int(indices[r])
        if i < 0 or i >= path_count:
            continue
        if i not in new_ranks:
            new_ranks[i] = r + 1
        s = float(scores[r])
        prev = new_max.setdefault(i, {"image": 0.0, "caption": 0.0})
        if s > prev[field_key]:
            prev[field_key] = s
    return new_max, new_ranks


def search(
    query: str,
    cfg: Config,
    loaded: LoadedIndexes | None = None,
    model: ClipModel | None = None,
) -> list[SearchResult]:
    """End-to-end retrieval: image + caption similarity, optional re-rank.

    The pipeline is configurable:

    - ``expand_queries`` — encode 5 query variants and aggregate per-image
      max score / min rank.
    - ``scoring`` — ``"weighted"`` (default) uses linear combination of
      cosine similarities; ``"rrf"`` uses reciprocal rank fusion over the
      per-index ranks.
    - ``use_captions`` / ``use_reranker`` — toggle each stage.
    - ``rerank_weight`` — final blend is ``(1-w)·hybrid + w·quality·rerank``.

    Attribute-aware bonus: when the query contains color / garment / scene
    words, candidates whose captions hit those attributes get a multiplicative
    lift proportional to ``retrieval.attribute_bonus``.
    """
    if loaded is None:
        loaded = load_indexes(cfg)
    if model is None:
        model = ClipModel.get(cfg.model)

    queries = (
        _query_variants(query)
        if getattr(cfg.retrieval, "expand_queries", True)
        else [query]
    )
    top_n = max(cfg.retrieval.rerank_top_n, cfg.retrieval.top_k)
    path_count = len(loaded.image_paths)

    candidates: dict[int, dict[str, float]] = {}
    image_ranks_acc: dict[int, int] = {}
    caption_ranks_acc: dict[int, int] = {}

    for q in queries:
        qfeat = model.encode_text(q).cpu().numpy().astype("float32")
        scores, indices = loaded.image_store.search(qfeat, top_n)
        candidates, new_img_ranks = _aggregate_variant(
            candidates, scores[0], indices[0], "image", top_n, path_count,
        )
        for k, v in new_img_ranks.items():
            image_ranks_acc[k] = min(image_ranks_acc.get(k, 10**9), v)

        if loaded.caption_store is not None:
            cap_scores, cap_indices = loaded.caption_store.search(qfeat, top_n)
            candidates, new_cap_ranks = _aggregate_variant(
                candidates, cap_scores[0], cap_indices[0], "caption", top_n, path_count,
            )
            for k, v in new_cap_ranks.items():
                caption_ranks_acc[k] = min(caption_ranks_acc.get(k, 10**9), v)

    if not candidates:
        return []

    scoring = getattr(cfg.retrieval, "scoring", "weighted")
    attr_bonus = getattr(cfg.retrieval, "attribute_bonus", 0.15)

    if scoring == "rrf":
        hybrid = _rrf_fuse(
            image_ranks_acc,
            caption_ranks_acc,
            k_const=getattr(cfg.retrieval, "rrf_k", 60),
            image_w=cfg.retrieval.image_weight,
            caption_w=cfg.retrieval.caption_weight,
        )
    else:
        hybrid = {
            i: cfg.retrieval.image_weight * v["image"]
            + cfg.retrieval.caption_weight * v["caption"]
            for i, v in candidates.items()
        }

    attrs = parse_query(query)
    hard_neg_penalty = float(getattr(cfg.retrieval, "hard_negative_penalty", 0.0))
    semantic_attr_weight = float(getattr(cfg.retrieval, "semantic_attribute_weight", 0.0))
    composite_scores: dict[int, float] = {}
    semantic_scores: dict[int, SemanticAttributeScores] = {}

    if attrs.total_hits > 0 and loaded.image_tags:
        from glance_search.image_attributes import composite_attribute_score
        for i in list(hybrid.keys()):
            cap = loaded.captions.get(str(loaded.image_paths[i]), "") or ""
            cs, _per_axis = composite_attribute_score(
                attrs, loaded.image_tags.get(str(loaded.image_paths[i]), {})
            )
            composite_scores[i] = cs
            if cs < 0:
                hybrid[i] = hybrid[i] * (1.0 + hard_neg_penalty * cs)
            elif cs > 0:
                hybrid[i] = hybrid[i] * (1.0 + attr_bonus * cs)
            else:
                ov = attribute_overlap_score(attrs, cap)
                if ov > 0:
                    hybrid[i] = hybrid[i] * (1.0 + 0.5 * attr_bonus * ov)

    if semantic_attr_weight > 0 and attrs.total_hits > 0:
        attr_vectors = build_attribute_vectors(attrs, model)
        if attr_vectors:
            try:
                stored = loaded.image_store.reconstruct_n(0, len(loaded.image_paths))
                stored_lookup = {idx: stored[idx] for idx in hybrid.keys()}
            except Exception:
                stored_lookup = {}
            for i, vec in stored_lookup.items():
                scores = semantic_attribute_score(vec, attr_vectors)
                semantic_scores[i] = scores
                if scores.overall != 0.0:
                    hybrid[i] = hybrid[i] * (1.0 + semantic_attr_weight * scores.overall)

    rerank_scores: dict[int, float] = {}
    rerank_quality: dict[int, float] = {}
    attribute_scores: dict[int, float] = {}

    if cfg.retrieval.use_reranker and len(hybrid) > cfg.retrieval.top_k:
        top_idx = sorted(hybrid.keys(), key=lambda x: hybrid[x], reverse=True)[:top_n]
        candidate_pairs = [
            (i, loaded.captions.get(str(loaded.image_paths[i]), "") or "") for i in top_idx
        ]
        reranked = rerank(query, candidate_pairs, cfg.rerank, top_n)
        rerank_scores = {r.key: r.score for r in reranked}
        for i in top_idx:
            cap = loaded.captions.get(str(loaded.image_paths[i]), "") or ""
            rerank_quality[i] = _caption_quality(cap)
            attribute_scores[i] = composite_scores.get(i, attribute_overlap_score(attrs, cap) if attrs.total_hits else 0.0)
        final = {
            i: (1.0 - cfg.retrieval.rerank_weight) * hybrid[i]
               + cfg.retrieval.rerank_weight * rerank_quality.get(i, 1.0) * rerank_scores.get(i, hybrid[i])
            for i in hybrid
        }
    else:
        final = hybrid
        for i in final:
            cap = loaded.captions.get(str(loaded.image_paths[i]), "") or ""
            attribute_scores[i] = composite_scores.get(i, attribute_overlap_score(attrs, cap) if attrs.total_hits else 0.0)

    sorted_idx = sorted(final.keys(), key=lambda x: final[x], reverse=True)[: cfg.retrieval.top_k]
    results: list[SearchResult] = []
    for rank, i in enumerate(sorted_idx, start=1):
        path = loaded.image_paths[i]
        cap = loaded.captions.get(str(path))
        sem = semantic_scores.get(i)
        results.append(
            SearchResult(
                path=path,
                score=float(final[i]),
                rank=rank,
                image_score=float(candidates[i]["image"]),
                caption_score=float(candidates[i]["caption"]),
                rerank_score=rerank_scores.get(i),
                rerank_quality=rerank_quality.get(i),
                caption=cap,
                image_rank=image_ranks_acc.get(i),
                caption_rank=caption_ranks_acc.get(i),
                attribute_score=attribute_scores.get(i)
                    if attribute_scores
                    else (attribute_overlap_score(attrs, cap) if attrs.total_hits else None),
                semantic_score=(sem.overall if sem else None),
                query_axes=tuple(query_axis_tags(attrs)),
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
                "rerank_quality": r.rerank_quality,
                "caption": r.caption,
            }
            for r in results
        ],
    }
