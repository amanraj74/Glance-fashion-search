"""Retriever CLI - text query -> top-k matching images.

Usage:
    python retriever/search.py
    python retriever/search.py --query "a red tie and a white shirt"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from glance_search.config import Config, load_config
from glance_search.errors import GlanceError
from glance_search.logging_setup import configure_logging, get_logger
from glance_search.model import ClipModel
from glance_search.pipeline import load_indexes, search

log = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search the FAISS index.")
    parser.add_argument("--query", "-q", help="natural-language query (omit to be prompted)")
    parser.add_argument("--top-k", "-k", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> int:
    cfg = load_config()
    args = _parse_args()
    if args.top_k is not None:
        from dataclasses import replace
        cfg = Config(
            model=cfg.model,
            index=cfg.index,
            retrieval=replace(cfg.retrieval, top_k=args.top_k),
            captions=cfg.captions,
            rerank=cfg.rerank,
            log_level=cfg.log_level,
        )
    if args.model or args.pretrained or args.device:
        from dataclasses import replace as dc_replace
        cfg = Config(
            model=dc_replace(
                cfg.model,
                name=args.model or cfg.model.name,
                pretrained=args.pretrained or cfg.model.pretrained,
                device=args.device or cfg.model.device,
            ),
            index=cfg.index,
            retrieval=cfg.retrieval,
            captions=cfg.captions,
            rerank=cfg.rerank,
            log_level=cfg.log_level,
        )
    configure_logging(cfg.log_level)
    try:
        loaded = load_indexes(cfg)
    except GlanceError as exc:
        log.error("%s; run python indexer/build_index.py first", exc)
        return 1
    model = ClipModel.get(cfg.model)
    query = args.query or input("Enter your query: ").strip()
    if not query:
        log.error("empty query")
        return 1
    hits = search(query, cfg, loaded=loaded, model=model)
    if not hits:
        print("\nNo results.\n")
        return 0
    print("\nTop results:\n")
    for h in hits:
        print(f"{h.rank}. {h.path} | score={h.score:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
