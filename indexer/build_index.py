"""Indexer CLI - embeds images and persists a FAISS index.

Usage:
    python indexer/build_index.py
    python indexer/build_index.py --model ViT-B-32 --pretrained laion2b_s34b_b79k
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from glance_search.config import Config, ModelConfig, load_config
from glance_search.errors import GlanceError
from glance_search.logging_setup import configure_logging, get_logger
from glance_search.pipeline import build_image_index

log = get_logger(__name__)


def _parse_args(cfg: Config) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a FAISS index over the image corpus.")
    parser.add_argument("--model", default=cfg.model.name, help="open_clip model name")
    parser.add_argument("--pretrained", default=cfg.model.pretrained, help="pretrained tag")
    parser.add_argument("--device", default=cfg.model.device, help="cpu | cuda | auto")
    parser.add_argument("--backend", choices=("flat", "ivfflat"), default=cfg.index.backend)
    return parser.parse_args()


def main() -> int:
    cfg = load_config()
    args = _parse_args(cfg)
    cfg = Config(
        model=ModelConfig(
            name=args.model,
            pretrained=args.pretrained,
            device=args.device,
        ),
        index=cfg.index.__class__(**{**cfg.index.__dict__, "backend": args.backend}),
        retrieval=cfg.retrieval,
        captions=cfg.captions,
        rerank=cfg.rerank,
        log_level=cfg.log_level,
    )
    configure_logging(cfg.log_level)
    log.info(
        "starting index build model=%s pretrained=%s device=%s backend=%s",
        cfg.model.name, cfg.model.pretrained, cfg.model.device, cfg.index.backend,
    )
    try:
        store, kept = build_image_index(cfg)
    except GlanceError as exc:
        log.error("index build failed: %s", exc)
        return 1
    log.info("done. index ntotal=%d kept=%d", store.ntotal, len(kept))
    return 0


if __name__ == "__main__":
    sys.exit(main())
