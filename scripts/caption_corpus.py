"""scripts/caption_corpus.py — generate BLIP captions for the image corpus.

Usage:
    python scripts/caption_corpus.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from glance_search.captions import caption_corpus
from glance_search.config import load_config
from glance_search.embedder import list_images
from glance_search.errors import GlanceError
from glance_search.logging_setup import configure_logging, get_logger

log = get_logger(__name__)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Generate BLIP captions for all images.")
    parser.add_argument("--dry-run", action="store_true", help="load config and exit (no model download)")
    args = parser.parse_args()

    cfg = load_config()
    if args.dry_run:
        print(f"dry-run ok: model={cfg.captions.model} image_dir={cfg.index.image_dir}")
        return 0
    configure_logging(cfg.log_level)
    try:
        paths = list_images(cfg.index.image_dir)
    except GlanceError as exc:
        log.error("%s", exc)
        return 1
    log.info("captioning %d images with %s", len(paths), cfg.captions.model)
    try:
        captions = caption_corpus(paths, cfg.captions, cfg.caption_path_obj)
    except GlanceError as exc:
        log.error("captioning failed: %s", exc)
        return 1
    log.info("done: %d captions at %s", len(captions), cfg.caption_path_obj)
    return 0


if __name__ == "__main__":
    sys.exit(main())
