"""scripts/build_caption_index.py — generate BLIP captions and embed them.

Run after `python indexer/build_index.py`. Output:
    output/captions.json            {path: caption_text}
    output/captions.index           FAISS IndexFlatIP over caption embeddings
    output/caption_meta.json        paths aligned to caption index rows
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from glance_search.config import load_config
from glance_search.errors import GlanceError
from glance_search.logging_setup import configure_logging, get_logger
from glance_search.pipeline import build_caption_index

log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate BLIP captions and embed them.")
    parser.add_argument("--dry-run", action="store_true", help="load config and exit (no model download)")
    args = parser.parse_args()

    cfg = load_config()
    if args.dry_run:
        print(f"dry-run ok: caption model={cfg.captions.model} output={cfg.caption_path_obj}")
        return 0

    configure_logging(cfg.log_level)
    try:
        store, captions = build_caption_index(cfg)
    except GlanceError as exc:
        log.error("caption index build failed: %s", exc)
        return 1
    log.info("done: %d captions, index ntotal=%d", len(captions), store.ntotal)
    return 0


if __name__ == "__main__":
    sys.exit(main())
