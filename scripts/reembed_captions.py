"""scripts/reembed_captions.py - re-embed existing BLIP captions with the current model.

Use this when you change the embedding backend and want to rebuild ONLY the
caption index without re-running BLIP on all images.

    python scripts/reembed_captions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from glance_search.config import load_config
from glance_search.errors import GlanceError
from glance_search.index_store import FaissStore
from glance_search.logging_setup import configure_logging, get_logger
from glance_search.model import ClipModel

log = get_logger(__name__)


def main() -> int:
    cfg = load_config()
    configure_logging(cfg.log_level)

    if not cfg.caption_path_obj.exists():
        log.error("missing captions cache: %s", cfg.caption_path_obj)
        log.error("run scripts/build_caption_index.py first to generate captions.")
        return 1

    import json
    captions = json.loads(cfg.caption_path_obj.read_text(encoding="utf-8"))
    if not captions:
        log.error("captions cache is empty")
        return 1
    log.info("loaded %d cached captions from %s", len(captions), cfg.caption_path_obj)

    image_paths = sorted(
        p for p in (cfg.image_dir_path).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    )
    log.info("found %d images under %s", len(image_paths), cfg.image_dir_path)

    pairs = [(p, captions[str(p)]) for p in image_paths if captions.get(str(p), "").strip()]
    if not pairs:
        log.error("no captioned images to embed")
        return 1
    log.info("embedding %d captions with %s", len(pairs), cfg.model.name)

    model = ClipModel.get(cfg.model)
    paths = [p for p, _ in pairs]
    texts = [t for _, t in pairs]
    with __import__("torch").no_grad():
        feats = model.encode_text(texts)
    embeddings = feats.cpu().numpy().astype("float32")

    store = FaissStore(dim=embeddings.shape[1], cfg=cfg.index)
    store.build(embeddings)
    meta_path = cfg.caption_index_path_obj.with_name("caption_meta.json")
    store.save(paths, cfg.caption_index_path_obj, meta_path)
    log.info("wrote %s (dim=%d, rows=%d)", cfg.caption_index_path_obj, embeddings.shape[1], len(paths))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except GlanceError as exc:
        log.error("%s", exc)
        sys.exit(1)
