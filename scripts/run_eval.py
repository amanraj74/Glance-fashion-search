"""scripts/run_eval.py — run the 5 rubric evaluation queries and save grids.

Usage:
    python scripts/run_eval.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from PIL import Image, ImageDraw

from glance_search.config import Config, load_config
from glance_search.errors import GlanceError
from glance_search.index_store import FaissStore
from glance_search.logging_setup import configure_logging, get_logger
from glance_search.model import ClipModel
from glance_search.pipeline import load_indexes, search

log = get_logger(__name__)

RUBRIC_QUERIES = [
    ("01_yellow_raincoat", "A person in a bright yellow raincoat."),
    ("02_business_office", "Professional business attire inside a modern office."),
    ("03_blue_shirt_park", "Someone wearing a blue shirt sitting on a park bench."),
    ("04_casual_city", "Casual weekend outfit for a city walk."),
    ("05_red_tie_white_shirt", "A red tie and a white shirt in a formal setting."),
]

OUT_DIR = Path("eval/results")


def render_grid(slug: str, query: str, hits, top_k: int, out: Path) -> None:
    cell = 256
    pad = 8
    label_h = 28
    images = []
    labels = []
    for h in hits[:top_k]:
        try:
            img = Image.open(h.path).convert("RGB").resize((cell, cell))
            images.append(img)
            labels.append(f"#{h.rank}  {h.score:.3f}")
        except Exception as exc:
            log.warning("could not load %s: %s", h.path, exc)
    if not images:
        return
    width = cell * len(images) + pad * (len(images) + 1)
    height = cell + pad * 2 + label_h * 2
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, pad), f"Q: {query}", fill="black")
    for i, (im, label) in enumerate(zip(images, labels)):
        x = pad + i * (cell + pad)
        canvas.paste(im, (x, pad + label_h))
        draw.text((x, pad + label_h + cell + 4), label, fill="black")
    canvas.save(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the 5 rubric evaluation queries.")
    parser.add_argument("--dry-run", action="store_true", help="load config and exit (no model download)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    if args.dry_run:
        print(f"dry-run ok: model={cfg.model.name} top_k={cfg.retrieval.top_k}")
        return 0
    configure_logging(cfg.log_level)
    try:
        loaded = load_indexes(cfg)
    except GlanceError as exc:
        log.error("%s; build the index first", exc)
        return 1
    model = ClipModel.get(cfg.model)
    summary = {}
    for slug, query in RUBRIC_QUERIES:
        log.info("running query: %s", slug)
        try:
            hits = search(query, cfg, loaded=loaded, model=model)
        except GlanceError as exc:
            log.error("query %s failed: %s", slug, exc)
            continue
        results = [{"rank": h.rank, "path": str(h.path), "score": h.score} for h in hits]
        (OUT_DIR / f"{slug}.json").write_text(
            json.dumps({"slug": slug, "query": query, "results": results}, indent=2),
            encoding="utf-8",
        )
        render_grid(slug, query, hits, cfg.retrieval.top_k, OUT_DIR / f"{slug}.png")
        summary[slug] = {
            "query": query,
            "top_score": hits[0].score if hits else None,
        }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("wrote eval to %s", OUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
