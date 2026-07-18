"""scripts/run_ablation.py — A/B test pipeline configurations on the 5 rubric queries.

Generates `eval/results/ablation.csv` with rows:
    config, query, top1_score, top5_score, mean_image_score, mean_caption_score
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from glance_search.config import Config, load_config
from glance_search.errors import GlanceError
from glance_search.logging_setup import configure_logging, get_logger
from glance_search.pipeline import load_indexes, search_with_breakdown
from glance_search.model import ClipModel

log = get_logger(__name__)

RUBRIC_QUERIES = [
    ("01_yellow_raincoat", "A person in a bright yellow raincoat."),
    ("02_business_office", "Professional business attire inside a modern office."),
    ("03_blue_shirt_park", "Someone wearing a blue shirt sitting on a park bench."),
    ("04_casual_city", "Casual weekend outfit for a city walk."),
    ("05_red_tie_white_shirt", "A red tie and a white shirt in a formal setting."),
]

CONFIGS = [
    ("image_only", {"use_captions": False, "use_reranker": False, "caption_weight": 0.0, "image_weight": 1.0}),
    ("image_captions", {"use_captions": True, "use_reranker": False, "caption_weight": 0.4, "image_weight": 0.6}),
    ("image_captions_rerank", {"use_captions": True, "use_reranker": True, "caption_weight": 0.4, "image_weight": 0.6}),
]

OUT_PATH = Path("eval/results/ablation.csv")


def _variant(cfg: Config, overrides: dict) -> Config:
    return Config(
        model=cfg.model,
        index=cfg.index,
        retrieval=replace(
            cfg.retrieval,
            use_captions=overrides["use_captions"],
            use_reranker=overrides["use_reranker"],
            caption_weight=overrides["caption_weight"],
            image_weight=overrides["image_weight"],
        ),
        captions=cfg.captions,
        rerank=cfg.rerank,
        log_level=cfg.log_level,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(f"dry-run ok: {len(RUBRIC_QUERIES)} queries × {len(CONFIGS)} configs = {len(RUBRIC_QUERIES) * len(CONFIGS)} rows")
        return 0

    cfg = load_config()
    configure_logging(cfg.log_level)
    try:
        loaded = load_indexes(cfg)
    except GlanceError as exc:
        log.error("%s", exc)
        return 1
    model = ClipModel.get(cfg.model)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for cfg_name, overrides in CONFIGS:
        if overrides["use_captions"] and loaded.caption_store is None:
            log.warning("caption index missing; skipping config=%s", cfg_name)
            continue
        vcfg = _variant(cfg, overrides)
        for slug, query in RUBRIC_QUERIES:
            log.info("[%s] %s", cfg_name, slug)
            breakdown = search_with_breakdown(query, vcfg, loaded=loaded, model=model)
            results = breakdown["results"]
            top1 = results[0]["score"] if results else 0.0
            top5 = sum(r["score"] for r in results) / max(len(results), 1)
            mean_img = sum(r["image_score"] for r in results) / max(len(results), 1)
            mean_cap = sum(r["caption_score"] for r in results) / max(len(results), 1)
            rows.append({
                "config": cfg_name,
                "query_slug": slug,
                "query": query,
                "top1_score": round(top1, 4),
                "top5_mean_score": round(top5, 4),
                "mean_image_score": round(mean_img, 4),
                "mean_caption_score": round(mean_cap, 4),
            })

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log.info("wrote %s (%d rows)", OUT_PATH, len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
