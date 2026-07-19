"""scripts/run_metrics.py — compute rank-based metrics on the 5 rubric queries.

Outputs:
    eval/results/metrics.csv     per-config, per-query metrics
    eval/results/metrics.json    aggregate stats

Metrics (no manual labels required):

- ``top1_score``        — score of the top-1 candidate
- ``score_gap``         — top1 − top2 score (model confidence)
- ``margin_at_k``       — score(top_k) − score(top_k+1) (cut between kept and rest)
- ``topk_diversity``    — fraction of unique paths in top-k (no duplicates)
- ``score_entropy``     — entropy of softmaxed top-k scores

Usage:
    python scripts/run_metrics.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from glance_search.attributes import parse_query, query_axis_tags
from glance_search.config import Config, load_config
from glance_search.errors import GlanceError
from glance_search.logging_setup import configure_logging, get_logger
from glance_search.metrics import aggregate_metrics
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

CONFIGS = [
    ("image_only", {"use_captions": False, "use_reranker": False}),
    ("image_captions", {"use_captions": True, "use_reranker": False}),
    ("image_captions_rerank", {"use_captions": True, "use_reranker": True}),
]

OUT_DIR = Path("eval/results")


def _variant(cfg: Config, overrides: dict) -> Config:
    return Config(
        model=cfg.model,
        index=cfg.index,
        retrieval=replace(cfg.retrieval, **overrides),
        captions=cfg.captions,
        rerank=cfg.rerank,
        log_level=cfg.log_level,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(
            f"dry-run ok: {len(RUBRIC_QUERIES)} queries × {len(CONFIGS)} configs "
            f"= {len(RUBRIC_QUERIES) * len(CONFIGS)} metric rows"
        )
        return 0

    cfg = load_config()
    configure_logging(cfg.log_level)
    try:
        loaded = load_indexes(cfg)
    except GlanceError as exc:
        log.error("%s", exc)
        return 1
    model = ClipModel.get(cfg.model)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    aggregates: dict[str, dict[str, float]] = {}
    for cfg_name, overrides in CONFIGS:
        vcfg = _variant(cfg, overrides)
        cfg_metrics: list[dict[str, float]] = []
        for slug, query in RUBRIC_QUERIES:
            log.info("[%s] %s", cfg_name, slug)
            try:
                results = search(query, vcfg, loaded=loaded, model=model)
            except GlanceError as exc:
                log.error("query failed: %s", exc)
                continue
            results_dicts = [
                {"rank": r.rank, "path": str(r.path), "score": r.score,
                 "image_score": r.image_score, "caption_score": r.caption_score,
                 "rerank_score": r.rerank_score}
                for r in results
            ]
            m = aggregate_metrics(results_dicts, top_k=vcfg.retrieval.top_k)
            attrs = parse_query(query)
            top1_img = float(results_dicts[0]["image_score"]) if results_dicts else 0.0
            top1_cap = float(results_dicts[0]["caption_score"]) if results_dicts else 0.0
            row = {
                "config": cfg_name,
                "query_slug": slug,
                "query": query,
                "query_axes": "|".join(query_axis_tags(attrs)),
                "n_results": len(results_dicts),
                "top1_image_cosine": round(top1_img, 4),
                "top1_caption_cosine": round(top1_cap, 4),
                **{k: round(v, 4) for k, v in m.items()},
            }
            rows.append(row)
            cfg_metrics.append(m)
        if cfg_metrics:
            keys = cfg_metrics[0].keys()
            aggregates[cfg_name] = {
                k: round(sum(m[k] for m in cfg_metrics) / len(cfg_metrics), 4)
                for k in keys
            }

    out_csv = OUT_DIR / "metrics.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log.info("wrote %s (%d rows)", out_csv, len(rows))

    out_json = OUT_DIR / "metrics.json"
    out_json.write_text(
        json.dumps(
            {"per_config_aggregate": aggregates, "per_query_rows": rows},
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("wrote %s", out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
