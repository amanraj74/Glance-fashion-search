"""scripts/build_image_attribute_cache.py - build per-image attribute tags."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from glance_search.image_attributes import build_image_attribute_cache
from glance_search.logging_setup import configure_logging, get_logger

log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build per-image attribute tag cache.")
    parser.add_argument(
        "--captions",
        default="output/captions.json",
        help="Path to the captions cache.",
    )
    parser.add_argument(
        "--out",
        default="output/image_tags.json",
        help="Where to write the tag cache.",
    )
    args = parser.parse_args()

    configure_logging("INFO")
    cap_path = Path(args.captions)
    out_path = Path(args.out)

    if not cap_path.exists():
        log.error("captions cache not found: %s", cap_path)
        return 1

    captions = json.loads(cap_path.read_text(encoding="utf-8"))
    log.info("loaded %d captions from %s", len(captions), cap_path)
    cache = build_image_attribute_cache(captions, out_path)

    from glance_search.image_attributes import collect_axis_inventory
    inv = collect_axis_inventory(cache)
    log.info("axis inventory:")
    for axis, counts in inv.items():
        top5 = sorted(counts.items(), key=lambda x: -x[1])[:5]
        log.info("  %s: %s", axis, top5)
    return 0


if __name__ == "__main__":
    sys.exit(main())