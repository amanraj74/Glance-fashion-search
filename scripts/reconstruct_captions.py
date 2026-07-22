"""scripts/reconstruct_captions.py - rebuild captions.json from partial multi-prompt cache.

Used after a `--force` caption regen that was interrupted before finishing.
Joins the per-prompt variants into the legacy ``{path: text}`` format expected
by the rest of the pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from glance_search.captions import _join_captions
from glance_search.logging_setup import configure_logging, get_logger

log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi", default="output/captions_multi.json")
    parser.add_argument("--out", default="output/captions.json")
    args = parser.parse_args()

    configure_logging("INFO")
    multi_path = Path(args.multi)
    out_path = Path(args.out)

    if not multi_path.exists():
        log.error("multi-prompt cache not found: %s", multi_path)
        return 1

    raw = json.loads(multi_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "captions" in raw:
        per_image = raw["captions"]
    else:
        per_image = raw
    log.info("loaded %d multi-prompt entries", len(per_image))

    joined = {p: _join_captions(cs) for p, cs in per_image.items() if cs}
    out_path.write_text(json.dumps(joined, indent=2), encoding="utf-8")
    log.info("wrote %d joined captions to %s", len(joined), out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())