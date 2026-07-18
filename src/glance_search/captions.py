"""BLIP image captioning. Offline, resumable, batched."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor

from glance_search.config import CaptionsConfig
from glance_search.errors import CaptionError
from glance_search.logging_setup import get_logger

log = get_logger(__name__)


def _load_model(model_name: str):
    try:
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name).eval()
    except Exception as exc:
        raise CaptionError(f"failed to load caption model {model_name}: {exc}") from exc
    return processor, model


def caption_corpus(
    image_paths: list[Path],
    cfg: CaptionsConfig,
    out_path: Path,
) -> dict[str, str]:
    """Generate one BLIP caption per image. Resumable: existing entries are kept."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    captions: dict[str, str] = {}
    if out_path.exists():
        try:
            captions = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            log.warning("could not read existing captions at %s; starting fresh", out_path)

    missing = [p for p in image_paths if str(p) not in captions]
    if not missing:
        log.info("all %d captions already cached at %s", len(image_paths), out_path)
        return captions

    processor, model = _load_model(cfg.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    for start in tqdm(range(0, len(missing), cfg.batch_size), desc="caption"):
        chunk = missing[start : start + cfg.batch_size]
        imgs = []
        good = []
        for p in chunk:
            try:
                imgs.append(Image.open(p).convert("RGB"))
                good.append(p)
            except Exception as exc:
                log.warning("skip %s: %s", p, exc)
        if not good:
            continue
        try:
            inputs = processor(images=imgs, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=cfg.max_new_tokens,
                    num_beams=cfg.num_beams,
                )
            decoded = [processor.decode(o, skip_special_tokens=True) for o in out]
            for p, text in zip(good, decoded):
                captions[str(p)] = text.strip()
        except Exception as exc:
            log.warning("caption batch failed (size=%d): %s", len(good), exc)
            for p in good:
                captions.setdefault(str(p), "")

        if (start // cfg.batch_size) % 10 == 0:
            out_path.write_text(json.dumps(captions, indent=2), encoding="utf-8")

    out_path.write_text(json.dumps(captions, indent=2), encoding="utf-8")
    log.info("wrote %d captions to %s", len(captions), out_path)
    return captions
