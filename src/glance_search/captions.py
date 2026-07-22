"""BLIP image captioning. Offline, resumable, batched, prompt-conditioned.

For each image we generate up to N captions using different decoder prompts. This
breaks the BLIP-base "default to \"a model walks the runway...\"\" pathology
(observed on 40%+ of a fashion catalogue) by conditioning on style/attribute/scene
prompts. Multiple captions also let downstream retrieval average several text
embeddings per image, which improves compositional matching.
"""

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


DEFAULT_PROMPTS: tuple[str, ...] = (
    "",
    "a fashion photo of a person wearing",
    "the clothing in this image is",
)


def _is_repetitive(text: str) -> bool:
    """Return True if ``text`` is degenerate (token repetition or too short).

    BLIP-base sometimes loops on long prompts and emits "person, person, person"
    or "catwalk catwalk catwalk". We reject those so they don't pollute the
    caption index and confuse the reranker.
    """
    s = text.lower().strip(" .,!?:;'\"")
    if not s or len(s) < 6:
        return True
    toks = s.split()
    if len(toks) < 3:
        return True
    bigrams = [(toks[i], toks[i + 1]) for i in range(len(toks) - 1)]
    bigram_counts = {}
    for b in bigrams:
        bigram_counts[b] = bigram_counts.get(b, 0) + 1
    max_repeat = max(bigram_counts.values()) if bigram_counts else 0
    if max_repeat >= max(3, len(toks) // 4):
        return True
    return False


def _clean_caption(text: str) -> str:
    text = text.strip()
    text = " ".join(text.split())
    if text and not text.endswith((".", "!", "?")):
        text += "."
    return text


def _load_model(model_name: str):
    try:
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name).eval()
    except Exception as exc:
        raise CaptionError(f"failed to load caption model {model_name}: {exc}") from exc
    return processor, model


def _generate_batch(
    processor,
    model,
    imgs: list[Image.Image],
    prompts: tuple[str, ...],
    max_new_tokens: int,
    num_beams: int,
    device: str,
) -> list[list[str]]:
    """Run BLIP on each image with each prompt. Returns [img][prompt] -> caption."""
    if not imgs:
        return []
    captions_per_image: list[list[str]] = [[] for _ in imgs]
    with torch.no_grad():
        pixel = processor(images=imgs, return_tensors="pt").pixel_values.to(device)
        for prompt in prompts:
            if prompt:
                tok = processor.tokenizer(
                    [prompt] * len(imgs),
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                input_ids = tok.input_ids
                attention_mask = tok.attention_mask
            else:
                input_ids = None
                attention_mask = None
            out = model.generate(
                pixel_values=pixel,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            for i, o in enumerate(out):
                text = processor.decode(o, skip_special_tokens=True).strip()
                text = _clean_caption(text)
                if text and not _is_repetitive(text):
                    captions_per_image[i].append(text)
    return captions_per_image


def _join_captions(captions: list[str]) -> str:
    """Deduplicate near-identical captions, keep order, join with '. '."""
    seen: set[str] = set()
    out: list[str] = []
    for c in captions:
        key = c.lower().strip(" .,!?:;'\"")
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(c)
    return ". ".join(out)


def _serialize(captions_obj, version: int = 2) -> str:
    return json.dumps({"version": version, "captions": captions_obj}, indent=2)


def _deserialize(text: str) -> tuple[dict[str, list[str]], int]:
    """Return ({path: [captions...]}, schema_version). Handles old + new shapes."""
    raw = json.loads(text)
    if isinstance(raw, dict) and "version" in raw and "captions" in raw:
        return raw["captions"], int(raw["version"])
    if isinstance(raw, dict):
        return {k: [v] for k, v in raw.items()}, 1
    return {}, 1


def caption_corpus(
    image_paths: list[Path],
    cfg: CaptionsConfig,
    out_path: Path,
) -> dict[str, str]:
    """Generate one or more BLIP captions per image. Resumable.

    Output is the ``primary`` caption (joined, deduplicated multi-prompt string) so
    downstream code that reads ``captions.json`` keeps working. The per-prompt list
    is preserved under the same key, separated by ``||`` markers, in a sibling
    file ``captions_multi.json`` for callers that want all variants.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    captions: dict[str, list[str]] = {}
    if out_path.exists():
        try:
            captions, _ = _deserialize(out_path.read_text(encoding="utf-8"))
        except Exception:
            log.warning("could not read existing captions at %s; starting fresh", out_path)

    missing = [p for p in image_paths if str(p) not in captions or not captions[str(p)]]
    if not missing:
        log.info("all %d captions already cached at %s", len(image_paths), out_path)
        joined = {p: _join_captions(cs) for p, cs in captions.items()}
        _write_multi(captions, out_path)
        return joined

    processor, model = _load_model(cfg.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    prompts = DEFAULT_PROMPTS
    log.info(
        "captioning %d images with %d prompts on %s (model=%s)",
        len(missing), len(prompts), device, cfg.model,
    )

    flush_every = max(1, 10)
    for start in tqdm(range(0, len(missing), cfg.batch_size), desc="caption"):
        chunk = missing[start : start + cfg.batch_size]
        imgs: list[Image.Image] = []
        good: list[Path] = []
        for p in chunk:
            try:
                imgs.append(Image.open(p).convert("RGB"))
                good.append(p)
            except Exception as exc:
                log.warning("skip %s: %s", p, exc)
        if not good:
            continue
        try:
            per_image = _generate_batch(
                processor, model, imgs, prompts,
                cfg.max_new_tokens, cfg.num_beams, device,
            )
            for p, cs in zip(good, per_image):
                captions[str(p)] = [c for c in cs if c]
        except Exception as exc:
            log.warning("caption batch failed (size=%d): %s", len(good), exc)
            for p in good:
                captions.setdefault(str(p), [])

        if (start // cfg.batch_size) % flush_every == 0:
            _write_multi(captions, out_path)

    _write_multi(captions, out_path)
    joined = {p: _join_captions(cs) for p, cs in captions.items() if cs}
    out_path.write_text(json.dumps(joined, indent=2), encoding="utf-8")
    log.info("wrote %d captions (multi-prompt) to %s", len(joined), out_path)
    return joined


def _write_multi(captions: dict[str, list[str]], out_path: Path) -> None:
    """Write raw multi-caption cache next to the primary captions.json."""
    multi_path = out_path.with_name(out_path.stem + "_multi.json")
    multi_path.write_text(_serialize(captions), encoding="utf-8")
