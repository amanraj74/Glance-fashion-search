"""Image embedding pipeline. Batched, L2-normalized, with per-file error handling."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from glance_search.errors import EmbeddingError
from glance_search.logging_setup import get_logger
from glance_search.model import ClipModel

log = get_logger(__name__)

VALID_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def list_images(root: Path | str, exts: Iterable[str] = VALID_EXTS) -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise EmbeddingError(f"image directory does not exist: {root}")
    ext_set = {e.lower() for e in exts}
    paths = sorted(p for p in root.iterdir() if p.suffix.lower() in ext_set and p.is_file())
    return paths


def embed_corpus(
    paths: list[Path],
    model: ClipModel,
    batch_size: int = 16,
) -> tuple[np.ndarray, list[Path]]:
    """Embed a corpus in batches, skipping unreadable images.

    Returns (N, D) float32 matrix and the path list that succeeded.
    """
    if not paths:
        raise EmbeddingError("no images to embed")
    if batch_size <= 0:
        raise EmbeddingError("batch_size must be > 0")

    kept: list[Path] = []
    rows: list[np.ndarray] = []
    for start in tqdm(range(0, len(paths), batch_size), desc="embed"):
        chunk = paths[start : start + batch_size]
        imgs: list[Image.Image] = []
        local: list[Path] = []
        for p in chunk:
            try:
                img = Image.open(p)
                img.load()
                imgs.append(img)
                local.append(p)
            except Exception as exc:
                log.warning("skip %s: %s", p, exc)
        if not imgs:
            continue
        try:
            feats = model.encode_images(imgs).cpu().numpy().astype("float32")
            rows.append(feats)
            kept.extend(local)
        except torch.cuda.OutOfMemoryError as exc:
            log.error("GPU OOM, reduce batch_size: %s", exc)
            raise EmbeddingError("GPU OOM during embedding") from exc
        except Exception as exc:
            log.warning("batch failed (size=%d): %s", len(imgs), exc)
    if not rows:
        raise EmbeddingError("no images were embedded successfully")
    return np.concatenate(rows, axis=0), kept


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def open_image(path: Path) -> Image.Image:
    img = Image.open(path)
    img.load()
    return img.convert("RGB")
