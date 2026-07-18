"""OpenCLIP-style model wrapper. Cached per process."""

from __future__ import annotations

from pathlib import Path

import open_clip
import torch
from PIL import Image

from glance_search.config import ModelConfig
from glance_search.errors import ModelLoadError
from glance_search.logging_setup import get_logger

log = get_logger(__name__)


def _resolve_device(pref: str) -> str:
    if pref != "auto":
        return pref
    return "cuda" if torch.cuda.is_available() else "cpu"


class ClipModel:
    """Wraps an OpenCLIP model + preprocess + tokenizer."""

    _cache: dict[tuple[str, str, str], "ClipModel"] = {}

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.device = _resolve_device(cfg.device)
        try:
            log.info("loading model name=%s pretrained=%s device=%s", cfg.name, cfg.pretrained, self.device)
            model, _, preprocess = open_clip.create_model_and_transforms(
                cfg.name,
                pretrained=cfg.pretrained,
                cache_dir=cfg.cache_dir,
            )
            tokenizer_name = cfg.name if "hf-hub:" not in cfg.name else cfg.name.split("/", 1)[1]
            try:
                self.tokenizer = open_clip.get_tokenizer(cfg.name)
            except Exception:
                self.tokenizer = open_clip.get_tokenizer(tokenizer_name)
        except Exception as exc:
            raise ModelLoadError(f"failed to load model {cfg.name}: {exc}") from exc
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        self.dim = self._detect_dim()

    @classmethod
    def get(cls, cfg: ModelConfig) -> "ClipModel":
        key = (cfg.name, cfg.pretrained, _resolve_device(cfg.device))
        if key not in cls._cache:
            cls._cache[key] = cls(cfg)
        return cls._cache[key]

    def _detect_dim(self) -> int:
        try:
            sample = torch.zeros(1, 3, 224, 224, device=self.device)
            with torch.no_grad():
                feat = self.model.encode_image(sample)
            return int(feat.shape[-1])
        except Exception:
            return 512

    @torch.no_grad()
    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        tensors = torch.stack([self.preprocess(im.convert("RGB")) for im in images]).to(self.device)
        feats = self.model.encode_image(tensors)
        return torch.nn.functional.normalize(feats, dim=-1)

    @torch.no_grad()
    def encode_text(self, text: str | list[str]) -> torch.Tensor:
        if isinstance(text, str):
            text = [text]
        tokens = self.tokenizer(text).to(self.device)
        feats = self.model.encode_text(tokens)
        return torch.nn.functional.normalize(feats, dim=-1)
