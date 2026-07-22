"""Runtime configuration loaded from YAML and overridden by environment.

Single source of truth. CLI scripts read `Config` and never hardcode paths
or model names.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    name: str = "ViT-B-16-SigLIP-512"
    pretrained: str = "webli"
    cache_dir: str | None = None
    device: str = "auto"


@dataclass(frozen=True)
class IndexConfig:
    image_dir: str = "dataset/images"
    output_dir: str = "output"
    index_path: str = "output/faiss.index"
    metadata_path: str = "output/metadata.json"
    caption_path: str = "output/captions.json"
    caption_index_path: str = "output/captions.index"
    backend: str = "flat"
    ivf_nlist: int = 100
    ivf_nprobe: int = 8

    @property
    def index_path_obj(self) -> Path:
        return Path(self.index_path)

    @property
    def metadata_path_obj(self) -> Path:
        return Path(self.metadata_path)

    @property
    def caption_path_obj(self) -> Path:
        return Path(self.caption_path)

    @property
    def caption_index_path_obj(self) -> Path:
        return Path(self.caption_index_path)


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 5
    rerank_top_n: int = 150
    rerank_weight: float = 0.35
    caption_weight: float = 0.35
    image_weight: float = 0.65
    use_captions: bool = True
    use_reranker: bool = True
    expand_queries: bool = True
    rerank_min_caption_chars: int = 12
    scoring: str = "rrf"
    rrf_k: int = 60
    attribute_bonus: float = 0.30
    hard_negative_penalty: float = 0.20
    semantic_attribute_weight: float = 0.80


@dataclass(frozen=True)
class CaptionsConfig:
    enabled: bool = True
    model: str = "Salesforce/blip-image-captioning-base"
    batch_size: int = 16
    max_new_tokens: int = 30
    num_beams: int = 3


@dataclass(frozen=True)
class RerankConfig:
    enabled: bool = True
    model: str = "cross-encoder/ms-marco-MiniLM-L-2-v2"
    batch_size: int = 16


@dataclass(frozen=True)
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    captions: CaptionsConfig = field(default_factory=CaptionsConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    log_level: str = "INFO"

    @property
    def image_dir_path(self) -> Path:
        return Path(self.index.image_dir)

    @property
    def index_path_obj(self) -> Path:
        return Path(self.index.index_path)

    @property
    def metadata_path_obj(self) -> Path:
        return Path(self.index.metadata_path)

    @property
    def caption_path_obj(self) -> Path:
        return Path(self.index.caption_path)

    @property
    def caption_index_path_obj(self) -> Path:
        return Path(self.index.caption_index_path)


_ENV_PREFIX = "GLANCE_"


def _coerce(value: Any, sample: Any) -> Any:
    """Coerce `value` to match the type of `sample` (a default from the dataclass)."""
    if isinstance(sample, bool):
        return str(value).strip().lower() in ("1", "true", "yes", "on")
    if isinstance(sample, int) and not isinstance(sample, bool):
        try:
            return int(value)
        except (TypeError, ValueError):
            return value
    if isinstance(sample, float):
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    return value


def _coerce_section(section_obj: Any, raw_values: dict[str, Any]) -> dict[str, Any]:
    coerced: dict[str, Any] = {}
    for k, v in raw_values.items():
        if not hasattr(section_obj, k):
            continue
        sample = getattr(section_obj, k)
        coerced[k] = _coerce(v, sample)
    return coerced


def _apply_overrides(cfg: Config, raw: dict[str, Any]) -> Config:
    sections = {
        "model": cfg.model,
        "index": cfg.index,
        "retrieval": cfg.retrieval,
        "captions": cfg.captions,
        "rerank": cfg.rerank,
    }
    for section_name, section_obj in sections.items():
        if section_name not in raw:
            continue
        coerced = _coerce_section(section_obj, raw[section_name])
        if coerced:
            sections[section_name] = replace(section_obj, **coerced)
    log_level_raw = raw.get("log_level", cfg.log_level)
    log_level = _coerce(log_level_raw, cfg.log_level) if not isinstance(log_level_raw, str) else log_level_raw
    return Config(
        model=sections["model"],
        index=sections["index"],
        retrieval=sections["retrieval"],
        captions=sections["captions"],
        rerank=sections["rerank"],
        log_level=log_level,
    )


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load configuration. Order: defaults -> YAML file -> env vars."""
    raw: dict[str, Any] = {}
    cfg_path = Path(path) if path else Path("config.yaml")
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    cfg = Config()
    cfg = _apply_overrides(cfg, raw)

    env_raw: dict[str, Any] = {}
    for key, value in os.environ.items():
        if not key.startswith(_ENV_PREFIX) or "__" not in key:
            continue
        section, field_name = key[len(_ENV_PREFIX):].lower().split("__", 1)
        env_raw.setdefault(section, {})[field_name] = value
    if env_raw:
        cfg = _apply_overrides(cfg, env_raw)
    return cfg
