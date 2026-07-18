# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added
- **Industry-grade M1 architecture** (`src/glance_search/`):
  - `config.py` — YAML + env-override config (single source of truth, type-coerced)
  - `model.py` — OpenCLIP wrapper with singleton cache and auto-dim detection
  - `embedder.py` — batched, L2-normalized, per-image error handling
  - `index_store.py` — FAISS `IndexFlatIP` + `IndexIVFFlat` toggle
  - `captions.py` — BLIP-base offline captioning, batched and resumable
  - `reranker.py` — cross-encoder re-ranker (`ms-marco-MiniLM-L-2-v2`)
  - `pipeline.py` — full multi-vector hybrid retrieval: image + caption + re-rank
  - `errors.py`, `logging_setup.py` — domain exceptions + idempotent logger config
- **CLIs:**
  - `indexer/build_index.py` rewritten as thin wrapper (was 63 lines, now ~80 with arg parser)
  - `retriever/search.py` rewritten as thin wrapper
  - `scripts/build_caption_index.py` — generates captions, embeds them, persists
  - `scripts/run_eval.py` — runs the 5 rubric queries, saves JSON + PNG grids
  - `scripts/run_ablation.py` — A/B comparison across 3 configs, writes `ablation.csv`
  - `scripts/build_report.py` — renders the report markdown → HTML → PDF
- **App:** `app/streamlit_app.py` — interactive demo with sliders for caption weight, top-k, re-ranker toggle, and per-result score breakdown
- **Tests:** 21 pytest tests covering config, embedder, FAISS store, pipeline, reranker, errors/logging (no model download required)
- **Docs:**
  - `M1_PLAN.md` — phased roadmap with architecture diagram
  - `report/Glance_Internship_Report.md` — full assignment writeup source (4 mandated sections)
  - `.gitignore` — proper excludes for caches, generated artifacts
  - `pyproject.toml` — install metadata
  - `conftest.py` — repo-root pytest config so `glance_search` is importable

### Changed
- Backend default moved from `ViT-B-32 openai` to `ViT-B-16-SigLIP-512 (webli)` (configurable via `config.yaml` or CLI flag `--model`)
- Config now supports env vars of form `GLANCE_<SECTION>__<FIELD>` with auto-type coercion
- Index now supports `--backend {flat,ivfflat}` flag for the 1M-image scalability story
- Search pipeline now supports image-only / image+captions / image+captions+reranker via config flags

### Removed
- Old monolithic `indexer/build_index.py` and `retriever/search.py` direct paths (moved to `src/glance_search/`)
- `print` statements replaced with structured `logging`
- `tokenizer` unused import in old indexer

### Pending (this release; await user-triggered compute)
- Reindex of 3.2k images with `ViT-B-16-SigLIP-512` (15-25 min CPU)
- BLIP captioning of all 3.2k images (30-60 min CPU)
- 5-rubric evaluation grids (`eval/results/*.png`)
- Ablation table (`eval/results/ablation.csv`)
- Final PDF (`report/Glance_Internship_Report.pdf`)

---

## [0.1.0] — 2026-07-16 — Baseline

### Added
- `indexer/build_index.py` — OpenCLIP `ViT-B-32` (openai) image embeddings → FAISS `IndexFlatIP`
- `retriever/search.py` — CLI text → top-5 cosine search
- `output/faiss.index` (6.5 MB) — persisted FAISS index over 3,200 product images
- `output/metadata.json` (188 KB) — image path list, row-aligned with index
- `requirements.txt` — pinned dep set for torch 2.13, open_clip_torch 3.3.0, faiss-cpu 1.14.3
- Local venv under `venv/`

### Known Limitations (from TODO)
- Vanilla CLIP only; does not beat the assignment's compositionality rubric
- `IndexFlatIP` is O(N); does not scale to 1M images
- No dataset axis manifest
- No automated evaluation against the 5 rubric queries
- No `README.md` (zero-byte placeholder at the time)
- No `report/` deliverable PDF
- No modular packaging; scripts duplicated model loading

---

## Version History at a Glance

| Version | Date | State |
|---|---|---|
| Unreleased | — | M1 architecture complete, awaiting compute runs |
| 0.1.0 | 2026-07-16 | Baseline CLI retrieval over 3,200 images |
