# PROJECT_STATUS.md

Last updated: **2026-07-18**. Source of truth: files on disk. No invented progress.

## Identification

| Field | Value |
|---|---|
| Project Name | glance-fashion-search |
| Repo Type | ML Internship Assignment (Glance) |
| Assignment | Multimodal Fashion & Context Retrieval |
| Owner | Intern (single contributor) |
| Documentation Set | AGENT.md, PROJECT_STATUS.md, TODO.md, README.md, CHANGELOG.md, M1_PLAN.md, plan.md, status.md, conftest.py, pyproject.toml, .gitignore |
| Last Updated | 2026-07-18 |

## Project Goal

Build a text-to-image retrieval system that returns matching product images for natural language queries combining clothing type, color, environment, and style/vibe — beating vanilla CLIP on compositionality and fine-grained fashion attributes. (Source: `Glance ML Internship Assignment.md`.)

## Current Version

`v0.2.0-M1-architecture`

## Current Architecture

```
                       ┌────────────────────────────┐
                       │   Query (raw text)          │
                       └──────────────┬─────────────┘
                                      │ encode_text
                                      ▼
                       ┌────────────────────────────┐
                       │   OpenCLIP text encoder     │
                       │   ViT-B-16-SigLIP-512       │
                       └──────────────┬─────────────┘
                                      │ 1 × 768-d
                                      ▼
   ┌──────────────────────────────────────────────────────┐
   │   FAISS IndexFlatIP (image) ──→ top-N=50 image idx    │
   │   FAISS IndexFlatIP (caption) ──→ top-N=50 caption idx │
   └──────────────────────────────────────────────────────┘
                                      │ hybrid score: α·img + β·caption
                                      ▼
                       ┌────────────────────────────┐
                       │   Cross-encoder re-ranker   │
                       │   ms-marco-MiniLM-L-2-v2    │
                       └──────────────┬─────────────┘
                                      │
                                      ▼
                                top-k results
```

Layers:

- **CLI** (`indexer/`, `retriever/`, `scripts/`, `app/`) — thin entry points.
- **Embedder** (`src/glance_search/{model,embedder}.py`) — wraps OpenCLIP, cached singleton.
- **Vector store** (`src/glance_search/index_store.py`) — FAISS flat or IVFFlat, persisted.
- **Captions** (`src/glance_search/captions.py`) — BLIP-base offline captioning.
- **Re-ranker** (`src/glance_search/reranker.py`) — cross-encoder.
- **Pipeline** (`src/glance_search/pipeline.py`) — end-to-end search.

## Current Sprint

M1 Industry-Grade Sprint — **all code complete; awaiting user-triggered compute (R1–R5 in TODO.md)**.

## Completed Features

- [x] Modular `src/glance_search/` package (10 modules)
- [x] YAML + env-override config with type coercion
- [x] OpenCLIP wrapper with singleton cache and auto-dim
- [x] FAISS flat + IVFFlat wrapper (toggle via `--backend`)
- [x] Batched, L2-normalized image embedding with per-file error handling
- [x] BLIP-base offline captioning (resumable, batched)
- [x] Cross-encoder re-ranker
- [x] Multi-vector late-interaction search
- [x] 5-rubric evaluation harness (`scripts/run_eval.py`)
- [x] A/B ablation harness (`scripts/run_ablation.py`)
- [x] Streamlit interactive demo (`app/streamlit_app.py`)
- [x] Report generator (`scripts/build_report.py`) → HTML; PDF if weasyprint present
- [x] 21-test pytest suite (no model download)
- [x] Production README, M1 plan, project status, changelog, AGENT handbook
- [x] `pyproject.toml`, `.gitignore`, `conftest.py`

## Features In Progress

- (none — coding complete; awaiting compute runs)

## Pending Features

| Item | Status |
|---|---|
| R1 — Reindex with new backend | pending user |
| R2 — Build caption index | pending user |
| R3 — Run 5-rubric evaluation | pending user |
| R4 — Run ablation comparison | pending user |
| R5 — Generate PDF report | pending user |
| R6 — Streamlit demo launch | optional, pending user |
| Image-to-image search | deferred to future |
| Dataset axis manifest | deferred to future |
| LICENSE file | TODO |

## Backend Status

| Field | Value |
|---|---|
| Language | Python 3.10 |
| ML Framework | torch 2.13.0 + torchvision 0.28.0 |
| CLIP Wrapper | open_clip_torch 3.3.0 |
| Captioning | transformers 5.13.1 (BLIP-base) |
| Vector DB | faiss-cpu 1.14.3 |
| Re-ranker | sentence-transformers 5.6.0 (cross-encoder/ms-marco-MiniLM-L-2-v2) |
| Other | accelerate 1.14.0, pytest, Pillow 12.3, numpy 2.2.6, PyYAML 6.0, tqdm 4.68.4 |
| Device | Auto (CUDA if available, else CPU). Currently CPU. |
| Status | All code complete; nothing run yet with new backend |

## Frontend Status

Streamlit demo (`app/streamlit_app.py`) — code complete, awaiting `pip install streamlit` + run.

## Database Status

- FAISS binary index file (`output/faiss.index`) — built with **old** backend (`ViT-B-32 openai`, dim=512). Awaiting rebuild with new backend.
- JSON metadata file (`output/metadata.json`) — 3,200 paths, row-aligned with current index.
- No real DB, no transactions, no migrations.

## Infrastructure Status

- Local Windows machine, PowerShell 5.1.
- Python venv at `venv/`.
- `sentence-transformers`, `accelerate`, `pytest` installed.
- No Docker, no CI/CD, no cloud config, no Terraform.

## Deployment Status

Not deployed. Runs locally via `python retriever/search.py`. Future work targets FastAPI + Docker.

## Testing Status

- Unit tests: **21 pass** (`pytest tests/`). No model download required.
- Integration tests: covered by pipeline tests with stub model.
- Evaluation harness: code complete, not run yet.
- Smoke test: implicit (manual run of `search.py`).

## Documentation Status

| Doc | File | Status |
|---|---|---|
| AI Engineering Handbook | `AGENT.md` | Complete |
| Project Status | `PROJECT_STATUS.md`, `status.md` | Complete |
| Engineering Roadmap | `TODO.md`, `plan.md` | Complete |
| User README | `README.md` | Complete |
| Changelog | `CHANGELOG.md` | Complete |
| Assignment Brief | `Glance ML Internship Assignment.md` and `.pdf` | Present |
| Industry Plan | `M1_PLAN.md` | Complete |
| Code docstrings | All modules | Complete |
| Deliverable PDF in `report/` | `report/Glance_Internship_Report.md` source complete; PDF pending R5 | Partial |

## Known Bugs

None observed in code or unit tests. Cannot verify against 5 rubric queries until R1+R2+R3 run.

## Known Risks

| ID | Risk | Mitigation |
|---|---|---|
| R1 | Vanilla CLIP fails compositionality (rubric concern) | T2 — caption augmentation + re-ranker; verify post-R3 |
| R2 | No deliverable PDF — half the grade | R5 |
| R3 | `IndexFlatIP` O(N) — does not scale to 1M | T5 — IVFFlat wrapper ready |
| R4 | No dataset manifest — can't prove axis coverage | T6 — TODO |
| R5 | No evaluation evidence for the 5 queries | R3 |
| R6 | Model loaded twice (indexer + retriever) | T2 — fixed via singleton cache in `model.py` |

## Technical Debt

- `requirements.txt` still contains a few unneeded transitive deps (typer, shellingham, click) — left as-is to avoid breaking other tools.
- `LogLevel` global state in `logging_setup.py` is process-wide; fine for CLI but would need rework for a library.
- No image preprocessing tests (rely on stub model).
- Captions are 1 per image; could be 3-5 for richer compositionality.

## Blockers

None operational. All code paths tested; nothing depends on user input except R1–R5.

## Next Milestone

**R1–R5** (reindex, captions, eval, ablation, PDF). User-triggered; total wall time ~1.5–2 hours on CPU.

## Recommended Next Task

**R1** — `python indexer/build_index.py`. Smallest, highest-rubric-impact. Replaces the vanilla CLIP index with the fashion-aware one and unblocks R2–R5.

## Cross-Reference Map

- `AGENT.md` — engineering rules, definition of done, workflow.
- `TODO.md` — current task list, priorities, acceptance criteria.
- `M1_PLAN.md` — phased roadmap with architecture diagram.
- `README.md` — user-facing setup, run, eval commands.
- `CHANGELOG.md` — version history.
- `plan.md` / `status.md` — duplicate of this file + `TODO.md` under alternative names.
- `Glance ML Internship Assignment.md` — assignment brief (root of truth for requirements).