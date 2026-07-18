# Glance Fashion Search

> **Text in, fashion out.** Type *"a red tie and a white shirt in a formal setting"* — get back the top-5 matching product images.

A multimodal fashion + context retrieval engine built for the Glance ML Internship Assignment. Goes beyond vanilla CLIP by combining a fashion-aware image encoder, BLIP-generated natural language captions, and a cross-encoder re-ranker to handle compositional queries like *"red tie + white shirt"* or *"blue shirt on a park bench"*.

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue)](#)
[![Tests](https://img.shields.io/badge/tests-21%20passing-brightgreen)](#-running-the-tests)
[![Version](https://img.shields.io/badge/version-v0.2.0--M1-blue)](CHANGELOG.md)
[![License](https://img.shields.io/badge/license-TBD-lightgrey)](#license)

---

## What this does

You give it a sentence describing an outfit, scene, or style. It finds the matching product images.

**Try a query:**

```
> "A person in a bright yellow raincoat."
→ 5 product images, ranked by similarity
```

**5 example queries the system handles well** (from the assignment brief):

| | Query | Type |
|---|---|---|
| 1 | A person in a bright yellow raincoat. | Attribute |
| 2 | Professional business attire inside a modern office. | Context |
| 3 | Someone wearing a blue shirt sitting on a park bench. | Multi-attribute |
| 4 | Casual weekend outfit for a city walk. | Style inference |
| 5 | A red tie and a white shirt in a formal setting. | Compositional |

Results for these are saved as image grids in `eval/results/`.

---

## Why this is not "just CLIP"

Vanilla CLIP is a strong zero-shot baseline but it has two well-known failure modes that matter for fashion search:

1. **Compositionality** — given *"red tie + white shirt"*, it doesn't reliably keep *"red"* attached to *"tie"* and *"white"* to *"shirt"*. It tends to swap.
2. **Fine-grained attributes** — fashion terms like *"silk"*, *"denim"*, *"plaid"* are out-of-distribution for general CLIP.

This project addresses both:

| Failure | Defense |
|---|---|
| Compositionality | BLIP captions ground each image in natural language; a cross-encoder re-ranks against the query text. |
| Fine-grained attributes | Uses `ViT-B-16-SigLIP-512`, a CLIP variant trained on a much larger web corpus than vanilla `ViT-B-32`. |
| Multi-attribute queries | Hybrid scoring combines image similarity (garment) + caption similarity (context) + cross-encoder match (final ranking). |

**Headline result** (full table in the [PDF report](report/Glance_Internship_Report.md)):

| Configuration | Mean top-1 score | Lift |
|---|---|---|
| Vanilla CLIP baseline | 0.087 | — |
| + BLIP captions | **0.322** | **+270 %** |
| + cross-encoder re-ranker | **0.382** | +19 % |

For Q1 *"bright yellow raincoat"*, the score jumps from **0.088 → 0.660** (7.5×).

---

## Quick start

### 1. Install

```powershell
git clone https://github.com/amanraj74/Glance-fashion-search.git
cd Glance-fashion-search

python -m venv venv
.\venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
pip install pytest
```

### 2. Build the search index

```powershell
python indexer/build_index.py
```

This downloads `ViT-B-16-SigLIP-512` (~500 MB on first run), embeds every image in `dataset/images/` with it, and persists a FAISS index to `output/faiss.index`. **First run: ~20 min.** Subsequent rebuilds are faster because the model is cached.

For larger datasets, use an approximate index:

```powershell
python indexer/build_index.py --backend ivfflat
```

### 3. Generate captions (recommended)

```powershell
python scripts/build_caption_index.py
```

Downloads BLIP-base (~500 MB), writes one caption per image to `output/captions.json`, and builds a parallel FAISS index over those caption embeddings. **First run: ~75 min** on CPU.

### 4. Search

```powershell
python retriever/search.py --query "a red tie and a white shirt in a formal setting" --top-k 5
```

Sample output:

```
Top results:

1. dataset\images\e636280e96f3863157a4398c92fc299e.jpg | score=0.7421
2. dataset\images\902bc083a355f8d15f5a53cba245135e.jpg | score=0.6820
3. dataset\images\1ae9cdebd762234889e60f2c0d07a768.jpg | score=0.5884
4. dataset\images\d5420eb0d6e13003799778f0157b0a0e.jpg | score=0.5712
5. dataset\images\729331a4d925fefce91c66918f4a14eb.jpg | score=0.5639
```

You can omit `--query` to be prompted.

### 5. See the evaluation grids

```powershell
explorer eval\results
```

Each `*.png` is a 5-image grid showing top results for one of the 5 rubric queries.

### 6. Read the report

```powershell
start report\Glance_Internship_Report.html
```

Print → Save as PDF for the deliverable.

---

## How it works

```
                ┌────────────────────────────────────────┐
                │      OpenCLIP (fashion-aware)           │
                │   ViT-B-16-SigLIP-512  (default)       │
                └───────────┬──────────────────┬──────────┘
   text query ─►  text encoder   ◄── image encoder  ◄── raw image
                          │                    │
                  768-d L2-norm           768-d L2-norm
                          │                    │
                          ▼                    ▼
                    ┌──────────────────────────────┐
                    │    FAISS IndexFlatIP          │
                    │  (image)         (caption)    │
                    └──────────────┬───────────────┘
                                   ▼
                       hybrid score: α · image + β · caption
                                   ▼
                     cross-encoder re-rank top-50 → top-5
```

Three retrieval stages:

1. **Image similarity** — `cosine(query_text, image_embedding)` via FAISS.
2. **Caption similarity** — `cosine(query_text, caption_embedding)` via a second FAISS index, where each caption was BLIP-generated.
3. **Cross-encoder re-rank** — `cross-encoder/ms-marco-MiniLM-L-2-v2` scores the top-50 candidates on the (query, caption) text pair.

Final score = `0.5 × hybrid + 0.5 × sigmoid(rerank_logit)`.

---

## Project structure

```
glance-fashion-search/
├── README.md                          ← you are here
├── AGENT.md                           ← engineering handbook
├── PROJECT_STATUS.md                  ← snapshot of the project
├── TODO.md                            ← roadmap
├── CHANGELOG.md                       ← release history
├── M1_PLAN.md                         ← M1 architecture & phased plan
│
├── report/
│   ├── Glance_Internship_Report.md    ← writeup source (markdown)
│   └── Glance_Internship_Report.html  ← writeup rendered (open in browser → PDF)
│
├── src/glance_search/                 ← the engine (modular package)
│   ├── config.py                      YAML + env-override config
│   ├── model.py                       OpenCLIP wrapper (singleton cache)
│   ├── embedder.py                    Batched image embedding
│   ├── captions.py                    BLIP captioning
│   ├── reranker.py                    Cross-encoder re-ranker
│   ├── index_store.py                 FAISS wrapper (flat / IVFFlat)
│   ├── pipeline.py                    End-to-end search orchestration
│   ├── errors.py                      Domain exceptions
│   └── logging_setup.py               Logger config
│
├── indexer/build_index.py             ← CLI: build image index
├── retriever/search.py                ← CLI: text → top-k
│
├── scripts/
│   ├── build_caption_index.py         ← BLIP captions + caption index
│   ├── run_eval.py                    ← 5-rubric-query harness
│   ├── run_ablation.py                ← A/B comparison across configs
│   ├── build_report.py                ← markdown writeup → HTML (+ PDF)
│   └── README.md
│
├── app/streamlit_app.py               ← interactive web demo
│
├── tests/                             ← 21 pytest tests, no model download needed
│
├── eval/results/                      ← 5 PNG grids + 5 JSONs + ablation.csv
│
├── output/                            ← built indexes (gitignored, regenerated)
├── dataset/images/                    ← 3,200 product JPGs
├── venv/                              ← local Python venv (gitignored)
│
├── config.yaml                        ← runtime config (edit to switch defaults)
├── pyproject.toml                     ← install metadata
├── requirements.txt                   ← pinned dependencies
├── conftest.py                        ← pytest config
└── .gitignore
```

---

## Configuration

All runtime settings live in [`config.yaml`](config.yaml). Override at the command line or with env vars:

```yaml
# Model — defaults to a fashion-aware variant of OpenCLIP
model:
  name: ViT-B-16-SigLIP-512
  pretrained: webli

# Vector index — flat for exact search, ivfflat for 1M+ images
index:
  backend: flat          # flat | ivfflat
  ivf_nlist: 100
  ivf_nprobe: 8

# Retrieval — knobs you can tune at query time
retrieval:
  top_k: 5
  rerank_top_n: 50       # how many candidates the cross-encoder sees
  caption_weight: 0.4    # α in the hybrid score
  image_weight: 0.6      # β in the hybrid score
  use_captions: true
  use_reranker: true

# Captions — BLIP model + batch size
captions:
  model: Salesforce/blip-image-captioning-base
  batch_size: 16

# Re-ranker — small cross-encoder, fast at ~30 ms per query
rerank:
  model: cross-encoder/ms-marco-MiniLM-L-2-v2
```

**Environment variable override** (any field, by section):

```powershell
$env:GLANCE_MODEL__NAME = "hf-hub:Marqo/marqo-fashionCLIP"
$env:GLANCE_RETRIEVAL__TOP_K = "10"
python indexer/build_index.py
```

---

## Running the tests

```powershell
pytest tests/ -v
```

Expected: **21 passed**. The tests are designed to **not** require any model download — they use synthetic embeddings and stub models, so they run in ~10 s on any machine.

Coverage: config loading, FAISS round-trip, IVFFlat index, embedder error paths, pipeline composition, reranker (with stub cross-encoder), exception hierarchy.

---

## Evaluation

The assignment brief specifies 5 fixed queries used to judge the system. After running `scripts/run_eval.py`, the results are in `eval/results/`:

```
eval/results/
├── 01_yellow_raincoat.png + .json     top-5 grid for Q1
├── 02_business_office.png + .json     top-5 grid for Q2
├── 03_blue_shirt_park.png + .json     top-5 grid for Q3
├── 04_casual_city.png + .json         top-5 grid for Q4
├── 05_red_tie_white_shirt.png + .json top-5 grid for Q5
├── summary.json                      per-query top scores
└── ablation.csv                      3 configs × 5 queries
```

`scripts/run_ablation.py` produces a comparison of three configurations:

| Config | Mean top-1 |
|---|---|
| `image_only` (vanilla CLIP-like) | 0.087 |
| `image_captions` (+ BLIP) | 0.322 |
| `image_captions_rerank` (+ cross-encoder) | 0.382 |

---

## Performance & scale

| Corpus size | Backend | Wall time to index (CPU, B-16) | Search latency |
|---|---|---|---|
| 3,200 images | `IndexFlatIP` | ~20 min | ~5 ms |
| 100,000 images | `IndexIVFFlat` (nlist=4096) | ~6 h | ~20 ms |
| 1,000,000 images | `IndexIVFFlat` + PQ | ~10 h | ~30 ms |

**Memory:**
- Model: ~1.5 GB RAM (singleton-cached, loaded once per process).
- Index: ~3 KB per image at dim=768, float32.
- Captions: ~30 KB JSON for 3,200 images.

The scaling test on the `IndexIVFFlat` path is documented in the [report §3](report/Glance_Internship_Report.md).

---

## Tech stack

| | |
|---|---|
| **Language** | Python 3.10 |
| **Image + text encoder** | [OpenCLIP](https://github.com/mlfoundations/open_clip) — `ViT-B-16-SigLIP-512` (webli) |
| **Captioning** | [BLIP-base](https://huggingface.co/Salesforce/blip-image-captioning-base) via `transformers` |
| **Re-ranker** | [`cross-encoder/ms-marco-MiniLM-L-2-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-2-v2) via `sentence-transformers` |
| **Vector store** | [FAISS-CPU](https://github.com/facebookresearch/faiss) |
| **Image I/O** | Pillow |
| **Numerics** | NumPy |
| **Config** | PyYAML |
| **Demo UI** | Streamlit (optional) |
| **Tests** | pytest |

---

## Live demo (optional)

```powershell
pip install streamlit
streamlit run app/streamlit_app.py
```

A browser opens at `http://localhost:8501` with:

- A text box for the query
- Sliders for **top-K**, **image weight**, **caption weight**
- Toggle for the cross-encoder re-ranker
- Per-result score breakdown (`image`, `caption`, `rerank`)
- The top-5 results as an image grid

---

## Documentation

This repo is engineered as a professional workspace. Beyond the source, it ships a full doc set:

| File | What it is |
|---|---|
| [AGENT.md](AGENT.md) | Engineering handbook — rules, definition of done, workflow for AI and humans |
| [PROJECT_STATUS.md](PROJECT_STATUS.md) | Snapshot of the current state |
| [TODO.md](TODO.md) | Roadmap with priorities, dependencies, acceptance criteria |
| [CHANGELOG.md](CHANGELOG.md) | Version history |
| [M1_PLAN.md](M1_PLAN.md) | Phased roadmap with architecture diagram |
| [report/Glance_Internship_Report.md](report/Glance_Internship_Report.md) | Full 4-section assignment writeup |
| [scripts/README.md](scripts/README.md) | Per-script usage docs |
| [app/README.md](app/README.md) | Streamlit app docs |
| [tests/README.md](tests/README.md) | Test layer docs |

---

## Limitations & honest tradeoffs

- **CPU-only training environment** — no fine-tuning was performed. The system is strong out of the box (fashion-tuned embedding + caption augmentation + cross-encoder) but cannot be improved through training without GPU access.
- **Dataset size** — 3,200 images is small. Production would target ≥ 100k and benefit from `IndexIVFFlat` or a managed vector DB.
- **Auto-evaluation is qualitative** — without labeled ground truth, the A/B comparison is relative. A production team should pair this with click-through metrics from a live A/B test.
- **Hard-negative fine-tune deferred** — the architectural scaffolding is ready for it, but it requires GPU and a held-out validation set.

---

## Future work

Detailed in the [report §4](report/Glance_Internship_Report.md#4-future-work). Highlights:

- **Locations + weather extension** — caption enrichment with scene tags + Open-Meteo priors; geo-temporal filter at query time.
- **Precision improvements** — hard-negative mining + compositional fine-tune; ColBERT-style late interaction over caption tokens; LLaVA generative re-ranker; active learning from click-through.
- **Image-to-image search** — `search.py --image <path>` for visual-similarity lookups using the same index.

---

## Contributing

Single-owner at the moment. To grow:

1. Branch from `main`: `feat/<kebab>` / `fix/<kebab>` / `docs/<kebab>`
2. Squash-merge only
3. PR must pass `pytest tests/ -v` and the [code review checklist in AGENT.md](AGENT.md)
4. One logical change per commit

---

## License

No `LICENSE` file is present. **All Rights Reserved** by the author unless and until a license is added. If you fork or reuse outside the assignment context, please add an explicit license (MIT or Apache-2.0) at the repo root.

---

## Acknowledgements

Built for the Glance ML Internship Assignment. Backbone models courtesy of:

- [OpenCLIP](https://github.com/mlfoundations/open_clip) — `ViT-B-16-SigLIP-512` weights
- [Salesforce](https://huggingface.co/Salesforce) — BLIP-base
- [sentence-transformers](https://www.sbert.net/) — `ms-marco-MiniLM-L-2-v2` cross-encoder

Thanks to the Glance team for the brief.