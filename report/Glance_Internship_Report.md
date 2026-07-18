# Multimodal Fashion & Context Retrieval

## Glance ML Internship — Submission Writeup

**Author:** Aman Jaiswal
**Repository:** [github.com/amanraj74/Glance-fashion-search](https://github.com/amanraj74/Glance-fashion-search)
**Brief:** [Glance ML Internship Assignment](../Glance%20ML%20Internship%20Assignment.md) — text-to-image retrieval over a fashion catalogue with compositional, contextual, and style-inference queries.

---

## TL;DR

The assignment asks for a search engine that returns product images for natural-language fashion queries — *"bright yellow raincoat"*, *"business attire in a modern office"*, *"red tie and white shirt in a formal setting"* — and explicitly warns against a vanilla CLIP baseline because it fails compositionality. This writeup documents the system we built and the reasoning behind it.

**Approach in one sentence.** Embed every catalogue image with a fashion-tuned CLIP model, generate a BLIP caption for each image, score the query against both the image embedding and the caption embedding, and re-rank the top candidates with a cross-encoder.

**Headline result.** On the assignment's five test queries, mean top-1 score goes from **0.104** (image-only baseline) to **0.437** with captions + re-ranking — a 4.2× improvement. On the compositional query *"red tie + white shirt, formal"*, the score rises from 0.071 to 0.309.

---

## Table of contents

1. [Approaches considered](#1-approaches-considered)
2. [Chosen approach](#2-chosen-approach)
3. [Evaluation results](#3-evaluation-results)
4. [Future work](#4-future-work)
5. [Appendix A — Codebase](#appendix-a--codebase)
6. [Appendix B — How to reproduce](#appendix-b--how-to-reproduce)
7. [Appendix C — Honest tradeoffs](#appendix-c--honest-tradeoffs)

---

## 1. Approaches considered

The problem is text-to-image retrieval over a fashion catalogue. We considered six plausible solutions and ranked them against three constraints from the brief: (a) must beat vanilla CLIP on compositional queries, (b) must be zero-shot, (c) must run on CPU within an intern-scale compute budget.

### 1.1 Vanilla CLIP — *rejected*

A single `ViT-B-32` model encodes both images and text. Top-k by cosine similarity.

| Strengths | Weaknesses |
|---|---|
| Trivial to implement | Fails the exact compositionality test the brief calls out |
| Tiny index, fast search | General-domain training is out-of-distribution for *"silk"*, *"plaid"*, *"raincoat"* |
| Many ready-made checkpoints | Single-vector similarity collapses fine-grained signal |

**Verdict.** Discarded. The brief itself warns against this approach.

### 1.2 Fashion-tuned CLIP — *adopted as the backbone*

Replace `ViT-B-32` with a model trained on fashion data: `Marqo/marqo-fashionCLIP` (a fashion-finetuned ViT-L/14, OpenCLIP-compatible), or `ViT-B-16-SigLIP-512` trained on a much larger web corpus.

| Strengths | Weaknesses |
|---|---|
| Domain knowledge for fashion attributes | Does *not* fix compositionality on its own |
| Plug-in compatible with the existing pipeline | Larger checkpoint, slower CPU inference |
| Strong public performance on fashion benchmarks | One vector per image still can't disentangle attributes |

**Verdict.** Adopted as the embedding backbone. `Marqo/marqo-fashionCLIP` is the current default; `ViT-B-16-SigLIP-512` is flagged as a near-equivalent alternative in `config.yaml`.

### 1.3 Caption-augmented dual-vector retrieval — *adopted*

Generate a natural-language caption for every image with BLIP-base. Embed captions with the same CLIP text encoder. Score the query against both vectors and combine:

```
score(query, image) = α · cosine(query, image_emb) + β · cosine(query, caption_emb)
```

| Strengths | Weaknesses |
|---|---|
| Captions decompose images into named attributes | Captions are imperfect proxies of the image |
| Late-interaction handles compositionality | Doubles index size |
| Reuses existing encoders | Adds a one-time BLIP pass |
| Known to lift scores on CLIP-Composition / SugarCrepe benchmarks | |

**Verdict.** Adopted. This is the load-bearing piece of the system — without it, mean top-1 falls from 0.437 to 0.104.

### 1.4 Late-interaction (ColBERT-style) over caption tokens — *deferred*

Embed every caption token, store them all, and at query time take the max similarity between query tokens and caption tokens.

| Strengths | Weaknesses |
|---|---|
| Theoretically best for compositionality | Roughly 100× storage cost (one vector per token) |
| Proven in text-IR benchmarks | Slower at query time |
| | Marginal gain over §1.3 on most of our queries |

**Verdict.** Deferred to future work. The marginal gain on a 3.2k corpus is not worth the complexity.

### 1.5 Cross-encoder re-ranking — *adopted*

After retrieving the top-50–100 candidates via §1.3, score each `(query, caption_text)` pair with a text cross-encoder (`cross-encoder/ms-marco-MiniLM-L-2-v2`) and reorder.

| Strengths | Weaknesses |
|---|---|
| Significant precision bump on long-tail queries | ~30–60 ms extra latency per query |
| 30 MB checkpoint, easy to cache | Depends on §1.3 captions being present |
| Standard in modern IR pipelines | |

**Verdict.** Adopted. Toggleable via `config.yaml`; defaults to on.

### 1.6 External metadata + tags — *deferred*

Attach structured tags (brand, garment type, season, occasion) at index time and filter before vector search.

| Strengths | Weaknesses |
|---|---|
| Precise filtering | Requires a reliable tagger |
| Compositional via AND-of-filters | The catalogue doesn't ship with these tags |

**Verdict.** Deferred. Useful in production but requires a tagging pipeline that wasn't in scope here.

### 1.7 What we picked — at a glance

| # | Component | Choice |
|---|---|---|
| Embedder | Fashion-aware OpenCLIP | `Marqo/marqo-fashionCLIP` (default), `ViT-B-16-SigLIP-512` opt-in |
| Image index | FAISS `IndexFlatIP` | Exact cosine on normalised vectors |
| Caption generator | `Salesforce/blip-image-captioning-base` | Offline, batched, resumable |
| Caption index | FAISS `IndexFlatIP` over caption embeddings | One row per image |
| Hybrid score | Weighted sum (image + caption) | α, β configurable; defaults 0.5 / 0.5 |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-2-v2` | Applied to top-100 candidates |
| Scale-up path | `IndexIVFFlat` with PQ | One CLI flag away |

---

## 2. Chosen approach

This section explains the architecture in enough detail that a reviewer could re-derive the design choices.

### 2.1 Pipeline at a glance

The query path:

```
                      ┌────────────────────────────┐
                      │   Query (raw text)          │
                      └──────────────┬─────────────┘
                                     ▼
                      ┌────────────────────────────┐
                      │   Query expansion (optional)│
                      │   paraphrases for zero-shot │
                      └──────────────┬─────────────┘
                                     ▼
                      ┌────────────────────────────┐
                      │   Fashion-CLIP text encoder │
                      └──────────────┬─────────────┘
                                     │ 1 × D
                                     ▼
                      ┌────────────────────────────┐
                      │   FAISS IndexFlatIP (image) │ ── top-N=100 image indices
                      │   FAISS IndexFlatIP (cap)   │ ── top-N=100 caption indices
                      └──────────────┬─────────────┘
                                     │ 2N candidates
                                     ▼
                      ┌────────────────────────────┐
                      │   Hybrid scoring            │
                      │   α·image_sim + β·caption_sim│
                      └──────────────┬─────────────┘
                                     │ top-100
                                     ▼
                      ┌────────────────────────────┐
                      │   Cross-encoder re-ranker   │
                      │   ms-marco-MiniLM-L-2-v2    │
                      └──────────────┬─────────────┘
                                     │ ranked list
                                     ▼
                                 top-k results
```

The offline indexer runs once per catalogue:

```
images/                         caption generator (BLIP-base)
  │                                       │
  ├──► Fashion-CLIP image encoder        └──► JSON cache (resumable)
  │             │                                 │
  │             ▼                                 ▼
  │       faiss.index (image)             faiss.index (caption)
  └─────► metadata.json                  caption_meta.json
```

### 2.2 Why this beats vanilla CLIP on the brief's rubric

| Brief concern | What our system does |
|---|---|
| **Compositionality** — *"red tie + white shirt, formal"* | Caption augmentation lifts the binding into natural language. The cross-encoder reads `query` and `caption_text` together and judges whether the binding is preserved. |
| **Fine-grained attributes** — *"silk"*, *"denim"*, *"navy"* | Fashion-aware CLIP backbone has seen many more of these terms than general-domain models. |
| **Style inference** — *"casual weekend for a city walk"* | BLIP captions contain style vocabulary; the cross-encoder maps lifestyle intent to wardrobe semantics. |
| **Multi-attribute** — colour + garment + location | Each axis is encoded into a single vector; the hybrid score mixes image (garment) and caption (context); the cross-encoder disambiguates final ranking. |
| **Zero-shot generalisation** | No fine-tuning. Any new query that can be phrased in natural language works out of the box. |

### 2.3 Code organisation

The package boundary separates the ML logic from the engineering plumbing:

```
src/glance_search/
  config.py          YAML + env-override config (single source of truth)
  model.py           OpenCLIP wrapper (singleton cache, auto-dim)
  embedder.py        Batched image embedding, L2-normalised
  captions.py        BLIP-base offline captioning (resumable)
  reranker.py        Cross-encoder re-ranker
  index_store.py     FAISS IndexFlatIP + IndexIVFFlat
  pipeline.py        End-to-end search orchestration
  errors.py          Domain exceptions
  logging_setup.py   Idempotent logger config

indexer/build_index.py        CLI: embed images → faiss.index
retriever/search.py           CLI: text query → top-k
scripts/build_caption_index.py   CLI: BLIP captions + caption index
scripts/run_eval.py           CLI: 5-rubric-query harness
scripts/run_ablation.py       CLI: A/B comparison of configs
scripts/build_report.py       CLI: markdown writeup → HTML / PDF
app/streamlit_app.py          Streamlit interactive demo
tests/                        21 pytest tests, no model download
```

The indexer and retriever are deliberately thin: each is a CLI that loads config, calls into `src/glance_search/`, and writes/reads files. The ML logic lives in one place and can be unit-tested without spawning a subprocess.

### 2.4 Zero-shot story

Because the encoder is trained on web-scale image-text pairs, no query-specific training is required. The system can also paraphrase the input query before encoding (`retrieval.expand_queries: true` in `config.yaml`), which empirically helps when the user writes something the model hasn't seen in exactly that form.

---

## 3. Evaluation results

The assignment specifies five fixed queries. We evaluated three configurations against all five and recorded per-query top-1 score, top-5 mean score, and the per-component contributions.

### 3.1 The five queries

| # | Type | Query |
|---|---|---|
| Q1 | Single attribute | A person in a bright yellow raincoat. |
| Q2 | Context / setting | Professional business attire inside a modern office. |
| Q3 | Multi-attribute | Someone wearing a blue shirt sitting on a park bench. |
| Q4 | Style inference | Casual weekend outfit for a city walk. |
| Q5 | Compositional | A red tie and a white shirt in a formal setting. |

### 3.2 Quantitative results

Top-1 score per query, three configurations (from `eval/results/ablation.csv`):

| Configuration | Q1 yellow raincoat | Q2 business office | Q3 blue shirt + park | Q4 casual city | Q5 red tie + white shirt | **mean** |
|---|---:|---:|---:|---:|---:|---:|
| Image only | 0.103 | 0.091 | 0.112 | 0.144 | 0.071 | **0.104** |
| + BLIP captions | 0.336 | 0.359 | 0.356 | 0.310 | 0.310 | **0.334** |
| + cross-encoder re-rank | **0.660** | 0.287 | **0.639** | 0.292 | 0.309 | **0.437** |

**Stepwise lift:**

- Image-only → image + captions: **+221 %** (0.104 → 0.334)
- Image + captions → image + captions + re-rank: **+31 %** (0.334 → 0.437)
- Image-only → full pipeline: **+320 %** (0.104 → 0.437)

### 3.4 Per-query observations

- **Q1 — "A person in a bright yellow raincoat."** The strongest visual concept in the rubric. Image search alone misses it (raincoats skew dark in the model's prior). Captions lift it 3.3× (0.103 → 0.336); the cross-anker pushes it to 0.660 — a 6.4× improvement over baseline.
- **Q2 — "Professional business attire inside a modern office."** A context-heavy query. Image search is weak here (it picks suits but not the office setting). Captions carry most of the lift. The cross-encoder over-corrects and slightly drops the score, because the hybrid top-100 doesn't contain a clear winner for it to promote.
- **Q3 — "Someone wearing a blue shirt sitting on a park bench."** Similar shape to Q1. Image search finds blue shirts but misses the bench; captions tie the location to the garment; the re-ranker promotes the right composite. End-to-end lift: 5.7×.
- **Q4 — "Casual weekend outfit for a city walk."** Pure style inference — no specific colour or garment in the query. Captions lift this query the most (image search returns 0.144; captions alone get to 0.310). The re-ranker doesn't add much here for the same reason as Q2.
- **Q5 — "A red tie and a white shirt in a formal setting."** The canonical compositionality test. Image search fails badly (0.071). Captions rescue it to 0.310 (a 4.4× lift). The re-ranker keeps it roughly flat because the top-1 already matches.

### 3.5 Where it falls short — and why

The re-ranker's mixed impact on Q2 and Q4 is the single largest improvement opportunity. The cause is straightforward: `rerank_top_n = 100` means the cross-encoder sees the top-100 hybrid candidates, but for context-heavy queries the *truly* relevant item may sit at position 150–300 in the hybrid list. Two small fixes would recover this:

1. Raise `rerank_top_n` to 200 (one-line config change).
2. Or re-rank from the top-N of the *caption* index alone — for context-heavy queries, caption similarity is more reliable than image similarity.

Both are noted in the project roadmap and require no architectural changes.

---

## 4. Future work

The assignment asks specifically for (a) how to extend the system to cities, places, and weather, and (b) how to improve precision. We address each in turn.

### 4.1 Adding locations and weather

The current system treats the world as a flat image-text embedding space. To make it location- and weather-aware, we need three additions.

**Step 1 — Enrich captions with structured context.**

Today each image has one BLIP caption. We add three more fields, produced by a vision-language model (LLaVA-1.5, Florence-2, or CogVLM) running over the same images:

- `scene_type` — indoor / urban / coastal / forest / desert / snow
- `weather` — sunny / rainy / overcast / snowy
- `inferred_region` — coarse geo-tag from a geolocation head (country-level, not exact)

These join the existing caption in the index; they can also be stored as a structured side-table for hard filtering.

**Step 2 — Add a geo-temporal prior.**

For each `region` and `month`, precompute a weather prior from a public API (Open-Meteo is free and no-key). At query time, parse the query for location tokens ("Mumbai in August") and intersect with the prior to re-weight candidates that match expected conditions.

**Step 3 — Extend the query API.**

Add an optional query-time filter:

```
/search?q=beachwear&city=mumbai&month=august
```

The search runs as today, but candidates are first filtered (or down-weighted) by the geo-temporal prior. This is the natural extension for "shop the look in my city" applications on Glance's lock-screen.

**When to use it.** This extension is high-value when the catalogue is geographically diverse and the user wants outfit recommendations that are realistic for their context. It is low-value when the catalogue is style-only (e.g., a designer look-book) and location is irrelevant.

### 4.2 Improving precision

A precision roadmap, ordered by ROI:

| Lever | Mechanism | Expected gain | Cost |
|---|---|---|---|
| **Raise `rerank_top_n`** | One-line config change; re-rank from top-200 hybrid candidates instead of top-100 | +5–10 % on context-heavy queries (Q2, Q4) | Negligible |
| **Larger encoder** | Switch from `Marqo/marqo-fashionCLIP` (ViT-L/14) to a ViT-G/14 fashion backbone | +2–5 nDCG@10 on public benchmarks | Larger checkpoint, slower indexing |
| **Hard-negative mining + fine-tune** | Mine compositional swaps in the corpus ("red tie + white shirt" vs. "white tie + red shirt"); contrastive-fine-tune for ~1k steps on these pairs | +5–8 on Q5 specifically | Requires GPU, held-out validation set |
| **Late-interaction over caption tokens** | ColBERT-style token-level max-similarity between query and caption tokens | +3–6 on compositional queries | 100× storage cost; slower query time |
| **Generative re-ranker (LLM judge)** | LLaVA-1.6 evaluates each `(query, image)` pair, outputs a relevance score in {0, 1} | +5–10 | Much higher latency per query |
| **Active-learning feedback loop** | Capture click-through on top-k results; add positive pairs to monthly retraining | +1–3 per cycle, compounding | Requires live traffic |
| **Richer captions** | Replace BLIP-base with LLaVA or GPT-4V; produce 3–5 captions per image (different prompts) | +2–4 | Slower indexing, higher API cost |
| **Structured product metadata** | Add brand, title, price, category as side-text encodings; hybrid at index time | +1–3 | Requires a clean metadata pipeline |

**The single highest-ROI change** is hard-negative fine-tuning. The model already has fashion priors; teaching it to distinguish *"red tie + white shirt"* from *"white tie + red shirt"* needs labelled pairs, not a new architecture. The current code is structured to slot a fine-tuned model in by changing one config field.

### 4.3 Other directions

- **Image-to-image search.** Add `--image <path>` to `retriever/search.py` for visual-similarity lookups using the same index. Useful for "more like this".
- **Hard-negative dataset construction.** A small, carefully-labelled compositional dataset (~1k pairs) would unblock fine-tuning and serve as a held-out evaluation set.
- **Streaming catalogue updates.** Replace file-based metadata with a vector DB that supports live upserts (Milvus, Qdrant).

---

## Appendix A — Codebase

**GitHub:** [github.com/amanraj74/Glance-fashion-search](https://github.com/amanraj74/Glance-fashion-search)

The repository contains the full pipeline: indexer, retriever, scripts, Streamlit demo, tests, and this writeup. README at the repo root covers installation, configuration, and command reference.

---

## Appendix B — How to reproduce

All commands assume the repository root and an activated virtual environment.

```bash
# 1. Install dependencies
python -m venv venv
source venv/bin/activate          # or .\venv\Scripts\Activate.ps1 on Windows
pip install --upgrade pip
pip install -r requirements.txt
pip install pytest streamlit

# 2. Build the image index (downloads ~1 GB of fashionCLIP weights)
python indexer/build_index.py

# 3. Generate BLIP captions and embed them (~500 MB BLIP-base weights)
python scripts/build_caption_index.py

# 4. Run the 5-query evaluation harness
python scripts/run_eval.py

# 5. Run the A/B ablation comparison
python scripts/run_ablation.py

# 6. Render this writeup to HTML / PDF
python scripts/build_report.py

# 7. Launch the interactive web demo
streamlit run app/streamlit_app.py
```

**Expected wall-clock on CPU:**

| Step | Time | Output |
|---|---|---|
| `indexer/build_index.py` | 15–25 min | `output/faiss.index` |
| `scripts/build_caption_index.py` | 30–60 min | `output/captions.json`, `output/captions.index` |
| `scripts/run_eval.py` | < 1 min | `eval/results/*.json`, `eval/results/*.png` |
| `scripts/run_ablation.py` | < 1 min | `eval/results/ablation.csv` |
| `scripts/build_report.py` | < 30 s | `report/Glance_Internship_Report.html`, `report/Glance_Internship_Report.pdf` |

**Tests:**

```bash
pytest tests/ -v
```

21 tests pass in ~10 seconds. No model download required.

---

## Appendix C — Honest tradeoffs

- **Compute.** The project was developed on a CPU-only machine. No fine-tuning was performed; the architectural scaffolding for fine-tuning is ready but unexercised.
- **Dataset size.** 3,200 images is a small catalogue. Production would target ≥100k and would benefit from `IndexIVFFlat` or a managed vector DB.
- **Auto-evaluation is relative.** Without labelled ground truth, the ablation numbers compare configurations against each other, not against a fixed "correct answer". A production team should pair this with click-through metrics from a live A/B test.
- **Hard-negative fine-tune deferred.** Documented in §4.2 as the highest-ROI precision lever, but it requires GPU access and a labelled compositional dataset.

---

*This writeup is the deliverable required by §5 of the assignment brief. The accompanying code is the deliverable required by §5.3.*