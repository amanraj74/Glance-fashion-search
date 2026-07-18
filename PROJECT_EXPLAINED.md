# About This Project — A Plain-English Explainer

This document explains, in simple terms, what the **Glance Fashion Search** project is, why it exists, what was built during the internship, and what was learned. It is written for someone who is technically curious but hasn't worked on multimodal search before.

If you read this top-to-bottom, you should be able to explain the project to anyone else afterwards.

---

## 1. The one-sentence summary

The project is a **search engine for fashion images**: you type a sentence describing an outfit, and it shows you the photos that match.

Example: type *"a red tie and a white shirt in a formal setting"* → get back five product photos, ranked by how well they match.

---

## 2. The assignment

This was a take-home assignment from the **Glance ML Internship**. Glance makes the lock-screen content layer on Android phones (it shows news, games, sports, and shopping content without unlocking the phone).

The brief said, roughly:

> Build a system that, given a sentence like *"bright yellow raincoat"* or *"blue shirt on a park bench"*, returns the most relevant product images from a catalogue. The system should handle fashion-specific vocabulary (silk, denim, formal, casual), combine multiple attributes in one query (colour + clothing + location), and beat a vanilla baseline.

A copy of the full brief is at [`Glance ML Internship Assignment.md`](Glance%20ML%20Internship%20Assignment.md).

### Why this matters to Glance

Glance's lock-screen surfaces "shop the look" suggestions. For that feature to feel magical, it needs to understand *intent* — "show me casual weekend outfits that look good in Mumbai in August" — not just exact keywords. The search engine behind that experience is exactly what this project is a small-scale prototype of.

---

## 3. Why is this actually hard?

A naive approach is to use **CLIP** (a model from OpenAI that maps images and text into the same vector space). It works surprisingly well for general queries. It also fails in well-known ways for fashion:

1. **Compositionality** — give it *"red tie and white shirt"* and it often returns *"red shirt and white tie"*. It knows the words "red" and "tie" but doesn't keep them paired correctly.
2. **Fashion vocabulary** — terms like *"silk"*, *"plaid"*, *"tailored"*, *"athleisure"* are rare in the general web data CLIP was trained on. It under-recognises them.
3. **Multi-attribute queries** — *"blue shirt on a park bench"* needs both the garment and the setting to bind together. Plain CLIP mixes them up.

The assignment explicitly warned: "vanilla application of CLIP will fail; build something better." So the goal wasn't just to use CLIP — it was to extend it.

---

## 4. The approach in plain English

The final system is a **three-stage pipeline**. Each stage fixes a different weakness of the previous one.

### Stage 1 — Find visually similar images (CLIP)

Embed every catalogue image as a 768-dimensional vector using **fashion-aware CLIP** (`Marqo/marqo-fashionCLIP`, a CLIP variant fine-tuned on fashion data). Embed the user's query the same way. Find the nearest images by cosine similarity.

This stage is good at recognising individual attributes ("yellow", "raincoat", "park bench") but bad at binding them together.

### Stage 2 — Add a "caption lens" (BLIP)

Before searching, run a vision-language model (**BLIP-base**) over every catalogue image and write a one-sentence description: *"a man in a navy blazer standing in a glass-walled office"*. Embed those captions too, into a second index.

At query time, score the query against **both** the image and the caption. Combine the scores. This is the load-bearing trick: now the same image is represented through two lenses — raw pixels and natural language — and natural language happens to preserve attribute binding much better than pixels alone.

### Stage 3 — Re-rank the shortlist (cross-encoder)

The first two stages give a ranked list of, say, the top 100 candidates. Run a small **cross-encoder** model (a different transformer trained to read two pieces of text and judge their relevance) over each `(query, caption)` pair. Promote the best fits.

This stage catches the cases where stage 2 was *almost right* but missed by a few ranks. It costs ~30 ms per query but bumps precision meaningfully.

### Net effect

The same query goes from returning mediocre results on plain CLIP to returning highly relevant images on the full pipeline. Concretely, on the assignment's five test queries:

| Pipeline version | Average score on top-1 result |
|---|---:|
| Image only (the vanilla baseline) | 0.10 |
| Image + BLIP captions | 0.33 |
| Image + captions + cross-encoder rerank | **0.44** |

That's a **4× improvement**. On the hardest query — *"bright yellow raincoat"* — the score goes from 0.10 to 0.66, a 6.4× lift.

---

## 5. What was actually built

### The catalogue

3,200 product images scraped or assembled from open fashion datasets. They cover variations in **environment** (offices, streets, parks, homes), **clothing type** (formal, casual, outerwear), and **colour**. The catalogue is committed to a local `dataset/` folder but is git-ignored because of size.

### The codebase

A small, modular Python package called `glance_search`. It is split into clean responsibilities:

```
src/glance_search/
├── config.py        YAML + env-override config — single source of truth
├── model.py         fashion-CLIP wrapper with caching
├── embedder.py      batched image embedding
├── captions.py      BLIP caption generation (resumable)
├── reranker.py      cross-encoder re-ranker
├── index_store.py   FAISS wrapper (flat / IVFFlat)
├── pipeline.py      end-to-end search orchestration
├── errors.py        domain exceptions
└── logging_setup.py logger config
```

Plus thin command-line entry points:

```
indexer/build_index.py        embed images → save FAISS index
retriever/search.py           text query → top-k images
scripts/build_caption_index.py BLIP captions + caption index
scripts/run_eval.py           run 5 rubric queries, save PNG grids
scripts/run_ablation.py       A/B compare 3 configurations
scripts/build_report.py       render writeup → HTML/PDF
app/streamlit_app.py          interactive web demo
```

### The tests

21 pytest tests that run in 10 seconds with no model downloads. They use synthetic embeddings and stub models, so a reviewer can clone the repo, install, and verify the code is correct in under a minute.

### The writeup

A 4-section PDF (`report/Glance_Internship_Report.pdf`) covering:

1. **Approaches considered** — six possible solutions, with tradeoffs and verdicts
2. **Chosen approach** — the three-stage pipeline explained in detail
3. **Evaluation results** — per-query numbers, observations, failure-mode discussion
4. **Future work** — how to extend to locations/weather, how to improve precision

### The web demo

A Streamlit app where you can type a query, slide the weight between image and caption similarity, toggle the re-ranker, and see per-result score breakdowns. Useful for demos and for intuition-building.

---

## 6. How well it works

The assignment specifies five queries to be judged on. Results:

| Query | Top-1 score (full pipeline) |
|---|---:|
| A person in a bright yellow raincoat. | **0.66** |
| Professional business attire inside a modern office. | 0.29 |
| Someone wearing a blue shirt sitting on a park bench. | **0.64** |
| Casual weekend outfit for a city walk. | 0.29 |
| A red tie and a white shirt in a formal setting. | 0.31 |

The strong rows (0.6+) are queries where the **visual concept is concrete** — a yellow raincoat, a blue shirt on a bench. The weaker rows (0.3) are **context-heavy, style-inference** queries ("business attire in a modern office", "casual weekend for a city walk") where no single garment dominates the image. The system still finds reasonable outfits for these, but the score is lower because the catalogue doesn't contain many "obviously" office or "obviously" casual examples.

The biggest single failure mode: when the catalogue itself is missing the right image, no amount of clever ranking will find it. The reranker can only choose among candidates it's given. This is a **catalogue coverage** problem, not an algorithm problem.

---

## 7. What was learned

### About ML

- **CLIP is a strong baseline but not enough.** For compositional queries, you need a second view of the data — captions, tags, or another embedding space.
- **Caption augmentation is a cheap, big win.** Running BLIP once over the corpus and adding a parallel index is one of the highest-ROI things you can do for fashion search.
- **Cross-encoder rerank is great when the shortlist is good, mediocre when it isn't.** If the reranker's input is dominated by irrelevant items, it can't promote the relevant ones it never sees. The fix is "rerank more candidates," not "rerank better."
- **Fine-tuning would help but isn't required.** The hardest queries — "red tie + white shirt" — would benefit from contrastive fine-tuning on compositional swaps. The codebase is structured for it; it just needs labelled data and a GPU.

### About engineering

- **Separate the ML logic from the engineering plumbing.** The package boundary (`src/glance_search/`) keeps model loading, embedding, indexing, and search logic testable and reusable. The CLI scripts (`indexer/`, `retriever/`) are thin wrappers.
- **Configuration as data, not code.** All knobs — model name, index backend, weights, reranker settings — live in `config.yaml` and can be overridden by environment variables. No editing source code to change behaviour.
- **Resumability matters for long pipelines.** The caption-generation script saves progress every 10 batches. If you Ctrl-C after 30 minutes, you don't lose your work.
- **Test without the model.** The unit tests use synthetic embeddings and stub cross-encoders. They run in seconds and don't require Hugging Face auth or 1 GB of model downloads.

### About the assignment itself

- The brief explicitly rewards "thoughtful solution" over "lots of code." A six-approach comparison table with clear verdicts is worth more than a single working implementation with no reasoning.
- The rubric cares about zero-shot generalisation, modularity, and scalability, not about fine-tuning or external APIs. The work was scoped accordingly.

---

## 8. What would come next

If this project continued, the most valuable next steps are:

### Short term (1–2 days each)

- **Hard-negative fine-tune.** Mine compositional swaps from the corpus (e.g., images with both red and white clothing), create contrastive pairs, fine-tune for 1k steps. Should improve compositional queries by another 5–10 points.
- **Richer captions.** Replace BLIP-base with LLaVA-1.5 or GPT-4V. Multiple captions per image (one per region) would give the reranker more to work with.
- **Active-learning feedback loop.** Capture click-through on top-5 results, treat clicks as positive signals, retrain monthly.

### Medium term (1–2 weeks)

- **Locations + weather.** Add structured tags (scene type, weather, region) per image. At query time, intersect with geo-temporal priors (Open-Meteo) to filter candidates by "what's plausible for this user, here, now."
- **Structured metadata.** Add brand, title, price, category as side-text embeddings. Hybrid at index time.
- **Image-to-image search.** Allow visual-similarity lookups — "find me more like this one" — using the same index.

### Long term

- **Managed vector DB.** Replace local FAISS with Milvus, Qdrant, or Pinecone for live catalogue updates, distributed search, and operational observability.
- **FastAPI service.** Wrap `pipeline.search` in an HTTP endpoint with caching, rate limiting, and a metrics layer.

These are documented in detail in [`report/Glance_Internship_Report.md` § 4](report/Glance_Internship_Report.md#4-future-work).

---

## 9. Glossary

If any of the terms above were unfamiliar, here's a one-line definition for each:

- **CLIP** — A neural network from OpenAI that learns to match images with their captions. Used as the backbone here.
- **OpenCLIP** — An open-source re-implementation of CLIP. Used here because it supports many more model variants.
- **fashionCLIP** — A CLIP variant fine-tuned on fashion data; recognises silk, denim, athleisure, etc. better than general CLIP.
- **BLIP** — A vision-language model that writes natural-language captions for images.
- **FAISS** — Facebook's library for fast similarity search over millions of vectors.
- **Cross-encoder** — A small transformer that reads two pieces of text together and outputs a relevance score. More accurate than comparing two separate embeddings, but slower.
- **Cosine similarity** — A way to measure how close two vectors are pointing. Used everywhere in embedding-based search.
- **Compositionality** — The ability to correctly bind attributes to their owners ("red tie" not "tie that's red somewhere").
- **Zero-shot** — Handling a query the model has never seen in training, just from its general understanding of language.
- **Reranking** — Taking the top candidates from a fast first-pass retrieval and reordering them with a more expensive, more accurate model.

---

## 10. Where to go next

| You want to… | Read this |
|---|---|
| Run the project yourself | [`RUN.md`](RUN.md) |
| See the user-facing README | [`README.md`](README.md) |
| Read the assignment writeup | [`report/Glance_Internship_Report.md`](report/Glance_Internship_Report.md) |
| Read the original brief | [`Glance ML Internship Assignment.md`](Glance%20ML%20Internship%20Assignment.md) |
| Browse the code | `src/glance_search/` and `indexer/`, `retriever/`, `scripts/` |
| Run the tests | `pytest tests/ -v` |
| Try the web demo | `streamlit run app/streamlit_app.py` |

---

*This document was written as part of the Glance ML Internship submission. The actual engineering work — indexer, retriever, captions, re-ranker, evaluation harness, and writeup — is the deliverable; this file is the explanation of that work for a non-specialist reader.*
