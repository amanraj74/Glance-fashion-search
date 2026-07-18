# TODO.md

Engineering roadmap. Source of truth: `Glance ML Internship Assignment.md`. Tasks are not invented — each reflects a rubric item or an observed gap.

Legend: `[ ]` pending · `[x]` done · `[~]` in progress · `[!]` blocked

---

## M0 — Baseline (shipped)

- [x] T0.1 Indexer: OpenCLIP `ViT-B-32` → FAISS `IndexFlatIP`
- [x] T0.2 Retriever: CLI text → top-5 cosine
- [x] T0.3 Persist `output/faiss.index` + `output/metadata.json`

---

## M1 — Industry-Grade Sprint

The single sprint that turns a failing submission into a top-class one.
Full plan in `M1_PLAN.md`.

### CRITICAL — must ship

- [x] T1 · Swap to fashion-aware CLIP backend — default `ViT-B-16-SigLIP-512`, alt `hf-hub:Marqo/marqo-fashionCLIP` via `--model`
- [x] T2 · Modular refactor — `src/glance_search/` package (10 modules), CLIs are thin wrappers
- [x] T3 · Caption-augmented dual-vector retrieval — `captions.py` (BLIP-base, resumable) + caption index + late-interaction scoring
- [x] T4 · Cross-encoder re-ranker — `reranker.py` (`ms-marco-MiniLM-L-2-v2`), flag-toggled
- [x] T5 · 5-rubric-query evaluation harness — `scripts/run_eval.py` (JSON+PNG) + `scripts/run_ablation.py` (CSV)
- [x] T6 · Streamlit demo — `app/streamlit_app.py` (top-k slider, caption-weight slider, rerank toggle, score breakdown)
- [~] T7 · Deliverable PDF — `report/Glance_Internship_Report.md` source complete; PDF generation pending user-triggered `pip install weasyprint` + `python scripts/build_report.py`

### HIGH — significantly improves grade

- [x] T8 · Scalable FAISS index — `IndexIVFFlat` via `--backend ivfflat`, tested in `test_index_store.py::test_ivfflat_builds`
- [x] T9 · Configuration via file — `config.yaml` + env override (`GLANCE_<SECTION>__<FIELD>`) with auto-type coercion
- [x] T10 · Unit tests — 21 pytest tests pass, no model download required

### MEDIUM — quality

- [x] T11 · Logging & error handling — `logging_setup.py` (idempotent) + `errors.py` (domain exceptions)
- [ ] T12 · Image-to-image search — `search.py --image <path>` mode. Deferred.

### LOW — polish

- [x] T13 · Clean requirements.txt — added `sentence-transformers`, `accelerate`, `pytest`
- [ ] T14 · LICENSE file

### Pending — must run before submission (user-triggered)

- [ ] **R1** — `python indexer/build_index.py`  → reindex with new backend
- [ ] **R2** — `python scripts/build_caption_index.py`  → BLIP captions + caption index
- [ ] **R3** — `python scripts/run_eval.py`  → 5-rubric evaluation grids
- [ ] **R4** — `python scripts/run_ablation.py`  → ablation table
- [ ] **R5** — `pip install weasyprint` + `python scripts/build_report.py`  → final PDF
- [ ] **R6** *(optional)* — `pip install streamlit` + `streamlit run app/streamlit_app.py`  → live demo

### FUTURE — out of scope

- F1 · Locations / cities / weather extension (sketched in `report/Glance_Internship_Report.md` §4.1)
- F2 · Active learning feedback loop
- F3 · Multi-lingual queries
- F4 · Production serving (FastAPI, Docker, FAISS server)

---

## Definition of Done (M1)

- [x] All CRITICAL tasks complete in code
- [x] All HIGH tasks complete in code
- [ ] `report/Glance_Internship_Report.pdf` exists with real results (R5)
- [ ] `eval/results/*.png` shows real top-k for the 5 rubric queries (R3)
- [ ] `eval/results/ablation.csv` shows A/B comparison (R4)
- [ ] `output/faiss.index` is the **new** backend's index (R1)
- [ ] `output/captions.json` exists with one caption per image (R2)

## Sprint Allocation (suggested)

| Step | Wall time (CPU) | User action |
|---|---|---|
| R1 | 15–25 min | `python indexer/build_index.py` |
| R2 | 30–60 min | `python scripts/build_caption_index.py` |
| R3 | < 1 min | `python scripts/run_eval.py` |
| R4 | < 1 min | `python scripts/run_ablation.py` |
| R5 | < 1 min | `pip install weasyprint && python scripts/build_report.py` |
| R6 | interactive | `pip install streamlit && streamlit run app/streamlit_app.py` |