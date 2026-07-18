# M1 — Industry-Grade Plan

**Project:** glance-fashion-search — multimodal fashion + context retrieval.
**North Star:** A submission that demonstrably **beats vanilla CLIP** on the 5 rubric queries, scales to 1M images, ships an interactive demo, and reads like a publication-grade system report.

---

## 1. Architecture (locked)

```
                       ┌────────────────────────────┐
                       │   Query decomposition      │
                       │   (no external models)     │
                       └──────────────┬─────────────┘
                                      │ tokens: [garment, color, scene, vibe]
                                      ▼
   ┌──────────────────────────────────────────────────────────┐
   │              Embedder: Fashion-aware CLIP                  │
   │   backend: ViT-B-16 SigLIP   (default)                    │
   │   alt:     hf-hub:Marqo/marqo-fashionCLIP (flag)          │
   └────┬─────────────────────────────────────────┬─────────────┘
        │ image emb (512/768)                     │ text emb
        ▼                                          ▼
   ┌────────────────────────┐         ┌─────────────────────────┐
   │ FAISS IndexFlatIP      │         │ FAISS IndexFlatIP        │
   │ images, dim=D          │         │ captions, dim=D          │
   └────────────────────────┘         └─────────────────────────┘
                                                  ▲
                                          caption emb per image
                                                  │
                           offline BLIP-base captioning (3.2k images)
                                                  │
                                          ┌──────────────────────┐
                                          │ BLIP-base            │
                                          │ (transformers)       │
                                          └──────────────────────┘

   Late-interaction scoring:
       score(q, img) = α · sim(q, img_emb)
                     + β · max_i sim(q, caption_emb_i)
                              │
                              ▼
            top-50 re-ranked by cross-encoder(ms-marco-MiniLM-L-2-v2)
                              │
                              ▼
                       top-5 returned
```

---

## 2. Why this beats vanilla CLIP

| Failure of baseline CLIP | Our defense |
|---|---|
| Fine-grained fashion attributes get muddled | Fashion-aware CLIP encoder (B-16 SigLIP / Marqo) |
| Compositionality ("red tie, white shirt" vs swap) | Dual-vector image+captions, late-interaction scoring |
| Single-similarity misses subtle mismatches | Cross-encoder re-ranker (ms-marco-MiniLM-L-2-v2) |
| `IndexFlatIP` won't scale to 1M | IVFFlat + PQ behind `--index ivfflat` flag |
| No proof of correctness | 5-rubric-query harness → JSON + PNG grids |

---

## 3. Phased Execution

| Phase | Goal | Deliverables | Est. wall time |
|---|---|---|---|
| **P1 Foundation** | Replace vanilla CLIP backend + clean code | Refactored `src/glance_search/` package, configurable embedder, reindexed index | ~45 min |
| **P2 Captions** | Inject text-side knowledge to defeat compositionality | BLIP captions over 3.2k images, caption index, late-interaction score | ~90 min |
| **P3 Re-ranker** | Boost precision on top-50 | sentence-transformers cross-encoder wired in, flag-toggled | ~30 min |
| **P4 Evaluation** | Prove it works | 5-rubric-query harness, JSON+PNG outputs, ablation table | ~45 min |
| **P5 Demo** | Impressive delivery | Streamlit app, screenshots for PDF | ~45 min |
| **P6 Writeup** | Top-class PDF | 4-section PDF per assignment §5, architecture, ablations, qualitative grids, future-work sketches | ~3 h |
| **P7 Polish** | Production hygiene | tests, IVFFlat wrapper, .env, LICENSE | ~1 h |

**MVP definition** (a passing submission already):
- P1 + P2 + P4 + P6 done. ~6 hours.

**Stretch** (sets a new bar):
- All phases done. ~10–12 hours.

---

## 4. Files this plan creates

```
src/glance_search/
├── __init__.py
├── config.py                ← YAML + env, single source of truth
├── errors.py                ← domain exceptions
├── logging_setup.py         ← logger config
├── model.py                 ← CLIP wrapper (loads once, encode_image, encode_text)
├── embedder.py              ← batched image embedder, L2-normalization helpers
├── captions.py              ← BLIP captioning offline pipeline
├── reranker.py              ← cross-encoder re-ranker
├── index_store.py           ← FAISS wrappers (flat + ivfflat)
└── pipeline.py              ← high-level search orchestration

indexer/build_index.py       ← thin CLI over src
retriever/search.py          ← thin CLI over src
config.yaml                  ← runtime configuration
scripts/caption_corpus.py    ← runs BLIP over all images (P2)
scripts/run_eval.py          ← runs 5 rubric queries (P4)
app/streamlit_app.py         ← demo UI (P5)
report/Glance_Internship_Report.md  ← writeup source (P6)
report/Glance_Internship_Report.pdf ← final (P6)
tests/test_*.py              ← unit + integration (P7)
```

---

## 5. Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Marqo fashionCLIP not loadable from open_clip 3.3.0 | Med | Default to ViT-B-16 SigLIP, expose `--model` flag |
| CPU too slow for ViT-L/14 | High | Pick B-16 by default; L/14 only behind flag for GPU users |
| BLIP caption download fails or is huge | Low | Use `Salesforce/blip-image-captioning-base` (~500 MB) |
| Reindex outruns bash timeout | High | Run as background job with streamed logs |
| sentence-transformers / accelerate install fails | Low | Pin versions, fall back to sklearn for re-ranker |
| Captions hurt precision on some queries | Med | Ablation runs in P4 with/without captions |
| Eval has no labeled ground truth | High | Manual qualitative A/B + visual grid |

---

## 6. Open Decisions Resolved

| Decision | Choice | Rationale |
|---|---|---|
| Backend default | `ViT-B-16-SigLIP-512` (open_clip) | Best accuracy/speed ratio on CPU |
| Backend optional | `hf-hub:Marqo/marqo-fashionCLIP` | True fashion-tuned alternative |
| Caption model | `Salesforce/blip-image-captioning-base` | 500 MB, fast, strong captions |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-2-v2` | State-of-art precision, ~30 MB |
| Demo | Streamlit | Lightweight, Python-native, shareable |
| Index default | `IndexFlatIP` | Exact, fine at 3.2k |
| Index scale-up | `IndexIVFFlat` with PQ | Documented 1M-image path |

---

## 7. Done = all CRITICAL TODO items closed + PDF generated

Every phase ends with `pytest`, smoke check (`python retriever/search.py`), and the corresponding `CHANGELOG.md` + `PROJECT_STATUS.md` + `TODO.md` updates.
