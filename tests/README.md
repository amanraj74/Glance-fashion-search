# Tests

Pytest unit + integration tests for the `glance_search` package.

```powershell
.\venv\Scripts\python.exe -m pytest tests/ -v
```

Tests are designed to **not require any model download** — they use synthetic
embeddings, stub models, and tiny in-memory fixtures.

## Layout

- `test_config.py` — config loading, env overrides, dataclass integrity
- `test_errors_logging.py` — exception hierarchy + logger setup
- `test_embedder.py` — image listing, error paths
- `test_index_store.py` — FAISS round-trip, IVFFlat, missing-artifact error
- `test_pipeline.py` — end-to-end pipeline with stub model
- `test_reranker.py` — re-ranker with stub cross-encoder
