# Load and Resource Tests

These tests are intentionally separate from `pytest` because they exercise a
running service and the local embedding model.

## Initial Model and FAISS Memory Probe

Run this before deploying to a 500 MB host:

```powershell
uv run python tests/load/resource_probe.py --max-rss-mb 500
```

The probe reports RSS after:

- process start
- embedding service construction
- model load
- first embedding call
- vector store and FAISS index load
- first FAISS add/search/persist cycle

It exits with a non-zero status if RSS exceeds the configured limit.

## 300 Concurrent Validation Users

Start the API first, then run:

```powershell
k6 run tests/load/k6_validate_300.js
```

This checks:

- validation latency
- request error rate
- periodic `/api/v1/metrics/` availability
- in-app timings for embedding, FAISS search, vector store search, validation,
  and paper lookup
- process RSS from the metrics endpoint

## 300 Readers plus 10 Writers

Start the API first, then run:

```powershell
k6 run tests/load/k6_mixed_300_read_10_write.js
```

This keeps write pressure capped at 10 concurrent users while 300 users validate
ideas.
