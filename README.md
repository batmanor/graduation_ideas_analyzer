# Idea Management System Backend

FastAPI backend for a multilingual research idea validator. The API stores research paper metadata, keeps a FAISS vector index of paper text, and checks whether a submitted idea is similar to existing work.

This project is suitable for a final-project website or Flutter app because it exposes a JSON REST API, interactive Swagger documentation, and a layered backend structure that is easy to extend.

## Quick Start

```powershell
uv sync
uv run fastapi dev app/main.py
```

Then open:

```text
http://127.0.0.1:8000/docs
```

Minimum `.env`:

```env
GEMINI_API_KEY=your_api_key_here
```

Use these main endpoints first:

- `POST /api/v1/papers/` to add a paper.
- `POST /api/v1/validate/` to check a new idea.
- `GET /api/v1/dashboard/` to inspect stored papers and index size.

Run tests:

```powershell
uv run pytest
```

Note: embeddings run locally through ONNX. If the model folder is missing, the app can download the configured model from Hugging Face unless Hugging Face offline flags are enabled.

## Features

- Add research papers with `external_id`, title, abstract, and optional keywords.
- Generate missing keywords with Gemini when an API key is configured.
- Store paper metadata in SQLite through SQLAlchemy async sessions.
- Embed multilingual paper text locally with the ONNX export of `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- Store and search normalized vectors with FAISS.
- Validate a submitted idea and return novelty status plus similar paper matches.
- Read simple dashboard data and FAISS synchronization status.
- Run low-cost tests without loading the embedding model, FAISS index, or Gemini.

## Architecture

```text
Client website / Flutter app
        |
        v
FastAPI app: app/main.py
        |
        v
API routers: app/api/v1/endpoints
        |
        v
Pydantic schemas: app/schemas
        |
        v
Services: paper, validation, local ONNX vector store, Gemini
        |
        v
Repositories: SQLAlchemy async database access
        |
        v
SQLite metadata + FAISS vector index
```

Current structure rating: `7/10`.

Strengths: clear FastAPI layering, separated endpoints, schemas, services, repositories, and models, and a compact scope for an IT final project.

Weaknesses: CORS and auth are not implemented yet, the test suite is new and intentionally small, local database/vector artifacts are at the repository root, and update/delete repository methods are not exposed as public API endpoints.

Recommended target after documentation, tests, and small cleanup: `8.5/10`.

## API Base

Local development:

```text
http://127.0.0.1:8000
```

API prefix:

```text
/api/v1
```

Interactive docs:

```text
http://127.0.0.1:8000/docs
```

## Current Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/v1/papers/` | Add a paper record and enqueue vector indexing |
| `POST` | `/api/v1/validate/` | Validate a new idea against indexed papers |
| `GET` | `/api/v1/dashboard/` | Return total paper count, index length, and paged papers |
| `GET` | `/api/v1/dashboard/papers` | Return paged papers |
| `GET` | `/api/v1/dashboard/index-contents` | Return external IDs currently stored in FAISS |
| `GET` | `/api/v1/faiss_sync/sync/status` | Compare database external IDs with FAISS IDs |
| `POST` | `/api/v1/faiss_sync/sync/` | Persist the current FAISS index |
| `POST` | `/api/v1/faiss_sync/sync/full` | Add missing vectors or rebuild when stale IDs exist |
| `POST` | `/api/v1/faiss_sync/sync/full-rebuild` | Rebuild the FAISS index from database papers |

## Request Examples

Add a paper:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/papers/" \
  -H "Content-Type: application/json" \
  -d '{
    "external_id": 1001,
    "title": "Multilingual Semantic Search for Research Papers",
    "abstract": "This paper studies multilingual sentence embeddings for research retrieval.",
    "keywords": "semantic search, multilingual, embeddings"
  }'
```

Response:

```json
{
  "id": 1,
  "external_id": 1001,
  "title": "Multilingual Semantic Search for Research Papers",
  "abstract": "This paper studies multilingual sentence embeddings for research retrieval.",
  "keywords": "semantic search, multilingual, embeddings"
}
```

Validate an idea:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/validate/" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Arabic and English Research Idea Validator",
    "abstract": "A system that checks whether a student research idea is similar to existing papers.",
    "keywords": "idea validation, research, multilingual"
  }'
```

Response:

```json
{
  "is_novel": true,
  "message": "Idea appears to be novel!",
  "similar_papers": []
}
```

## Website And Flutter Integration

Use the deployed server URL as the client base URL:

```text
https://your-backend.example.com/api/v1
```

Website and Flutter clients should send JSON:

```http
Content-Type: application/json
```

Flutter example shape:

```dart
final response = await http.post(
  Uri.parse('$baseUrl/validate/'),
  headers: {'Content-Type': 'application/json'},
  body: jsonEncode({
    'title': title,
    'abstract': abstract,
    'keywords': keywords,
  }),
);
```

For browser-based websites, CORS must be added before production deployment. The recommended future setting is an `ALLOWED_ORIGINS` environment variable and FastAPI `CORSMiddleware`.

For auth-ready clients, a future implementation can accept:

```http
Authorization: Bearer <token>
```

For a final-project demo only, a simple token can contain signed JSON data with a demo claim such as `"signature": "this token is safe!!"`. Use a server-side secret from `.env`. Do not expose real production secrets in a public website client because browsers cannot hide shared secrets.

## Local Embedding Model

The app uses a local ONNX embedding model with a smaller runtime stack:

- Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Default Hugging Face repo: `Mo-alhariri/paraphrase-multilingual-minilm-l12-v2-int8`
- Runtime library: `light-embed`
- Inference engine: `onnxruntime`
- Default local folder: `./models/paraphrase-multilingual-MiniLM-L12-v2`
- Default ONNX file inside that folder: `model.int8.onnx`

The model is loaded lazily on the first endpoint that needs embeddings. If `PREWARM_EMBEDDING_MODEL=true`, startup loads the model immediately and fails fast if the model cannot be loaded.

Expected local structure:

```text
models/
  paraphrase-multilingual-MiniLM-L12-v2/
    config.json
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
    1_Pooling/
      config.json
    model.int8.onnx
```

You can override the path and ONNX file in `.env`:

```env
REPO_ID=Mo-alhariri/paraphrase-multilingual-minilm-l12-v2-int8
EMBEDDING_MODEL_PATH=./models/paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_ONNX_FILE=model.int8.onnx
EMBEDDING_POOLING_CONFIG_PATH=1_Pooling
EMBEDDING_DIM=384
PREWARM_EMBEDDING_MODEL=false
```

Do not set `HF_HUB_OFFLINE=1` or `TRANSFORMERS_OFFLINE=1` unless the model files already exist at `EMBEDDING_MODEL_PATH`.

## Setup

Install `uv`, then install dependencies:

```powershell
uv sync
```

Create `.env` in the project root:

```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.5-flash
SIMILARITY_THRESHOLD=0.75
```

The embedding model is loaded from `EMBEDDING_MODEL_PATH`. If the folder is missing and Hugging Face downloads are allowed, the app downloads the model automatically on first embedding use.

## Run

Development server:

```powershell
uv run fastapi dev app/main.py
```

Tests:

```powershell
uv run pytest
```

Lint and format:

```powershell
uv run ruff check .
uv run ruff format .
```

## System Requirements

- Python `3.12+`
- SQLite
- Local ONNX folder for `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Gemini API key if keyword generation is needed
- Enough disk space for the virtual environment, model cache, FAISS index, and SQLite database

## Deployment Notes

For Railway, choose one model strategy:

- Allow the app to download from Hugging Face: do not set `HF_HUB_OFFLINE=1` or `TRANSFORMERS_OFFLINE=1`.
- Bundle/provision the model files at `EMBEDDING_MODEL_PATH`: then offline flags are safe.

`PREWARM_EMBEDDING_MODEL=false` is the default and lets the API boot before loading the embedding model. Set it to `true` only when the model is already present or Railway can reach Hugging Face during startup.

Previous local size observations from this project environment before the ONNX runtime change:

- `.venv`: about `0.86 GB`
- local Hugging Face cache: about `0.53 GB`
- `paraphrase-multilingual-MiniLM-L12-v2` cache: about `457.5 MB`

The dependency stack now removes `sentence-transformers`, `transformers`, `torch`, CUDA/NVIDIA packages, SciPy, and scikit-learn from `uv.lock`. The local model still consumes disk space, but app dependencies should be much smaller.

Keep hosted package size under about `3.5 GB`, but verify the real deployment size on the selected host. Free hosts count disk usage differently, and the final size can change when dependencies, model files, build cache, or Python wheels are included.

Do not deploy local secrets or artifacts:

- `.env`
- `papers.db`
- `vector_index.faiss`
- `.venv/`
- local model folders unless the host intentionally needs them

## Troubleshooting

`GEMINI_API_KEY` validation error:
Set `GEMINI_API_KEY` in `.env` or in the deployment environment.

Embedding model startup error:
If the error mentions `HF_HUB_OFFLINE` or `TRANSFORMERS_OFFLINE`, remove those variables from Railway or provide the model files at `EMBEDDING_MODEL_PATH`. Otherwise, place the local ONNX model folder at `EMBEDDING_MODEL_PATH`, or update `.env` to point to the folder that contains the tokenizer files, `1_Pooling`, and the selected ONNX file.

Browser request blocked:
CORS is not implemented yet. Add `CORSMiddleware` and configure allowed origins.

Duplicate `external_id`:
The repository returns the existing paper when a duplicate external ID is inserted.

FAISS index out of sync:
Use `/api/v1/faiss_sync/sync/status`, then run full sync or full rebuild.

## Project Limits

- No production authentication yet.
- No CORS configuration yet.
- No public update/delete paper endpoints yet.
- FAISS update/delete synchronization is not fully exposed through paper CRUD.
- Validation response currently returns concise matches only, not detailed evidence.
- Local runtime artifacts are still stored in the repository root.

## Future Features

Simple improvements:

- `GET /health` for website and Flutter connectivity checks.
- `GET /` root endpoint with API name, version, docs URL, and status.
- CORS configuration using `ALLOWED_ORIGINS`.
- Public JavaScript and Flutter examples.
- Pagination validation for `limit` and `offset`.

Mid-level improvements:

- Expose full paper CRUD: list, detail, update, delete.
- Keep FAISS synchronized when papers are updated or deleted.
- Add validation report details: matched abstracts, similarity threshold, top-k setting, and optional Gemini novelty explanation.
- Add import/export endpoints for seed papers.
- Add admin dashboard summary endpoint with counts, index sync status, and last rebuild time.

Auth-ready REST improvements:

- Demo Bearer-token auth for final-project use.
- Signed JSON token with a server-side secret from `.env`.
- Clear separation between demo auth and production security.
- Reminder that public website clients cannot securely hide shared secrets.

Target folder structure:

```text
idea_management_system_backend/
  app/
    api/
      v1/
        endpoints/
        router.py
    core/
      config.py
      database.py
      security.py        # future auth helper
    models/
    repositories/
    schemas/
    services/
    utils/
    main.py
  tests/
    test_api_contracts.py
    test_processing.py
    test_repositories.py
  data/                  # future local-only db/faiss artifacts
  AGENTS.md
  README.md
  pyproject.toml
  uv.lock
```

## References

- FastAPI bigger apps: https://fastapi.tiangolo.com/tutorial/bigger-applications/
- FastAPI CORS: https://fastapi.tiangolo.com/tutorial/cors/
- FastAPI testing: https://fastapi.tiangolo.com/tutorial/testing/
- FastAPI security/JWT: https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/
- SQLAlchemy asyncio: https://docs.sqlalchemy.org/20/orm/extensions/asyncio.html
- Pydantic Settings: https://docs.pydantic.dev/latest/api/pydantic_settings/
- uv commands: https://docs.astral.sh/uv/reference/cli/
- FAISS getting started: https://github.com/facebookresearch/faiss/wiki/getting-started
- SentenceTransformers install: https://sbert.net/docs/installation.html
- SentenceTransformer backends: https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html
- Gemini API libraries: https://ai.google.dev/gemini-api/docs/libraries
- Gemini text generation: https://ai.google.dev/gemini-api/docs/text-generation
- Flutter networking: https://docs.flutter.dev/cookbook/networking/fetch-data
