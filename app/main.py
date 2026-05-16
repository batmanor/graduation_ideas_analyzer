import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api.v1.router import api_router
from .core.container import AppContainer
from .core.database import engine
from .core.logging import configure_logging
from .models import Base
from .services.embedding_service import EmbeddingService
from .services.llm_service import GeminiLLMService
from .services.vector_store import VectorStoreService
from .utils.download_model import ensure_model


configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application")

    # ── 1. Database ──────────────────────────────────────────────────────────
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # ── 2. Embedding model (heavy — ONNX, ~100 MB) ───────────────────────────
    ensure_model()
    embedding_service = EmbeddingService()
    embedding_service.load_model()
    logger.info("Embedding model ready")

    # ── 3. FAISS vector store (loads index from disk) ─────────────────────────
    vector_store = VectorStoreService(embedding_service)
    logger.info("FAISS index ready (%d vectors)", len(vector_store))

    # ── 4. Gemini LLM client (HTTP client, reused across requests) ────────────
    llm_service = GeminiLLMService()
    logger.info("Gemini LLM client ready")

    # ── 5. Store all singletons in app.state ──────────────────────────────────
    app.state.container = AppContainer(
        embedding_service=embedding_service,
        vector_store=vector_store,
        llm_service=llm_service,
    )

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("Shutting down application")
    try:
        await app.state.container.vector_store.persist()
        logger.info("FAISS index persisted")
    except Exception as e:
        logger.exception("Failed to persist FAISS index on shutdown: %s", e)


app = FastAPI(title="Multilingual Paper Idea Validator", lifespan=lifespan)

app.include_router(api_router, prefix="/api/v1")
