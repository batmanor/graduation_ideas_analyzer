import logging
from contextlib import asynccontextmanager

import asyncio
from fastapi import FastAPI

from .api.v1.router import api_router
from .core.config import settings
from .core.container import AppContainer
from .core.database import async_session, engine
from .core.logging import configure_logging
from .models import Base
from .services.llm_service import GeminiLLMService
from .services.paper_service import PaperService


configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    app.state.container = AppContainer(llm_service=GeminiLLMService())

    def prewarm_model():
        embedding_backend = app.state.container.get_embedding_backend()
        get_model = getattr(embedding_backend, "get_model", None)
        if get_model is not None:
            get_model()

    if settings.PREWARM_EMBEDDING_MODEL:
        try:
            await asyncio.to_thread(prewarm_model)
            logger.info("Embedding model prewarmed")
        except Exception as e:
            logger.exception("Failed to prewarm embedding model: %s", e)
            raise RuntimeError("Model initialization failed") from e
    else:
        logger.info("Embedding model prewarm disabled")

    if settings.AUTO_REBUILD_INDEX_ON_STARTUP:
        try:
            async with async_session() as session:
                paper_service = PaperService(session, app.state.container.llm_service)
                await app.state.container.get_vector_store().full_rebuild(paper_service)
            logger.info("FAISS index rebuilt from database")
        except Exception as e:
            logger.exception("Failed to rebuild FAISS index on startup: %s", e)
            raise RuntimeError("FAISS startup rebuild failed") from e

    logger.info("Application startup complete")

    yield

    logger.info("Shutting down application")

    try:
        await app.state.container.persist_vector_store_if_loaded()
        logger.info("FAISS index persisted if it was loaded")
    except Exception as e:
        logger.exception("Failed to persist FAISS index on shutdown: %s", e)


app = FastAPI(title="Multilingual Paper Idea Validator", lifespan=lifespan)

app.include_router(api_router, prefix="/api/v1")
