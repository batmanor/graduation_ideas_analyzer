import logging
from contextlib import asynccontextmanager

import asyncio
from fastapi import FastAPI

from .api.v1.router import api_router
from .core.config import settings
from .core.container import AppContainer
from .core.database import engine
from .core.logging import configure_logging
from .models import Base
from .services.llm_service import GeminiLLMService


configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    app.state.container = AppContainer(llm_service=GeminiLLMService())

    def prewarm_model():
        app.state.container.get_embedding_service().get_model()

    if settings.PREWARM_EMBEDDING_MODEL:
        try:
            await asyncio.to_thread(prewarm_model)
            logger.info("Embedding model prewarmed")
        except Exception as e:
            logger.exception("Failed to prewarm embedding model: %s", e)
            raise RuntimeError("Model initialization failed") from e
    else:
        logger.info("Embedding model prewarm disabled")

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
