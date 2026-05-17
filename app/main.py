import logging
from contextlib import asynccontextmanager

import asyncio
from fastapi import FastAPI

from .api.v1.router import api_router
from .core.container import AppContainer
from .core.database import engine
from .core.logging import configure_logging
from .models import Base
from .services.llm_service import GeminiLLMService
from .utils.download_model import ensure_model


configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application")

    logger.info("Ensuring embedding model is present")
    await asyncio.to_thread(ensure_model)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    app.state.container = AppContainer(llm_service=GeminiLLMService())
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
