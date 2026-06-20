from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
from supabase import create_async_client, AsyncClient

from app.core.config import settings
from app.core.container import AppContainer
from app.core.database import get_db
from app.services import PaperService, ValidationService, VectorStoreService
from app.services.embedding_backend import EmbeddingBackend
from app.services.llm_service import GeminiLLMService


# Singleton accessors

async def get_supabase_client() -> AsyncClient:
    client = await create_async_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    try:
        yield client
    finally:
        # optional: close if the client supports it
        pass

def get_container(request: Request) -> AppContainer:
    """Return the app-level singleton container stored during lifespan startup."""
    return request.app.state.container


def get_embedding_service(
    container: Annotated[AppContainer, Depends(get_container)],
) -> EmbeddingBackend:
    return container.get_embedding_backend()


def get_faiss_mgr(
    container: Annotated[AppContainer, Depends(get_container)],
) -> VectorStoreService:
    return container.get_vector_store()


def get_llm_service(
    container: Annotated[AppContainer, Depends(get_container)],
) -> GeminiLLMService:
    return container.llm_service


# Per-request service factories


async def get_paper_service(
    db: AsyncSession = Depends(get_db),
    llm_service: GeminiLLMService = Depends(get_llm_service),
) -> PaperService:
    return PaperService(db, llm_service)


async def get_validation_service(
    paper_service: PaperService = Depends(get_paper_service),
    vector_store: VectorStoreService = Depends(get_faiss_mgr),
) -> ValidationService:
    return ValidationService(paper_service=paper_service, vector_store=vector_store)


# Annotated shorthands used by endpoint files

PaperServiceDep = Annotated[PaperService, Depends(get_paper_service)]
ValidationServiceDep = Annotated[ValidationService, Depends(get_validation_service)]
VectorStoreDep = Annotated[VectorStoreService, Depends(get_faiss_mgr)]
