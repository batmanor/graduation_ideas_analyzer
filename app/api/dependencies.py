from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.container import AppContainer
from app.services import PaperService, VectorStoreService, ValidationService
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import GeminiLLMService


# ── Singleton accessors (O(1), no construction) ───────────────────────────────

def get_container(request: Request) -> AppContainer:
    """Return the app-level singleton container stored during lifespan startup."""
    return request.app.state.container


def get_embedding_service(
    container: Annotated[AppContainer, Depends(get_container)],
) -> EmbeddingService:
    return container.embedding_service


def get_faiss_mgr(
    container: Annotated[AppContainer, Depends(get_container)],
) -> VectorStoreService:
    return container.vector_store


def get_llm_service(
    container: Annotated[AppContainer, Depends(get_container)],
) -> GeminiLLMService:
    return container.llm_service


# ── Per-request service factories ─────────────────────────────────────────────

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


# ── Annotated shorthands used by endpoint files ───────────────────────────────

PaperServiceDep = Annotated[PaperService, Depends(get_paper_service)]
ValidationServiceDep = Annotated[ValidationService, Depends(get_validation_service)]
VectorStoreDep = Annotated[VectorStoreService, Depends(get_faiss_mgr)]