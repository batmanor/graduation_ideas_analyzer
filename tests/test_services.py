import asyncio
import numpy as np
from types import SimpleNamespace

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.models import Base
from app.schemas.validation import IdeaSubmit
from app.services.paper_service import PaperService
from app.services.validation_service import ValidationService


class FakeGeminiLLMService:
    async def generate_keywords_async(self, title: str, abstract: str) -> str:
        return f"{title}, {abstract}, generated"


async def _create_service_with_temp_db(llm_service=None):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session = session_factory()
    svc = PaperService(session, llm_service or FakeGeminiLLMService())
    return engine, session, svc


async def _exercise_paper_service_create_with_provided_keywords():
    engine, session, service = await _create_service_with_temp_db()
    try:
        paper = await service.create_paper(
            external_id=301,
            title="Provided Keywords",
            abstract="Does not need Gemini",
            keywords="manual, keywords",
        )
        fetched = await service.get_paper_by_external_id(301)
        return paper, fetched
    finally:
        await session.close()
        await engine.dispose()


async def _exercise_paper_service_generates_missing_keywords():
    # Inject the fake LLM directly — no monkeypatching needed
    engine, session, service = await _create_service_with_temp_db(
        llm_service=FakeGeminiLLMService()
    )
    try:
        return await service.create_paper(
            external_id=302,
            title="Generated Keywords",
            abstract="Needs Gemini",
            keywords=None,
        )
    finally:
        await session.close()
        await engine.dispose()


def test_paper_service_creates_and_reads_paper_with_provided_keywords():
    paper, fetched = asyncio.run(
        _exercise_paper_service_create_with_provided_keywords()
    )

    assert paper.id == fetched.id
    assert fetched.external_id == 301
    assert fetched.keywords == "manual, keywords"


def test_paper_service_generates_keywords_when_missing():
    paper = asyncio.run(_exercise_paper_service_generates_missing_keywords())

    assert paper.external_id == 302
    assert paper.keywords == "Generated Keywords, Needs Gemini, generated"


class FakeVectorStore:
    async def search(self, text: str, top_k: int = 5):
        assert (
            text
            == "New Retrieval Idea. Compare with existing papers. retrieval, search"
        )
        assert top_k == 5
        return np.array([0.91, 0.4, 0.88]), np.array([11, 99, 12])


class FakePaperService:
    async def get_papers_by_ids(self, external_ids):
        assert external_ids == [11, 99, 12]
        return [
            SimpleNamespace(external_id=11, title="Close Paper"),
            SimpleNamespace(external_id=12, title="Another Close Paper"),
        ]


def test_validation_service_builds_similarity_matches_and_novelty_message():
    service = ValidationService(
        paper_service=FakePaperService(),
        vector_store=FakeVectorStore(),
    )

    response = asyncio.run(
        service.validate_idea(
            IdeaSubmit(
                title="New Retrieval Idea",
                abstract="Compare with existing papers",
                keywords="retrieval, search",
            )
        )
    )

    assert response.is_novel is False
    assert response.message == "Idea closely resembles existing work."
    assert [match.external_id for match in response.similar_papers] == [11, 12]
    assert [match.similarity_score for match in response.similar_papers] == [0.91, 0.88]
