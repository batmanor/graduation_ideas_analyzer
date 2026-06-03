import asyncio
from dataclasses import dataclass

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api import dependencies
from app.main import app as main_app
from app.api.v1.router import api_router
from app.schemas.validation import ValidationResponse


@dataclass
class PaperRecord:
    id: int
    external_id: int
    title: str
    abstract: str
    keywords: str | None = None


class FakePaperService:
    def __init__(self):
        self.papers: list[PaperRecord] = [
            PaperRecord(
                id=1,
                external_id=200,
                title="Seed Paper",
                abstract="Seed abstract",
                keywords="seed",
            )
        ]

    async def create_paper(self, external_id, title, abstract, keywords=None):
        paper = PaperRecord(
            id=len(self.papers) + 1,
            external_id=external_id,
            title=title,
            abstract=abstract,
            keywords=keywords,
        )
        self.papers.append(paper)
        return paper

    async def get_all_papers(self, skip=0, limit=100):
        return self.papers[skip : skip + limit]

    async def get_all_external_ids(self):
        return {paper.external_id for paper in self.papers}


class FakeVectorStore:
    def __init__(self):
        self.added: list[tuple[int, str]] = []
        self.persisted = False
        self.full_synced = False
        self.full_rebuilt = False

    def __len__(self):
        return len(self.added)

    async def index_paper(self, paper):
        self.added.append((paper.external_id, paper.title, paper.abstract))

    def get_contents(self):
        return [external_id for external_id, _, _ in self.added]

    async def check_sync(self, external_ids):
        faiss_ids = set(self.get_contents())
        return {
            "is_sync": external_ids == faiss_ids,
            "missing_from_faiss": sorted(external_ids - faiss_ids),
            "extra_in_faiss": sorted(faiss_ids - external_ids),
        }

    async def persist(self):
        self.persisted = True

    async def full_sync(self, paper_service):
        self.full_synced = True

    async def full_rebuild(self, paper_service):
        self.full_rebuilt = True


class FakeValidationService:
    async def validate_idea(self, idea):
        return ValidationResponse(
            is_novel=True,
            message=f"Validated {idea.title}",
            similar_papers=[],
        )


async def _build_client():
    app = FastAPI()
    app.include_router(api_router, prefix="/api/v1")

    paper_service = FakePaperService()
    vector_store = FakeVectorStore()

    app.dependency_overrides[dependencies.get_paper_service] = lambda: paper_service
    app.dependency_overrides[dependencies.get_faiss_mgr] = lambda: vector_store
    app.dependency_overrides[dependencies.get_llm_service] = lambda: (
        None
    )  # no real Gemini calls
    app.dependency_overrides[dependencies.get_validation_service] = lambda: (
        FakeValidationService()
    )

    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url="http://testserver")
    return client, vector_store


async def _exercise_api_contracts():
    client, vector_store = await _build_client()
    async with client:
        create_response = await client.post(
            "/api/v1/papers/",
            json={
                "external_id": 201,
                "title": "Client Paper",
                "abstract": "A paper from a client app.",
                "keywords": "client, api",
            },
        )
        validate_response = await client.post(
            "/api/v1/validate/",
            json={
                "title": "Idea",
                "abstract": "Idea abstract",
                "keywords": "idea keywords",
            },
        )
        dashboard_papers_response = await client.get("/api/v1/dashboard/papers")
        index_response = await client.get("/api/v1/dashboard/index-contents")
        sync_status_response = await client.get("/api/v1/faiss_sync/sync/status")
        persist_response = await client.post("/api/v1/faiss_sync/sync/")
        full_sync_response = await client.post("/api/v1/faiss_sync/sync/full")
        full_rebuild_response = await client.post(
            "/api/v1/faiss_sync/sync/full-rebuild"
        )

    return {
        "create": create_response,
        "validate": validate_response,
        "dashboard_papers": dashboard_papers_response,
        "index": index_response,
        "sync_status": sync_status_response,
        "persist": persist_response,
        "full_sync": full_sync_response,
        "full_rebuild": full_rebuild_response,
        "vector_store": vector_store,
    }


def test_api_contracts_with_dependency_overrides():
    result = asyncio.run(_exercise_api_contracts())

    assert result["create"].status_code == 201
    assert result["create"].json()["external_id"] == 201
    assert result["vector_store"].added == [
        (201, "Client Paper", "A paper from a client app.")
    ]

    assert result["validate"].status_code == 200
    assert result["validate"].json() == {
        "is_novel": True,
        "message": "Validated Idea",
        "similar_papers": [],
    }

    assert result["dashboard_papers"].status_code == 200
    assert [paper["external_id"] for paper in result["dashboard_papers"].json()] == [
        200,
        201,
    ]
    assert result["index"].status_code == 200
    assert result["index"].json() == [201]

    assert result["sync_status"].status_code == 200
    assert result["sync_status"].json() == {
        "is_sync": False,
        "missing_from_faiss": [200],
        "extra_in_faiss": [],
    }
    assert result["persist"].status_code == 200
    assert result["persist"].json() == {"message": "FAISS index synchronized to disk."}
    assert result["full_sync"].status_code == 200
    assert result["full_sync"].json() == {"message": "Full sync completed."}
    assert result["full_rebuild"].status_code == 200
    assert result["full_rebuild"].json() == {"message": "Full rebuild completed."}
    assert result["vector_store"].persisted is True
    assert result["vector_store"].full_synced is True
    assert result["vector_store"].full_rebuilt is True


def test_main_app_imports_successfully():
    assert main_app.title == "Multilingual Paper Idea Validator"
