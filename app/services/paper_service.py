from fastapi import HTTPException

from app.repositories import PaperRepository
from app.models import Paper
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.paper import PaperCreate, PaperUpdate
from app.services.llm_service import GeminiLLMService


class PaperService:
    def __init__(self, db: AsyncSession, llm_service: GeminiLLMService):
        self.db = db
        self.repository = PaperRepository(db)
        self.llm_service = llm_service

    # -------------Read-------------
    async def get_all_papers(
        self, skip: int = 0, limit: int | None = 100
    ) -> list[Paper]:
        return await self.repository.get_all_papers(skip, limit)

    async def get_paper_by_id(self, paper_id: int) -> Paper | None:
        return await self.repository.get_paper_by_id(paper_id)

    async def get_paper_by_external_id(self, external_id: str) -> Paper | None:
        return await self.repository.get_paper_by_external_id(external_id)

    async def get_papers_by_ids(self, ids: list[int]) -> list[Paper]:
        return await self.repository.get_papers_by_ids(ids)
    
    async def get_papers_by_external_ids(
        self, external_ids: list[str]
    ) -> list[Paper]:
        return await self.repository.get_papers_by_external_ids(external_ids)

    async def get_all_external_ids(self) -> set[str]:
        return await self.repository.get_all_external_ids()
    
    async def get_all_ids(self) -> set[int]:
        return await self.repository.get_all_ids()

    async def count_papers(self) -> int:
        return await self.repository.count_papers()

    # -------------Write-------------
    async def create_paper(self, paper: PaperCreate) -> Paper | None:
        if paper.keywords is None:
            try:
                keywords = await self.llm_service.generate_keywords_async(
                    paper.title, paper.abstract
                )
            except Exception as e:
                raise HTTPException(
                    503,
                    "Couldn't contact with LLM to generate keywords, try adding keywords by yourself.",
                ) from e
        else:
            keywords = paper.keywords

        return await self.repository.create_paper(
            paper.external_id, paper.title, paper.abstract, keywords
        )

    async def update_paper(self, paper_id: int, paper: PaperUpdate) -> Paper | None:
        data = paper.model_dump(exclude_unset=True)
        if not data:
            return await self.repository.get_paper_by_id(paper_id)
        return await self.repository.update_paper(paper_id, data)

    async def delete_paper(self, paper_id: int) -> bool:
        return await self.repository.delete_paper(paper_id)
    
    async def upsert_paper_by_external_id(self, external_id: str, data: dict) -> Paper:
        return await self.repository.upsert_by_external_id(external_id, data)