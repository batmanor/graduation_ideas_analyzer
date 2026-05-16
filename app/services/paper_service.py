from fastapi import HTTPException

from app.repositories import PaperRepository
from app.models import Paper
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
from app.services.llm_service import GeminiLLMService


class PaperService:
    def __init__(self, db: AsyncSession, llm_service: GeminiLLMService):
        self.db = db
        self.repository = PaperRepository(db)
        self.llm_service = llm_service

    # -------------Read-------------
    async def get_all_papers(self, skip: int = 0, limit: int = 100):
        return await self.repository.get_all_papers(skip, limit)

    async def get_paper_by_id(self, paper_id: int):
        return await self.repository.get_paper_by_id(paper_id)

    async def get_paper_by_external_id(self, external_id: int):
        return await self.repository.get_paper_by_external_id(external_id)

    async def get_papers_by_ids(self, external_ids: list[int]):
        return await self.repository.get_papers_by_external_ids(external_ids)

    async def get_all_external_ids(self):
        return await self.repository.get_all_external_ids()

    async def count_papers(self) -> int:
        return await self.repository.count_papers()

    # -------------Write-------------
    async def create_paper(
        self,
        external_id: int,
        title: str,
        abstract: str,
        keywords: Optional[str] = None,
    ) -> Paper:
        if keywords is None:
            try:
                keywords = await self.llm_service.generate_keywords_async(
                    title, abstract
                )
            except Exception as e:
                raise HTTPException(
                    404,
                    "Couldn't contact with LLM to generate keywords, try adding keywords by yourself.",
                ) from e

        return await self.repository.create_paper(
            external_id, title, abstract, keywords
        )

    async def update_paper(self, paper_id: int, data: dict) -> Paper | None:
        return await self.repository.update_paper(paper_id, data)

    async def delete_paper(self, paper_id: int):
        return await self.repository.delete_paper(paper_id)
