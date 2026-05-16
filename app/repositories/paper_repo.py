from typing import Any
from sqlalchemy import func
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models import Paper
from sqlalchemy.exc import IntegrityError


class PaperRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_all_papers(self, skip: int = 0, limit: int = 100):
        stmt = select(Paper).offset(skip).limit(limit)
        papers = await self.session.execute(stmt)
        return papers.scalars().all()

    async def get_paper_by_id(self, paper_id: int):
        return await self.session.get(Paper, paper_id)

    async def get_paper_by_external_id(self, external_id: int):
        stmt = select(Paper).where(Paper.external_id == external_id)
        paper = await self.session.execute(stmt)
        return paper.scalar_one_or_none()
    
    async def get_papers_by_external_ids(self,external_ids: list[int])-> None| list[Paper]:
        if not external_ids:
            return []
        
        stmt = select(Paper).where(Paper.external_id.in_(external_ids))
        res =  await self.session.execute(stmt)
        return list(res.scalars())
        
    async def get_all_external_ids(self) -> set[int]:
        stmt = select(Paper.external_id)
        res = await self.session.execute(stmt)
        return {row[0] for row in res.all()}

    async def count_papers(self) -> int:
        stmt = select(func.count(Paper.id))
        res = await self.session.execute(stmt)
        return res.scalar_one()


    async def create_paper(
        self, 
        external_id: int, 
        title: str, 
        abstract: str, 
        keywords: str
    ) -> Paper | None:
        paper = Paper(
            external_id=external_id,
            title=title,
            abstract=abstract,
            keywords=keywords,
        )

        self.session.add(paper)

        try:
            await self.session.commit()
            await self.session.refresh(paper)
            return paper

        except IntegrityError:
            await self.session.rollback()
            return await self.get_paper_by_external_id(external_id)

        except Exception:
            await self.session.rollback()
            raise

    async def update_paper(self, paper_id: int, data: dict[str, Any]) -> Paper | None:
        ALLOWED_FIELDS = {"title", "abstract", "keywords"}

        db_paper = await self.get_paper_by_id(paper_id)
        if not db_paper:
            return None
        
        for key, value in data.items():
            if key in ALLOWED_FIELDS:
                setattr(db_paper, key, value)

        await self.session.commit()
        await self.session.refresh(db_paper)

        return db_paper

    async def delete_paper(self, paper_id: int):
        paper = await self.get_paper_by_id(paper_id)
        if not paper:
            return False
        try:
            await self.session.delete(paper)
            await self.session.commit()
            return True
        except Exception as e:
            await self.session.rollback()
            raise e
