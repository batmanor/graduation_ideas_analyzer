from sqlalchemy import func
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from app.models import Paper


class PaperRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_all_papers(self, skip: int = 0, limit: int | None = 100)->list[Paper]:
        stmt = select(Paper).offset(skip)
        if limit is not None:
            stmt = stmt.limit(limit)
        papers = await self.session.execute(stmt)
        return list(papers.scalars().all())

    async def get_paper_by_id(self, paper_id: int) -> Paper | None:
        return await self.session.get(Paper, paper_id)

    async def  get_papers_by_ids(
            self, ids: list[int]
    )-> list[Paper]:
        if not ids:
            return []
        
        stmt = select(Paper).where(Paper.id.in_(ids))
        res = await self.session.execute(stmt)
        return list(res.scalars())

    async def get_paper_by_external_id(self, external_id: str) -> Paper | None:
        stmt = select(Paper).where(Paper.external_id == external_id)
        paper = await self.session.execute(stmt)
        return paper.scalar_one_or_none()

    async def get_papers_by_external_ids(
        self, external_ids: list[str]
    ) -> list[Paper]:
        if not external_ids:
            return []

        stmt = select(Paper).where(Paper.external_id.in_(external_ids))
        res = await self.session.execute(stmt)
        return list(res.scalars())

    async def get_all_ids(self) -> set[int]:
        stmt = select(Paper.id)
        res = await self.session.execute(stmt)
        return {row[0] for row in res.all()}
    
    async def get_all_external_ids(self) -> set[str]:
        stmt = select(Paper.external_id)
        res = await self.session.execute(stmt)
        return {row[0] for row in res.all()}

    async def count_papers(self) -> int:
        stmt = select(func.count(Paper.id))
        res = await self.session.execute(stmt)
        return res.scalar_one()

    async def create_paper(
        self, external_id: str, title: str, abstract: str, keywords: str
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

    async def update_paper(self, paper_id: int, data: dict) -> Paper | None:
        db_paper = await self.get_paper_by_id(paper_id)

        if not db_paper:
            return None

        for field in ("title", "abstract", "keywords"):
            if field in data:
                setattr(db_paper, field, data[field])

        await self.session.commit()
        await self.session.refresh(db_paper)

        return db_paper

    async def delete_paper(self, paper_id: int) -> bool:
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

    async def upsert_by_external_id(self, external_id: str, data: dict) -> Paper:
        data["external_id"] = external_id
        stmt = sqlite_insert(Paper).values(**data)
        stmt = stmt.on_conflict_do_update(
            index_elements=[Paper.external_id],
            set_={
                k: stmt.excluded.get(k)
                for k in data
                if k != "external_id"
            }
        )
        await self.session.execute(stmt)
        await self.session.commit()

        # Return the upserted object
        result = await self.session.execute(
            select(Paper).where(Paper.external_id == external_id)
        )
        return result.scalar_one()