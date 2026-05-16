import asyncio

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.models import Base
from app.repositories.paper_repo import PaperRepository


async def _exercise_repository():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with session_factory() as session:
        repo = PaperRepository(session)

        created = await repo.create_paper(
            external_id=101,
            title="First Paper",
            abstract="An abstract",
            keywords="retrieval, validation",
        )
        duplicate = await repo.create_paper(
            external_id=101,
            title="Duplicate Paper",
            abstract="Duplicate abstract",
            keywords="duplicate",
        )
        all_papers = await repo.get_all_papers()
        external_ids = await repo.get_all_external_ids()
        updated = await repo.update_paper(created.id, {"title": "Updated", "bad": "x"})
        deleted = await repo.delete_paper(created.id)
        missing_delete = await repo.delete_paper(created.id)

    await engine.dispose()

    return {
        "created": created,
        "duplicate": duplicate,
        "all_papers": all_papers,
        "external_ids": external_ids,
        "updated": updated,
        "deleted": deleted,
        "missing_delete": missing_delete,
    }


def test_repository_create_duplicate_update_and_delete():
    result = asyncio.run(_exercise_repository())

    assert result["created"].external_id == 101
    assert result["duplicate"].id == result["created"].id
    assert len(result["all_papers"]) == 1
    assert result["external_ids"] == {101}
    assert result["updated"].title == "Updated"
    assert not hasattr(result["updated"], "bad")
    assert result["deleted"] is True
    assert result["missing_delete"] is False
