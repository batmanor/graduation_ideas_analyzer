from fastapi import APIRouter
from app.api.dependencies import PaperServiceDep, VectorStoreDep

router = APIRouter(tags=["faiss_sync"])


@router.get("/sync/status", response_model=dict)
async def sync_status(
    paper_service: PaperServiceDep,
    vector_store: VectorStoreDep,
):
    external_ids = await paper_service.get_all_external_ids()
    return await vector_store.check_sync(external_ids)


@router.post("/sync/")
async def sync_faiss_index(vector_store: VectorStoreDep):
    """Manually synchronize / save the FAISS index to disk."""
    try:
        await vector_store.persist()
        return {"message": "FAISS index synchronized to disk."}
    except Exception as e:
        return {"error": str(e)}


@router.post("/sync/full", response_model=dict)
async def full_sync(
    paper_service: PaperServiceDep,
    vector_store: VectorStoreDep,
):
    await vector_store.full_sync(paper_service)
    return {"message": "Full sync completed."}


@router.post("/sync/full-rebuild", response_model=dict)
async def full_rebuild(
    paper_service: PaperServiceDep,
    vector_store: VectorStoreDep,
):
    await vector_store.full_rebuild(paper_service)
    return {"message": "Full rebuild completed."}
