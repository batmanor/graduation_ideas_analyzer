from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.api.dependencies import PaperServiceDep, VectorStoreDep, get_supabase_client
from app.schemas.faiss_sync import SyncStatusResponse, MessageResponse
from app.services.sync_service import SyncService

router = APIRouter(tags=["faiss_sync"])

@router.get("/sync/status", response_model=SyncStatusResponse)
async def sync_status(paper_service: PaperServiceDep, vector_store: VectorStoreDep):
    """Return the current sync status between DB and FAISS index."""
    try:
        ids = await paper_service.get_all_ids()
        return await vector_store.check_sync(ids)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/sync", response_model=MessageResponse)
async def sync_to_disk(vector_store: VectorStoreDep):
    """Persist the in-memory FAISS index to disk."""
    try:
        await vector_store.persist()
        return {"message": "FAISS index synchronized to disk."}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/sync/full", response_model=MessageResponse)
async def full_sync(
    paper_service: PaperServiceDep,
    vector_store: VectorStoreDep,
):
    """Perform a full incremental sync (existing index stays)."""
    try:
        await vector_store.full_sync(paper_service)
        return {"message": "Full sync completed."}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/sync/full-rebuild", response_model=MessageResponse)
async def full_rebuild(
    paper_service: PaperServiceDep,
    vector_store: VectorStoreDep,
):
    """Wipe and completely rebuild the FAISS index from the database."""
    try:
        await vector_store.full_rebuild(paper_service)
        return {"message": "Full rebuild completed."}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
    

@router.post("/sync/from-supabase")
async def sync_from_supabase(
    paper_service: PaperServiceDep,
    vector_store: VectorStoreDep,
    supabase = Depends(get_supabase_client),
):
    try:
        sync_service = SyncService(paper_service, vector_store, supabase)
        await sync_service.sync_from_supabase()
        return {"message": "Local cache synced from Supabase."}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))