from fastapi import APIRouter, HTTPException, status, Query
from app.schemas.paper import PaperCreate, PaperResponse, PaperUpdate
from app.api.dependencies import PaperServiceDep, VectorStoreDep

router = APIRouter()


@router.get("/", response_model=list[PaperResponse])
async def get_papers(
    paper_service: PaperServiceDep,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    papers = await paper_service.get_all_papers(skip=offset, limit=limit)
    return papers


@router.get("/{supabase_id}", response_model=PaperResponse)
async def get_paper(
    paper_service: PaperServiceDep,
    supabase_id: str,
):
    paper = await paper_service.get_paper_by_external_id(supabase_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return paper


@router.post("/", status_code=status.HTTP_201_CREATED, response_model=PaperResponse)
async def add_paper(
    paper: PaperCreate,
    paper_service: PaperServiceDep,
    vector_store: VectorStoreDep,
):
    db_paper = await paper_service.create_paper(paper)

    if not db_paper:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR)

    try:
        await vector_store.index_paper(db_paper)
        await vector_store.persist()
    except Exception as e:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Paper was saved, but vector indexing failed. Run FAISS full sync.",
        ) from e

    return db_paper


@router.patch("/{supabase_id}", response_model=PaperResponse)
async def edit_paper(
    supabase_id: str,
    paper: PaperUpdate,
    paper_service: PaperServiceDep,
    vector_store: VectorStoreDep,
):
    db_paper = await paper_service.get_paper_by_external_id(supabase_id)
    if not db_paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    updated_paper = await paper_service.update_paper(db_paper.id, paper)
    if not updated_paper:
        raise HTTPException(status_code=404, detail="Paper not found after update")

    await vector_store.full_rebuild(paper_service)
    return updated_paper


@router.delete("/{supabase_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_paper(
    paper_service: PaperServiceDep,
    vector_store: VectorStoreDep,
    supabase_id: str,
):
    db_paper = await paper_service.get_paper_by_external_id(supabase_id)
    if not db_paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    if not await paper_service.delete_paper(db_paper.id):
        raise HTTPException(status_code=404, detail="Paper not found")

    await vector_store.full_rebuild(paper_service)


async def sync_with_supabase(
        
):
    ...