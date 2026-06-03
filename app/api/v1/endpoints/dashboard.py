from fastapi import APIRouter, Query
from app.schemas.validation import DashboardResponse
from app.schemas.paper import PaperResponse
from app.api.dependencies import PaperServiceDep, VectorStoreDep

router = APIRouter()


@router.get("/", response_model=DashboardResponse)
async def get_dashboard_details(
    paper_service: PaperServiceDep,
    vector_store: VectorStoreDep,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    total = await paper_service.count_papers()
    papers = await paper_service.get_all_papers(skip=offset, limit=limit)
    return DashboardResponse(
        total_papers=total,
        index_length=len(vector_store),
        papers=papers,
    )


@router.get("/papers", response_model=list[PaperResponse])
async def get_dashboard_papers(
    paper_service: PaperServiceDep,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    papers = await paper_service.get_all_papers(skip=offset, limit=limit)
    return papers


@router.get("/index-contents", response_model=list[int])
async def get_index_contents(
    vector_store: VectorStoreDep,
):
    return vector_store.get_contents()
