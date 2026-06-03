from fastapi import APIRouter, status
from app.schemas.paper import PaperCreate, PaperResponse
from app.api.dependencies import PaperServiceDep, VectorStoreDep

router = APIRouter()


@router.post("/", status_code=status.HTTP_201_CREATED, response_model=PaperResponse)
async def add_paper(
    paper: PaperCreate,
    paper_service: PaperServiceDep,
    vector_store: VectorStoreDep,
):
    db_paper = await paper_service.create_paper(
        paper.external_id, paper.title, paper.abstract, paper.keywords
    )
    await vector_store.index_paper(db_paper)
    return db_paper
