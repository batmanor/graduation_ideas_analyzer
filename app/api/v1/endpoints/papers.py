from fastapi import APIRouter, BackgroundTasks, status
from app.schemas.paper import PaperCreate, PaperResponse
from app.api.dependencies import PaperServiceDep, VectorStoreDep

router = APIRouter()


async def process_paper_background(vector_store, external_id: int, text: str):
    await vector_store.add_vector(external_id, text)


@router.post("/", status_code=status.HTTP_202_ACCEPTED, response_model=PaperResponse)
async def add_paper(
    paper: PaperCreate,
    background_tasks: BackgroundTasks,
    paper_service: PaperServiceDep,
    vector_store: VectorStoreDep,
):
    # add paper to database
    db_paper = await paper_service.create_paper(
        paper.external_id, paper.title, paper.abstract, paper.keywords
    )

    # add paper to vector store
    text_to_embed = f"{paper.title}. {paper.abstract}"
    background_tasks.add_task(
        process_paper_background, vector_store, paper.external_id, text_to_embed
    )

    return db_paper
