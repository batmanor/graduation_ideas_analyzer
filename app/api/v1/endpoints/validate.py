from fastapi import APIRouter
from app.schemas.validation import IdeaSubmit, ValidationResponse
from app.api.dependencies import ValidationServiceDep

router = APIRouter()


@router.post("/", response_model=ValidationResponse)
async def validate_idea(idea: IdeaSubmit, validation_service: ValidationServiceDep):
    return await validation_service.validate_idea(idea)
