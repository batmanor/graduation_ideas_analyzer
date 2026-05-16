from typing import Optional, List
from pydantic import BaseModel
from .paper import PaperResponse


class IdeaSubmit(BaseModel):
    title: str
    abstract: str
    keywords: str


class SimilarPaperMatch(BaseModel):
    external_id: int
    title: str
    similarity_score: float


class ValidationResponse(BaseModel):
    is_novel: bool
    message: str
    similar_papers: List[SimilarPaperMatch] = []


class DashboardResponse(BaseModel):
    total_papers: int
    index_length: int
    papers: List[PaperResponse]
