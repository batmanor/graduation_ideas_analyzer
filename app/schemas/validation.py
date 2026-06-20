from pydantic import BaseModel, Field


class IdeaSubmit(BaseModel):
    title: str = Field(min_length=1, max_length=500)
    abstract: str = Field(min_length=1, max_length=10000)
    keywords: str = Field(min_length=1, max_length=1000)


class SimilarPaperMatch(BaseModel):
    external_id: str
    title: str
    abstract: str   
    similarity_score: float


class ValidationResponse(BaseModel):
    is_novel: bool
    message: str
    similar_papers: list[SimilarPaperMatch] = Field(default_factory=list)


class DashboardResponse(BaseModel):
    total_papers: int
    index_length: int
