from typing import Optional
from pydantic import BaseModel


class PaperCreate(BaseModel):
    external_id: int
    title: str
    abstract: str
    keywords: Optional[str] = None


class PaperResponse(BaseModel):
    id: int
    external_id: int
    title: str
    abstract: str
    keywords: Optional[str] = None

    class Config:
        from_attributes = True


class PaperUpdate(BaseModel):
    id: int
    title: Optional[str] = None
    abstract: Optional[str] = None
    keywords: Optional[str] = None

    class Config:
        from_attributes = True
