from pydantic import BaseModel, ConfigDict, Field


class PaperCreate(BaseModel):
    external_id: str
    title: str = Field(min_length=1, max_length=500)
    abstract: str = Field(min_length=1, max_length=10000)
    keywords: str | None = Field(default=None, max_length=1000)


class PaperResponse(BaseModel):
    id: int
    external_id: str
    title: str
    abstract: str
    keywords: str | None = None

    model_config = ConfigDict(from_attributes=True)


class PaperUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=500)
    abstract: str | None = Field(default=None, min_length=1, max_length=10000)
    keywords: str | None = Field(default=None, max_length=1000)

    model_config = ConfigDict(from_attributes=True)
