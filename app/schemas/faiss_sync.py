from pydantic import BaseModel

class SyncStatusResponse(BaseModel):
    is_sync: bool
    missing_from_faiss: list[str]
    extra_in_faiss: list[str]

class MessageResponse(BaseModel):
    message: str