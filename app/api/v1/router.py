from fastapi import APIRouter

from .endpoints import papers, validate, dashboard, faiss_sync

api_router = APIRouter(tags=["system"])
api_router.include_router(papers.router, prefix="/papers", tags=["papers"])
api_router.include_router(validate.router, prefix="/validate", tags=["validate"])
api_router.include_router(dashboard.router, prefix="/dashboard", tags=["dashboard"])
api_router.include_router(faiss_sync.router, prefix="/faiss_sync", tags=["faiss_sync"])
