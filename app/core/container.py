"""
Application-level singleton container.

A single instance of this dataclass is stored in ``app.state.container`` during
FastAPI lifespan startup. Heavy resources such as the ONNX model and FAISS
index are created lazily on first use so hosted environments can mark the app
ready without waiting for ML initialization.
"""

from dataclasses import dataclass, field
import threading

from app.core.config import settings
from app.services.embedding_backend import (
    EmbeddingBackend,
    GeminiEmbeddingBackend,
    LocalEmbeddingBackend,
)
from app.services.llm_service import GeminiLLMService
from app.services.vector_store import VectorStoreService


@dataclass
class AppContainer:
    llm_service: GeminiLLMService
    embedding_backend: EmbeddingBackend | None = None
    vector_store: VectorStoreService | None = None
    _lock: threading.RLock = field(
        default_factory=threading.RLock, init=False, repr=False
    )

    def get_embedding_backend(self) -> EmbeddingBackend:
        with self._lock:
            if self.embedding_backend is not None:
                return self.embedding_backend

            if settings.EMBEDDING_PROVIDER == "local":
                self.embedding_backend = LocalEmbeddingBackend()
            elif settings.EMBEDDING_PROVIDER == "gemini":
                self.embedding_backend = GeminiEmbeddingBackend(self.llm_service)
            else:
                raise ValueError(
                    "EMBEDDING_PROVIDER must be either 'local' or 'gemini'."
                )
            return self.embedding_backend

    def get_vector_store(self) -> VectorStoreService:
        with self._lock:
            if self.vector_store is None:
                self.vector_store = VectorStoreService(self.get_embedding_backend())
            return self.vector_store

    async def persist_vector_store_if_loaded(self) -> None:
        if self.vector_store is not None:
            await self.vector_store.persist()
