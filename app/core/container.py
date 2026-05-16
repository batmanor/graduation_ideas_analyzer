"""
Application-level singleton container.

A single instance of this dataclass is stored in ``app.state.container`` during
FastAPI lifespan startup. Heavy resources such as the ONNX model and FAISS
index are created lazily on first use so hosted environments can mark the app
ready without waiting for ML initialization.
"""

from dataclasses import dataclass, field
import threading

from app.services.embedding_service import EmbeddingService
from app.services.llm_service import GeminiLLMService
from app.services.vector_store import VectorStoreService


@dataclass
class AppContainer:
    llm_service: GeminiLLMService
    embedding_service: EmbeddingService | None = None
    vector_store: VectorStoreService | None = None
    _lock: threading.RLock = field(
        default_factory=threading.RLock, init=False, repr=False
    )

    def get_embedding_service(self) -> EmbeddingService:
        with self._lock:
            if self.embedding_service is None:
                self.embedding_service = EmbeddingService()
            return self.embedding_service

    def get_vector_store(self) -> VectorStoreService:
        with self._lock:
            if self.vector_store is None:
                self.vector_store = VectorStoreService(self.get_embedding_service())
            return self.vector_store

    async def persist_vector_store_if_loaded(self) -> None:
        if self.vector_store is not None:
            await self.vector_store.persist()
