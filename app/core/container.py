"""
Application-level singleton container.

A single instance of this dataclass is stored in ``app.state.container``
during FastAPI lifespan startup.  Heavy resources (ONNX model, FAISS index,
Gemini HTTP client) are created exactly once here and then shared across all
requests through the dependency-injection layer in ``app.api.dependencies``.
"""
from dataclasses import dataclass

from app.services.embedding_service import EmbeddingService
from app.services.llm_service import GeminiLLMService
from app.services.vector_store import VectorStoreService


@dataclass
class AppContainer:
    embedding_service: EmbeddingService
    vector_store: VectorStoreService
    llm_service: GeminiLLMService
