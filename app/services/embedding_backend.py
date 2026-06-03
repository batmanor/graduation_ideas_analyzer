import logging
import asyncio
import time
from typing import Any, Protocol, runtime_checkable

import numpy as np
from light_embed import TextEmbedding  # pyright: ignore[reportMissingTypeStubs]

from ..core.config import settings
from .llm_service import GeminiLLMService
from ..utils.download_model import ensure_model

logger = logging.getLogger(__name__)


@runtime_checkable
class EmbeddingBackend(Protocol):
    async def embed(self, text: str) -> np.ndarray: ...

    async def embed_batch(self, texts: list[str]) -> np.ndarray: ...


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


class LocalEmbeddingBackend:
    def __init__(self) -> None:
        self._model = None

    def load_model(self) -> TextEmbedding:
        if self._model is not None:
            return self._model

        ensure_model()

        model_config: dict[str, Any] = {
            "onnx_file": settings.EMBEDDING_ONNX_FILE,
            "pooling_config_path": settings.EMBEDDING_POOLING_CONFIG_PATH,
            "normalize": False,
        }

        start = time.perf_counter()
        logger.info("Loading embedding model from %s", settings.EMBEDDING_MODEL_PATH)
        self._model = TextEmbedding(
            model_name_or_path=settings.EMBEDDING_MODEL_PATH,
            model_config=model_config,
        )
        logger.info("Embedding model loaded in %.2fs", time.perf_counter() - start)
        return self._model

    def get_model(self) -> TextEmbedding:
        return self.load_model()

    def _embed_sync(self, text: str) -> np.ndarray:
        model = self.get_model()

        embedding = model.encode([text])[0]
        embedding_np = np.array(embedding, dtype=np.float32)

        return normalize_embeddings(embedding_np.reshape(1, -1))

    def _embed_batch_sync(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        model = self.get_model()

        embeddings = model.encode(texts)
        embeddings_np = np.array(embeddings, dtype=np.float32)

        return normalize_embeddings(embeddings_np)

    async def embed(self, text: str) -> np.ndarray:
        return await asyncio.to_thread(self._embed_sync, text)

    async def embed_batch(self, texts: list[str]) -> np.ndarray:
        return await asyncio.to_thread(self._embed_batch_sync, texts)


class GeminiEmbeddingBackend:
    def __init__(self, llm_service: GeminiLLMService):
        self.llm_service = llm_service

    async def embed(self, text: str) -> np.ndarray:
        return await self.embed_batch([text])

    async def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        response = await self.llm_service.client.aio.models.embed_content(
            model=settings.GEMINI_EMBEDDING_MODEL,
            contents=texts,
        )
        values = [embedding.values for embedding in response.embeddings]
        return normalize_embeddings(np.array(values, dtype=np.float32))


EmbeddingService = GeminiEmbeddingBackend if settings.EMBEDDING_PROVIDER == 'gemini' else LocalEmbeddingBackend
