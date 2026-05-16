import logging
import os
import time
from typing import Any
import threading

from faiss import normalize_L2  # pyright: ignore[reportMissingTypeStubs]
from fastapi import HTTPException
import numpy as np
from light_embed import TextEmbedding  # pyright: ignore[reportMissingTypeStubs]

from ..core.config import settings
from ..utils.download_model import MODEL_PATH

logger = logging.getLogger(__name__)


class EmbeddingService:
    _model: TextEmbedding

    def __init__(self):
        self._lock = threading.Lock()

    def load_model(self):
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                503,
                "Local embedding model folder not found. "
                f"Expected: {settings.EMBEDDING_MODEL_PATH}. "
                "Download the ONNX sentence-transformer model there or set "
                "EMBEDDING_MODEL_PATH to its local folder.",
            )

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

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        normalize_L2(embeddings)
        return embeddings

    def embed(self, text: str) -> np.ndarray:
        with self._lock:
            embedding = next(iter(self._model.encode([text])))
        embedding_np = np.array(list(embedding), dtype=np.float32)

        if embedding_np.ndim == 1:
            embedding_np = embedding_np.reshape(1, -1)

        return self._normalize(embedding_np)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, settings.EMBEDDING_DIM), dtype=np.float32)

        with self._lock:
            embeddings = list(self._model.encode(texts))
        embeddings_np = np.array(embeddings, dtype=np.float32)
        if embeddings_np.ndim == 1:
            embeddings_np = embeddings_np.reshape(1, -1)

        return self._normalize(embeddings_np)
