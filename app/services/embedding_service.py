import logging
import time
from typing import Any

from faiss import normalize_L2  # pyright: ignore[reportMissingTypeStubs]

import numpy as np
from light_embed import TextEmbedding  # pyright: ignore[reportMissingTypeStubs]

from ..core.config import settings
from ..utils.download_model import ensure_model

logger = logging.getLogger(__name__)


class EmbeddingService:
    _model: TextEmbedding | None

    def __init__(self):
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

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        normalize_L2(embeddings)
        return embeddings

    def embed(self, text: str) -> np.ndarray:
        model = self.get_model()

        embedding = model.encode([text])[0]
        embedding_np = np.array(embedding, dtype=np.float32)

        return self._normalize(embedding_np.reshape(1, -1))

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, settings.EMBEDDING_DIM), dtype=np.float32)

        model = self.get_model()

        embeddings = model.encode(texts)
        embeddings_np = np.array(embeddings, dtype=np.float32)

        return self._normalize(embeddings_np)
