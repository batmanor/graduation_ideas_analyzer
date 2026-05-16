import asyncio
import logging
import os

import faiss  # pyright: ignore[reportMissingTypeStubs]
import numpy as np

from ..core.config import settings

logger = logging.getLogger(__name__)


class FaissIndexStore:
    def __init__(self, index_path: str):
        self.index_path = index_path
        self.index = self._new_index()
        self._lock: asyncio.Lock | None = None
        self.load()

    def __len__(self) -> int:
        return int(self.index.ntotal)

    def _new_index(self):
        return faiss.IndexIDMap(faiss.IndexFlatIP(settings.EMBEDDING_DIM))

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def load(self) -> None:
        if not os.path.exists(self.index_path):
            logger.info("Initialized empty FAISS index")
            return

        self.index = faiss.read_index(self.index_path)
        if self.index.d != settings.EMBEDDING_DIM:
            raise RuntimeError(
                "Existing FAISS index dimension does not match "
                f"EMBEDDING_DIM={settings.EMBEDDING_DIM}. Rebuild the index."
            )

        logger.info("Loaded FAISS index with %s vectors", self.index.ntotal)

    def save(self) -> None:
        faiss.write_index(self.index, self.index_path)

    async def persist(self) -> None:
        async with self._get_lock():
            self.save()

    def ids(self) -> set[int]:
        if self.index.ntotal == 0 or not hasattr(self.index, "id_map"):
            return set()

        return {int(idx) for idx in faiss.vector_to_array(self.index.id_map)}  # type: ignore

    async def contents(self) -> list[int]:
        async with self._get_lock():
            return sorted(self.ids())

    async def add(self, external_id: int, vector: np.ndarray) -> bool:
        async with self._get_lock():
            if external_id in self.ids():
                return False

            self.index.add_with_ids(vector, np.array([external_id], dtype=np.int64))  # type: ignore
            self.save()
            return True

    async def add_many(self, ids: list[int], vectors: np.ndarray) -> None:
        if not ids or len(vectors) == 0:
            return

        async with self._get_lock():
            ids_np = np.array(ids, dtype=np.int64)
            self.index.add_with_ids(vectors, ids_np)  # type: ignore

    async def search(
        self, vector: np.ndarray, top_k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.index.ntotal == 0:
            return np.array([]), np.array([])

        async with self._get_lock():
            distances, indices = self.index.search(vector, top_k)  # type: ignore

        return distances[0], indices[0]

    async def replace(self, ids: list[int], vectors: np.ndarray) -> None:
        async with self._get_lock():
            self.index = self._new_index()
            if ids and len(vectors) > 0:
                ids_np = np.array(ids, dtype=np.int64)
                self.index.add_with_ids(vectors, ids_np)  # type: ignore
            self.save()
