import asyncio
import logging
import os
from pathlib import Path

import faiss  # pyright: ignore[reportMissingTypeStubs]
import numpy as np

logger = logging.getLogger(__name__)


class FaissIndexStore:
    def __init__(self, index_path: str):
        self.index_path = index_path
        self.index = None
        self._write_lock: asyncio.Lock | None = None
        self.load()

    def __len__(self) -> int:
        return int(self.index.ntotal) if self.index is not None else 0

    def _new_index(self, dimension: int):
        return faiss.IndexIDMap(faiss.IndexFlatIP(dimension))

    def _ensure_index(self, dimension: int) -> None:
        if self.index is None:
            self.index = self._new_index(dimension)

    def _get_lock(self) -> asyncio.Lock:
        if self._write_lock is None:
            self._write_lock = asyncio.Lock()
        return self._write_lock

    def load(self) -> None:
        if not os.path.exists(self.index_path):
            logger.info("Initialized empty FAISS index")
            return

        self.index = faiss.read_index(self.index_path)
        logger.info("Loaded FAISS index with %s vectors", self.index.ntotal)

    def save(self) -> None:
        if self.index is None:
            Path(self.index_path).unlink(missing_ok=True)
            return
        faiss.write_index(self.index, self.index_path)

    async def persist(self) -> None:
        async with self._get_lock():
            self.save()

    def ids(self) -> set[int]:
        if (
            self.index is None
            or self.index.ntotal == 0
            or not hasattr(self.index, "id_map")
        ):
            return set()

        return {int(idx) for idx in faiss.vector_to_array(self.index.id_map)}  # type: ignore

    def contents(self) -> list[int]:
        return sorted(self.ids())

    def _validate_vectors(self, ids: list[int], vectors: np.ndarray) -> np.ndarray:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError(
                f"Expected a 2D embedding array, got shape {vectors.shape}."
            )
        if len(ids) != len(vectors):
            raise ValueError(
                f"Expected one FAISS id per vector, got {len(ids)} ids and "
                f"{len(vectors)} vectors."
            )
        self._ensure_index(vectors.shape[1])
        if self.index is None:
            raise RuntimeError("FAISS index was not initialized.")
        if vectors.shape[1] != self.index.d:
            raise ValueError(
                f"Embedding dimension {vectors.shape[1]} does not match FAISS "
                f"index dimension {self.index.d}."
            )
        return vectors

    async def add(self, external_id: int, vector: np.ndarray) -> bool:
        async with self._get_lock():
            if external_id in self.ids():
                return False

            vector = self._validate_vectors([external_id], vector)
            self.index.add_with_ids(vector, np.array([external_id], dtype=np.int64))  # type: ignore
            return True

    async def add_many(self, ids: list[int], vectors: np.ndarray) -> None:
        if not ids or len(vectors) == 0:
            return

        async with self._get_lock():
            vectors = self._validate_vectors(ids, vectors)
            ids_np = np.array(ids, dtype=np.int64)
            self.index.add_with_ids(vectors, ids_np)  # type: ignore

    async def search(
        self, vector: np.ndarray, top_k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.index is None or self.index.ntotal == 0:
            return np.array([]), np.array([])

        distances, indices = self.index.search(vector, top_k)  # type: ignore

        return distances[0], indices[0]

    async def replace(self, ids: list[int], vectors: np.ndarray) -> None:
        async with self._get_lock():
            self.index = None
            if ids and len(vectors) > 0:
                vectors = self._validate_vectors(ids, vectors)
                ids_np = np.array(ids, dtype=np.int64)
                self.index.add_with_ids(vectors, ids_np)  # type: ignore
            self.save()
