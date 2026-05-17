import asyncio
import logging
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np

from ..core.config import settings
from ..models.paper import Paper
from ..services.embedding_service import EmbeddingService
from ..services.faiss_index_store import FaissIndexStore

if TYPE_CHECKING:
    from ..services.paper_service import PaperService


FAISS_INDEX_PATH = "vector_index.faiss"
logger = logging.getLogger(__name__)


class VectorStoreService:
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.index_store = FaissIndexStore(FAISS_INDEX_PATH)

    def __len__(self) -> int:
        return len(self.index_store)

    async def persist(self) -> None:
        await self.index_store.persist()

    async def add_vector(self, external_id: int, text: str) -> None:
        vector = await asyncio.to_thread(self.embedding_service.embed, text)

        try:
            await self.index_store.add(external_id, vector)
        except Exception as e:
            logger.exception(
                "Failed to add vector for external_id=%s: %s", external_id, e
            )

    async def search(self, text: str, top_k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        vector = await asyncio.to_thread(self.embedding_service.embed, text)
        return await self.index_store.search(vector, top_k)

    async def get_contents(self) -> list[int]:
        return await self.index_store.contents()

    def get_faiss_ids(self) -> set[int]:
        return self.index_store.ids()

    async def check_sync(self, external_ids: set[int]) -> dict[str, Any]:
        faiss_ids = self.get_faiss_ids()
        missing_from_faiss = external_ids.difference(faiss_ids)
        extra_in_faiss = faiss_ids.difference(external_ids)

        return {
            "is_sync": not missing_from_faiss and not extra_in_faiss,
            "missing_from_faiss": sorted(missing_from_faiss),
            "extra_in_faiss": sorted(extra_in_faiss),
        }

    async def full_sync(self, paper_service: "PaperService") -> None:
        external_ids = await paper_service.get_all_external_ids()
        faiss_ids = self.get_faiss_ids()

        missing_ids = external_ids - faiss_ids
        extra_ids = faiss_ids - external_ids

        if extra_ids:
            logger.info(
                "Rebuilding FAISS index due to %s stale entries", len(extra_ids)
            )
            ids, vectors = await self._vectors_for_all_papers(paper_service)
            await self.index_store.replace(ids, vectors)
            return

        ids, vectors = await self._vectors_for_existing_papers(
            paper_service, missing_ids
        )
        await self.index_store.add_many(ids, vectors)
        await self.index_store.persist()

    async def full_rebuild(self, paper_service: "PaperService") -> None:
        ids, vectors = await self._vectors_for_all_papers(paper_service)
        await self.index_store.replace(ids, vectors)

    async def _vectors_for_all_papers(
        self, paper_service: "PaperService"
    ) -> tuple[list[int], np.ndarray]:
        papers: Iterable[Paper] = await paper_service.get_all_papers()
        return await self._embed_papers(papers)

    async def _vectors_for_existing_papers(
        self, paper_service: "PaperService", external_ids: Iterable[int]
    ) -> tuple[list[int], np.ndarray]:
        papers = []
        for external_id in external_ids:
            paper = await paper_service.get_paper_by_external_id(external_id)
            if paper:
                papers.append(paper)

        return await self._embed_papers(papers)

    async def _embed_papers(
        self, papers: Iterable[Paper]
    ) -> tuple[list[int], np.ndarray]:
        paper_list = list(papers)
        if not paper_list:
            return [], np.empty((0, settings.EMBEDDING_DIM), dtype=np.float32)

        texts = [paper.abstract for paper in paper_list]
        ids = [paper.external_id for paper in paper_list]
        vectors = await asyncio.to_thread(self.embedding_service.embed_batch, texts)

        return ids, vectors
