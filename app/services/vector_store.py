import logging
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np

from app.utils.processing import build_text

from ..core.config import settings
from ..models.paper import Paper
from ..services.embedding_backend import EmbeddingBackend
from .faiss_index import FaissIndex

if TYPE_CHECKING:
    from ..services.paper_service import PaperService


logger = logging.getLogger(__name__)


class VectorStoreService:
    def __init__(self, embedding_backend: EmbeddingBackend):
        self.embedding_backend = embedding_backend
        self.index_store = FaissIndex(str(settings.FAISS_INDEX_PATH))

    def __len__(self) -> int:
        return len(self.index_store)

    async def persist(self) -> None:
        await self.index_store.persist()

    async def index_paper(self, paper: Paper) -> None:
        text = build_text(paper.title, paper.abstract, paper.keywords)
        vector = await self.embedding_backend.embed(text)
        await self.index_store.add(paper.id, vector)

    async def search(self, text: str, top_k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        vector = await self.embedding_backend.embed(text)

        distances, indices = await self.index_store.search(vector, top_k)

        return (distances.flatten(), indices.flatten())

    def get_contents(self) -> list[int]:
        return self.index_store.contents()

    def get_faiss_ids(self) -> set[int]:
        return self.index_store.ids()

    async def check_sync(self, ids: set[int]) -> dict[str, Any]:
        faiss_ids = self.get_faiss_ids()
        missing_from_faiss = ids.difference(faiss_ids)
        extra_in_faiss = faiss_ids.difference(ids)

        return {
            "is_sync": not missing_from_faiss and not extra_in_faiss,
            "missing_from_faiss": sorted(missing_from_faiss),
            "extra_in_faiss": sorted(extra_in_faiss),
        }

    async def update_paper(self, paper: Paper) -> None:
        text = build_text(paper.title, paper.abstract, paper.keywords)
        vector = await self.embedding_backend.embed(text)
        await self.index_store.update(paper.id, vector)

    async def full_sync(self, paper_service: "PaperService") -> None:
        ids = await paper_service.get_all_ids()
        faiss_ids = self.get_faiss_ids()
        missing_from_faiss = ids.difference(faiss_ids)
        extra_in_faiss = faiss_ids.difference(ids)

        if extra_in_faiss:
            logger.info(
                "Rebuilding FAISS index due to %s stale entries", len(extra_in_faiss)
            )
            ids, vectors = await self._vectors_for_all_papers(paper_service)
            await self.index_store.replace(ids, vectors)
            return

        ids, vectors = await self._vectors_for_existing_papers(
            paper_service, missing_from_faiss
        )
        await self.index_store.add_many(ids, vectors)
        await self.index_store.persist()

    async def full_rebuild(self, paper_service: "PaperService") -> None:
        ids, vectors = await self._vectors_for_all_papers(paper_service)
        await self.index_store.replace(ids, vectors)

    async def _vectors_for_all_papers(
        self, paper_service: "PaperService"
    ) -> tuple[list[int], np.ndarray]:
        papers: Iterable[Paper] = await paper_service.get_all_papers(limit=None)
        return await self._embed_papers(papers)

    async def _vectors_for_existing_papers(self, paper_service, ids):
        papers = await paper_service.get_papers_by_ids(ids)
        
        paper_map = {paper.id: paper for paper in papers}
        ordered_papers = [paper_map[id] for id in ids if id in paper_map]
        if len(ordered_papers) != len(ids):
            logger.warning("Some IDs requested were not found in the database, skipping.")
        return await self._embed_papers(ordered_papers)

    async def _embed_papers(
        self, papers: Iterable[Paper]
    ) -> tuple[list[int], np.ndarray]:
        paper_list = list(papers)
        if not paper_list:
            return [], np.empty((0, 0), dtype=np.float32)

        texts = [
            build_text(paper.title, paper.abstract, paper.keywords)
            for paper in paper_list
        ]

        ids = [paper.id for paper in paper_list]
        vectors = await self.embedding_backend.embed_batch(texts)

        return ids, vectors
