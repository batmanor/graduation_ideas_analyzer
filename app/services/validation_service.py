import math

from app.utils.processing import build_text
from ..models.paper import Paper
from ..schemas import IdeaSubmit, ValidationResponse, SimilarPaperMatch
from ..services.paper_service import PaperService
from ..services.vector_store import VectorStoreService
from ..core.config import settings


class ValidationService:
    def __init__(self, paper_service: PaperService, vector_store: VectorStoreService):
        self.paper_service = paper_service
        self.vector_store = vector_store

    async def validate_idea(self, idea: IdeaSubmit) -> ValidationResponse:
        text_to_embed = build_text(title= idea.title, abstract= idea.abstract, keywords= idea.keywords)

        distances, indices = await self.vector_store.search(text_to_embed, top_k=5)

        if len(indices) == 0:
            return ValidationResponse(
                is_novel=True,
                message="Idea appears to be novel!",
                similar_papers=[],
            )

        valid_matches = self._filter_valid_matches(distances, indices)

        is_novel = not any(
            score >= settings.SIMILARITY_THRESHOLD for score, _ in valid_matches
        )

        # Build a fast lookup from int id to Paper
        matches = await self._build_matches(valid_matches)

        message = (
            "Idea appears to be novel!"
            if is_novel
            else "Idea closely resembles existing work."
        )

        return ValidationResponse(
            is_novel=is_novel,
            message=message,
            similar_papers=matches,
        )


    def _filter_valid_matches(self, distances, indices) -> list[tuple[float, int]]:
        """Clean FAISS results: cast to native types and discard -1, NaN, Inf."""
        valid_matches:list[tuple[float, int]] = []
        for dist, idx in zip(distances, indices):
            idx = int(idx)

            # FAISS returns -1 if there are not enough matches
            if idx == -1:
                continue

            score = float(dist)

            # Prevent JSON serialization errors from NumPy/FAISS NaN or Inf returns
            if math.isnan(score) or math.isinf(score):
                continue

            valid_matches.append((score, idx))
        return valid_matches

 
    async def _build_matches(self, valid_matches) -> list[SimilarPaperMatch]:
        """Fetch paper objects and build a list of SimilarPaperMatch."""
        if not valid_matches:
            return []
        
        ids = [idx for _, idx in valid_matches]
        papers = await self.paper_service.get_papers_by_ids(ids)
        id_to_paper: dict[int, Paper] = {p.id: p for p in papers} if papers else {}

        matches: list[SimilarPaperMatch] = []
        for score, pid in valid_matches:
            paper = id_to_paper.get(pid)
            if paper and score >= settings.SIMILARITY_THRESHOLD:
                matches.append(
                    SimilarPaperMatch(
                        external_id=paper.external_id,
                        title=paper.title,
                        abstract=paper.abstract,
                        similarity_score=score,
                    )
                )
                
        return matches

