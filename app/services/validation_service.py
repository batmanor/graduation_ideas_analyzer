import math
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
        text_to_embed = f"{idea.title}. {idea.abstract}. {idea.keywords}"

        distances, indices = await self.vector_store.search(text_to_embed, top_k=5)

        distances = distances.flatten()
        indices = indices.flatten()

        if len(indices) == 0:
            return ValidationResponse(
                is_novel=True,
                message="Idea appears to be novel!",
                similar_papers=[],
            )

        valid_matches = []
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

        is_novel = not any(
            score >= settings.SIMILARITY_THRESHOLD for score, _ in valid_matches
        )

        external_ids = [idx for _, idx in valid_matches]

        papers_map = {}
        if external_ids:
            papers = await self.paper_service.get_papers_by_ids(external_ids)
            papers_map: dict[int, Paper] = (
                {p.external_id: p for p in papers} if papers else {}
            )

        # -------------------------
        # Build matches (preserve order)
        # -------------------------
        matches: list[SimilarPaperMatch] = []

        for score, external_id in valid_matches:
            paper_model = papers_map.get(external_id)
            if not paper_model:
                continue

            if score >= settings.SIMILARITY_THRESHOLD:
                matches.append(
                    SimilarPaperMatch(
                        external_id=external_id,
                        title=paper_model.title,
                        similarity_score=score,
                    )
                )

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
