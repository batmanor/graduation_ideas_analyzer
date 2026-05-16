import logging

from google import genai

from ..core.config import settings
from ..models import Paper

logger = logging.getLogger(__name__)


class GeminiLLMService:
    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.models = (
            "gemini-3-flash-preview",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
        )

    async def generate_keywords_async(
        self, title: str, abstract: str, tries: int = 0
    ) -> str:
        """will query the LLM for keywords."""
        prompt = f"""Generate 5-10 keywords for the following paper:
            Title: {title}
            Abstract: {abstract},
            return only the keywords separated by commas."""
        try:
            response = await self.client.aio.models.generate_content(
                model=self.models[tries], contents=prompt
            )
        except genai.types.ServerError as e:  # type: ignore
            if tries < len(self.models) - 1:
                logger.warning(
                    "Gemini model %s failed; retrying with %s",
                    self.models[tries],
                    self.models[tries + 1],
                )
                return await self.generate_keywords_async(title, abstract, tries + 1)
            raise e
        return response.text

    # Not used for now, could be used in the future to save more space for the hosting
    async def analyze_novelty(
        self, idea_title: str, idea_abstract: str, similar_papers: list[Paper]
    ) -> str:
        """will query the LLM for novelty analysis."""
        prompt = f"""Analyze the novelty of the following idea:
            Title: {idea_title}
            Abstract: {idea_abstract},
            Similar papers: {similar_papers},
            return a novelty analysis."""
        response = await self.client.aio.models.generate_content(
            model="gemini-3-flash-preview", contents=prompt
        )
        logger.debug(
            "Generated novelty analysis for %s similar papers", len(similar_papers)
        )
        return response.text
