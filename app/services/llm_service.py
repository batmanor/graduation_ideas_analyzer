import logging

from google import genai

from ..core.config import settings

logger = logging.getLogger(__name__)


class GeminiLLMService:
    def __init__(self):
        if not settings.GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY is not configured")
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
