import logging

from google import genai

from ..core.config import settings

logger = logging.getLogger(__name__)


class GeminiLLMService:
    def __init__(self):
        if not settings.GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY is not configured")
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.model = settings.GEMINI_MODEL

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
                model=self.model, contents=prompt
            )
        except genai.types.ServerError as e:  # type: ignore
            logger.warning("Gemini model %s failed.")
            raise e
        return response.text if response.text else ''
