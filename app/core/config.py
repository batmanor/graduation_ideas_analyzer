from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    GEMINI_API_KEY: str | None = None
    GEMINI_MODEL: str = "gemini-2.5-flash"

    REPO_ID: str = Field(...)
    EMBEDDING_MODEL_PATH: str = Field(...)
    EMBEDDING_ONNX_FILE: str = Field(...)
    EMBEDDING_POOLING_CONFIG_PATH: str = Field(...)

    EMBEDDING_DIM: int = Field(...)
    SIMILARITY_THRESHOLD: float = 0.75

    LOG_LEVEL: str = "INFO"
    SQLALCHEMY_ECHO: bool = False

    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", case_sensitive=True
    )


settings = Settings()
