from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    SUPABASE_URL: str
    SUPABASE_KEY: str
    TABLE_NAME: str
    
    GEMINI_API_KEY: str | None = None
    GEMINI_MODEL: str = "gemini-2.5-flash"
    GEMINI_EMBEDDING_MODEL: str = "text-embedding-004"

    EMBEDDING_PROVIDER: Literal["local", "gemini"] = "local"

    REPO_ID: str = "Mo-alhariri/paraphrase-multilingual-minilm-l12-v2-int8"
    EMBEDDING_MODEL_PATH: str =  "models/paraphrase-multilingual-MiniLM-L12-v2-int8"

    EMBEDDING_ONNX_FILE: str = "model.int8.onnx"
    EMBEDDING_POOLING_CONFIG_PATH: str = "1_Pooling"

    PREWARM_EMBEDDING_MODEL: bool = False
    AUTO_REBUILD_INDEX_ON_STARTUP: bool = False
    DATABASE_URL: str = "sqlite+aiosqlite:///./papers.db"
    FAISS_INDEX_PATH: Path = Path("vector_index.faiss")

    SIMILARITY_THRESHOLD: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
    )

    LOG_LEVEL: str = "INFO"
    SQLALCHEMY_ECHO: bool = False

    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", case_sensitive=True
    )


settings = Settings()
