from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    GEMINI_API_KEY: str | None = None
    GEMINI_MODEL: str = "gemini-2.5-flash"

    REPO_ID: str = "Mo-alhariri/paraphrase-multilingual-minilm-l12-v2-int8"
    EMBEDDING_MODEL_PATH: str = "models/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_ONNX_FILE: str = "model.int8.onnx"
    EMBEDDING_POOLING_CONFIG_PATH: str = "1_Pooling"

    EMBEDDING_DIM: int = 384
    PREWARM_EMBEDDING_MODEL: bool = False
    SIMILARITY_THRESHOLD: float = 0.75

    LOG_LEVEL: str = "INFO"
    SQLALCHEMY_ECHO: bool = False

    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", case_sensitive=True
    )


settings = Settings()
