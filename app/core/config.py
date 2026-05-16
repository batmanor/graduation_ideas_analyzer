from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    SIMILARITY_THRESHOLD: float = 0.75
    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-2.5-flash"
    EMBEDDING_MODEL_PATH: str = "./models/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_ONNX_FILE: str = "model.onnx"
    EMBEDDING_POOLING_CONFIG_PATH: str = "1_Pooling"
    EMBEDDING_DIM: int = 384
    LOG_LEVEL: str = "INFO"
    SQLALCHEMY_ECHO: bool = False

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
