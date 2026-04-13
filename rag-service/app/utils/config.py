from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# rag-service/.env (not cwd-relative) so uvicorn from any working directory still picks up QDRANT_URL, etc.
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Default avoids requiring a local Qdrant process; set QDRANT_URL for persistent remote/local Qdrant.
    qdrant_url: str = ":memory:"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "rag_documents"
    embedding_model: str = "all-MiniLM-L6-v2"
    environment: str = "development"


@lru_cache
def get_settings() -> Settings:
    return Settings()
