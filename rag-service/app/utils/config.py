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
    # Keep rag-service startup fast/offline-friendly by not requiring model download.
    # Default matches `sentence-transformers/all-MiniLM-L6-v2` (384 dims).
    embedding_vector_size: int = 384
    # If true, try loading the embedding model during FastAPI startup.
    # If false (default), model loads on first embed call.
    embedding_load_on_startup: bool = False
    environment: str = "development"


@lru_cache
def get_settings() -> Settings:
    return Settings()
