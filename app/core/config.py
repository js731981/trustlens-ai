from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="Trust Lens API", validation_alias="APP_NAME")
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        validation_alias="ENVIRONMENT",
    )
    # When set to DEV, the API skips real LLM calls and returns a fixed mock (see financial_llm).
    app_env: str | None = Field(default=None, validation_alias="ENV")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    host: str = Field(default="0.0.0.0", validation_alias="HOST")
    port: int = Field(default=8000, validation_alias="PORT")

    llm_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LLM_API_KEY", "OPENAI_API_KEY"),
    )
    llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        validation_alias="LLM_BASE_URL",
    )
    llm_model: str = Field(default="gpt-4o-mini", validation_alias="LLM_MODEL")
    llm_timeout_seconds: float = Field(default=60.0, validation_alias="LLM_TIMEOUT_SECONDS")
    llm_max_tokens: int = Field(default=300, ge=1, le=4096, validation_alias="LLM_MAX_TOKENS")
    llm_max_retries: int = Field(default=4, ge=1, validation_alias="LLM_MAX_RETRIES")
    llm_temperature: float = Field(default=0.2, validation_alias="LLM_TEMPERATURE")
    llm_use_explanation: bool = Field(default=False, validation_alias="LLM_USE_EXPLANATION")

    # Local Ollama defaults (used when provider == "ollama" or as a fallback planner).
    ollama_model: str = Field(default="phi3", validation_alias="OLLAMA_MODEL")
    ollama_base_url: str = Field(default="http://localhost:11434", validation_alias="OLLAMA_BASE_URL")

    data_dir: Path = Field(
        default=Path("data"),
        validation_alias="TRUST_LENS_DATA_DIR",
        description="Directory for SQLite tracking DB and persisted analytics.",
    )

    rag_service_base_url: str = Field(
        default="http://localhost:8002",
        validation_alias="RAG_SERVICE_BASE_URL",
        description="RAG microservice base URL (POST {base}/search for retrieval-augmented ranking).",
    )
    rag_search_top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        validation_alias="RAG_SEARCH_TOP_K",
    )
    rag_service_timeout_seconds: float = Field(
        default=10.0,
        ge=0.5,
        validation_alias="RAG_SERVICE_TIMEOUT_SECONDS",
    )

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def use_llm_dev_mock(self) -> bool:
        return (self.app_env or "").strip().upper() == "DEV"


@lru_cache
def get_settings() -> Settings:
    return Settings()
