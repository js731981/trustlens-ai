from __future__ import annotations

import os

from app.core.config import get_settings

from .base import BaseLLM
from .ollama import OllamaLLM
from .openai import OpenAILLM
from .openrouter import OpenRouterLLM


def get_llm(provider: str) -> BaseLLM:
    normalized = (provider or "ollama").strip().lower()
    settings = get_settings()

    if normalized == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL") or settings.ollama_base_url or "http://localhost:11434"
        model = os.getenv("OLLAMA_MODEL") or settings.ollama_model or "phi3"
        fallback_model = os.getenv("OLLAMA_FALLBACK_MODEL") or None
        stream = (os.getenv("OLLAMA_STREAM") or "").strip().lower() in ("1", "true", "yes", "y", "on")
        return OllamaLLM(
            base_url=base_url,
            model=model,
            fallback_model=fallback_model,
            stream=stream,
            timeout_s=float(settings.llm_timeout_seconds),
            max_tokens=int(settings.llm_max_tokens),
        )
    if normalized == "openai":
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1"
        model = os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL") or settings.llm_model
        return OpenAILLM(
            base_url=base_url,
            model=model,
            timeout_s=float(settings.llm_timeout_seconds),
            max_tokens=int(settings.llm_max_tokens),
        )
    if normalized == "openrouter":
        base_url = os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
        model = os.getenv("OPENROUTER_MODEL") or os.getenv("LLM_MODEL") or settings.llm_model
        return OpenRouterLLM(
            base_url=base_url,
            model=model,
            timeout_s=float(settings.llm_timeout_seconds),
            max_tokens=int(settings.llm_max_tokens),
        )

    raise ValueError(
        f"Invalid LLM provider '{provider}'. Supported providers: ollama, openai, openrouter."
    )

