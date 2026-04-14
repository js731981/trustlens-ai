from __future__ import annotations

import os

from .base import BaseLLM
from .ollama import OllamaLLM
from .openai import OpenAILLM
from .openrouter import OpenRouterLLM


def get_llm(provider: str) -> BaseLLM:
    normalized = (provider or "ollama").strip().lower()

    if normalized == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
        model = os.getenv("OLLAMA_MODEL", "phi3")
        fallback_model = os.getenv("OLLAMA_FALLBACK_MODEL") or None
        stream = (os.getenv("OLLAMA_STREAM") or "").strip().lower() in ("1", "true", "yes", "y", "on")
        return OllamaLLM(
            base_url=base_url,
            model=model,
            fallback_model=fallback_model,
            stream=stream,
        )
    if normalized == "openai":
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1"
        return OpenAILLM(base_url=base_url)
    if normalized == "openrouter":
        base_url = os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
        return OpenRouterLLM(base_url=base_url)

    raise ValueError(
        f"Invalid LLM provider '{provider}'. Supported providers: ollama, openai, openrouter."
    )

