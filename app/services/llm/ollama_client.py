from __future__ import annotations

# Backwards-compatible import path shim.
# The implementation lives in `ollama.py`.
from .ollama import OllamaLLM

__all__ = ["OllamaLLM"]

