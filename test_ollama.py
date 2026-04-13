"""Standalone smoke test for Ollama via OllamaLLM (no FastAPI)."""

from __future__ import annotations

import os

from dotenv import load_dotenv

from services.llm.ollama import OllamaLLM


def main() -> None:
    load_dotenv()
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    llm = OllamaLLM(base_url=base_url)
    output = llm.generate("Say hello in one sentence")
    print(output)


if __name__ == "__main__":
    main()
