from __future__ import annotations

import os
from typing import Any

from .base import BaseLLM


class OpenRouterLLM(BaseLLM):
    def __init__(
        self,
        model: str = "mistralai/mixtral-8x7b",
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_s: float = 30.0,
        max_tokens: int = 300,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_tokens = int(max_tokens)

    def generate(self, prompt: str) -> str:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing OpenRouter API key. Set environment variable OPENROUTER_API_KEY."
            )

        try:
            import requests
        except ModuleNotFoundError as e:  # pragma: no cover
            raise RuntimeError(
                "Missing dependency: 'requests'. Install it with `pip install requests`."
            ) from e

        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            resp = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
        except requests.exceptions.Timeout as e:
            raise TimeoutError(
                f"OpenRouter request timed out after {self.timeout_s}s calling {url}."
            ) from e
        except requests.exceptions.HTTPError as e:
            status = getattr(resp, "status_code", None)
            body = ""
            try:
                body = resp.text
            except Exception:
                body = ""
            raise RuntimeError(
                f"OpenRouter HTTP error{f' {status}' if status is not None else ''}: {body}".strip()
            ) from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to call OpenRouter at {url}: {e}") from e

        try:
            data = resp.json()
        except ValueError as e:
            raise RuntimeError(
                f"OpenRouter returned non-JSON response: {resp.text}"
            ) from e

        try:
            choices = data["choices"]
            if not isinstance(choices, list) or not choices:
                raise KeyError("choices")
            message = choices[0]["message"]
            content = message.get("content", "")
        except Exception as e:
            err = data.get("error")
            if isinstance(err, dict) and err.get("message"):
                raise RuntimeError(f"OpenRouter API error: {err.get('message')}") from e
            raise RuntimeError(f"Unexpected OpenRouter response shape: {data}") from e

        if not isinstance(content, str):
            content = str(content)
        return content.strip()

