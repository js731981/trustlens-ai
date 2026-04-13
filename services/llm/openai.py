from __future__ import annotations

import os
from typing import Any

from .base import BaseLLM


class OpenAILLM(BaseLLM):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        timeout_s: float = 30.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def generate(self, prompt: str) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing OpenAI API key. Set environment variable OPENAI_API_KEY."
            )

        try:
            import httpx
        except ModuleNotFoundError as e:  # pragma: no cover
            raise RuntimeError(
                "Missing dependency: 'httpx'. Install it with `pip install httpx`."
            ) from e

        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            resp = httpx.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
        except httpx.TimeoutException as e:
            raise TimeoutError(
                f"OpenAI request timed out after {self.timeout_s}s calling {url}."
            ) from e
        except httpx.HTTPStatusError as e:
            status = getattr(e.response, "status_code", None)
            body_text = ""
            try:
                body_text = e.response.text
            except Exception:
                body_text = ""
            raise RuntimeError(
                f"OpenAI HTTP error{f' {status}' if status is not None else ''}: {body_text}".strip()
            ) from e
        except httpx.RequestError as e:
            raise RuntimeError(f"Failed to call OpenAI at {url}: {e}") from e

        try:
            data = resp.json()
        except ValueError as e:
            raise RuntimeError(
                f"OpenAI returned non-JSON response: {resp.text}"
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
                raise RuntimeError(f"OpenAI API error: {err.get('message')}") from e
            raise RuntimeError(f"Unexpected OpenAI response shape: {data}") from e

        if not isinstance(content, str):
            content = str(content)
        return content.strip()

