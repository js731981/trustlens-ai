from __future__ import annotations

import json
from typing import Any

from .base import BaseLLM


class OllamaLLM(BaseLLM):
    def __init__(
        self,
        model: str = "phi3",
        base_url: str = "http://localhost:11434",
        timeout_s: float = 300.0,
        fallback_model: str | None = None,
        stream: bool = False,
        max_tokens: int = 300,
    ) -> None:
        self.model = model
        self.fallback_model = fallback_model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.stream = bool(stream)
        self.max_tokens = int(max_tokens)

    def _model_sequence(self) -> list[str]:
        seq = [self.model]
        if self.fallback_model and self.fallback_model != self.model:
            seq.append(self.fallback_model)
        return seq

    def generate(self, prompt: str) -> str:
        try:
            import requests
        except ModuleNotFoundError as e:  # pragma: no cover
            raise RuntimeError(
                "Missing dependency: 'requests'. Install it with `pip install requests`."
            ) from e

        models = self._model_sequence()
        last_error: BaseException | None = None
        for idx, model_name in enumerate(models):
            try:
                return self._generate_one(requests, model_name, prompt)
            except BaseException as e:
                last_error = e
                if idx + 1 < len(models):
                    print(
                        f"[Ollama] model {model_name!r} failed ({e!s}); "
                        f"retrying with {models[idx + 1]!r}."
                    )
        assert last_error is not None
        raise last_error

    def _generate_one(
        self,
        requests: Any,
        model_name: str,
        prompt: str,
    ) -> str:
        url = f"{self.base_url}/api/chat"
        payload: dict[str, Any] = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": self.stream,
            "options": {"num_predict": self.max_tokens},
        }

        try:
            resp = requests.post(
                url, json=payload, timeout=self.timeout_s, stream=self.stream
            )
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(
                f"Could not connect to Ollama at {url}. "
                "Ensure Ollama is running and reachable (e.g. ollama serve)."
            ) from e
        except requests.exceptions.Timeout as e:
            raise TimeoutError(
                f"Ollama request timed out after {self.timeout_s}s calling {url}."
            ) from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama request failed for {url}: {e}") from e

        print(f"[Ollama] model={model_name!r} HTTP status: {resp.status_code}")

        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            err_body = resp.text[:500] if resp.text else ""
            raise RuntimeError(
                f"Ollama HTTP {resp.status_code} from {url}: {err_body}"
            ) from e

        if not self.stream:
            try:
                data = resp.json()
            except ValueError as e:
                body_preview = (resp.text or "")[:500]
                raise RuntimeError(
                    f"Ollama returned non-JSON response (first 500 chars): {body_preview}"
                ) from e
            message = data.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content.strip()
            raise RuntimeError("Unexpected Ollama response shape from /api/chat.")

        parts: list[str] = []
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            try:
                data = json.loads(raw_line)
            except json.JSONDecodeError as e:
                preview = raw_line[:300] if raw_line else ""
                raise RuntimeError(
                    f"Ollama stream line invalid JSON (first 300 chars): {preview}"
                ) from e

            message = data.get("message")
            if isinstance(message, dict) and "content" in message:
                chunk = message["content"]
                if chunk:
                    parts.append(chunk if isinstance(chunk, str) else str(chunk))

        return "".join(parts).strip()
