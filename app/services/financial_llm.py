from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from typing import Any, Final
from uuid import uuid4

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.financial import FinancialQueryResponse
from app.prompts.registry import render_prompt
from app.services import tracking_store
from app.services.llm.factory import get_llm
from app.services.llm.prompt_builder import (
    documents_from_product_names,
    format_top_results_block,
    insurance_catalog_product_names,
)
from app.services.utils.parser import parse_llm_json

logger = get_logger(__name__)

_DEV_MOCK_PARSED: Final[dict[str, Any]] = {
    "ranked_products": [
        {"name": "Mock Insurance", "rank": 1, "reason": "test"},
    ],
    "explanation": "DEV mock response (LLM skipped).",
}


class FinancialLLMConfigurationError(RuntimeError):
    pass


class FinancialLLMUpstreamError(RuntimeError):
    pass


class FinancialLLMResponseParseError(RuntimeError):
    pass


def _retryable(exc: BaseException) -> bool:
    if isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.ConnectTimeout,
            httpx.RemoteProtocolError,
        ),
    ):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code in (400, 401, 403, 404, 422):
            return False
        return code in (408, 425, 429) or 500 <= code <= 599
    return False


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    while lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = _strip_markdown_fence(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise FinancialLLMResponseParseError("Model output was not valid JSON.") from exc
    raise FinancialLLMResponseParseError("Model output was not valid JSON.")


def _parse_financial_response(raw_text: str) -> FinancialQueryResponse:
    payload = _extract_json_object(raw_text)
    try:
        result = FinancialQueryResponse.model_validate(payload)
    except Exception as exc:
        raise FinancialLLMResponseParseError("JSON did not match the expected schema.") from exc
    result.ranked_products.sort(key=lambda p: p.rank)
    return result


def build_financial_prompt(
    user_query: str,
    template_id: str = "financial_ranking",
    *,
    catalog_product_names: tuple[str, ...] | None = None,
    retrieved_documents: Iterable[dict[str, Any]] | None = None,
) -> str:
    variables: dict[str, str] = {"user_query": user_query}
    if template_id == "financial_ranking":
        if retrieved_documents is not None:
            docs = [d for d in retrieved_documents if isinstance(d, dict)]
            variables["top_results"] = format_top_results_block(docs)
        elif catalog_product_names is not None:
            variables["top_results"] = format_top_results_block(
                documents_from_product_names(catalog_product_names),
            )
        else:
            variables["top_results"] = format_top_results_block(
                documents_from_product_names(insurance_catalog_product_names()),
            )
    system_prompt, user_prompt = render_prompt(template_id, variables)
    return f"{system_prompt}\n\n{user_prompt}".strip()


LLM_DEFLECTION_MARKER = "Please provide your question"
_STRICT_JSON_ONLY_PREFIX = "STRICT JSON ONLY. DO NOT DEVIATE.\n\n"


def _financial_llm_output_ignores_instructions(raw: str) -> bool:
    return LLM_DEFLECTION_MARKER in raw or "{" not in raw


def parse_financial_response(raw_text: str) -> FinancialQueryResponse:
    return _parse_financial_response(raw_text)


async def query_financial_llm_multi(
    user_query: str,
    *,
    provider: str = "ollama",
    template_id: str = "financial_ranking",
    session_id: str | None = None,
    run_index: int = 0,
    prompt_override: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Provider-based LLM call using `services.llm.factory.get_llm`.

    Returns (raw_text, parsed_response) where parsed_response is always a dict
    from ``parse_llm_json`` (never raises). Strict schema validation is used
    only for tracking when it succeeds.

    When ``prompt_override`` is set, it is sent as the full user prompt instead
    of rendering ``template_id`` from ``user_query`` (``user_query`` is still
    used for tracking metadata).
    """
    settings = get_settings()
    if settings.use_llm_dev_mock:
        raw = json.dumps(_DEV_MOCK_PARSED)
        safe_env = parse_llm_json(raw)
        safe_payload = safe_env.get("data") if isinstance(safe_env, dict) else None
        if not isinstance(safe_payload, dict):
            safe_payload = {"ranked_products": []}
        sid = session_id or str(uuid4())
        response_id = str(uuid4())
        parsed_dump: str | None = None
        parse_error: str | None = None
        try:
            parsed = _parse_financial_response(raw)
            parsed_dump = parsed.model_dump_json()
        except FinancialLLMResponseParseError as exc:
            parse_error = str(exc)
            try:
                parsed_dump = json.dumps(safe_payload)
            except Exception:
                parsed_dump = None
        try:
            await asyncio.to_thread(
                tracking_store.record_llm_response,
                response_id=response_id,
                session_id=sid,
                run_index=run_index,
                template_id=template_id,
                user_query=user_query,
                raw_content=raw,
                parsed_json=parsed_dump,
                parse_error=parse_error,
                model=f"{provider}:dev_mock",
            )
        except Exception:
            logger.exception("llm_tracking_write_failed")
        logger.info(
            "financial_llm_dev_mock",
            extra={"provider": provider, "template_id": template_id},
        )
        return raw, safe_payload

    if prompt_override is not None:
        prompt = prompt_override
    else:
        prompt = build_financial_prompt(user_query, template_id=template_id)
        prompt = prompt[:500]
    try:
        llm = get_llm(provider)
    except ValueError as exc:
        raise FinancialLLMConfigurationError(str(exc)) from exc

    print("Provider:", provider)
    print("Prompt preview:", prompt[:200])

    try:
        raw = await asyncio.to_thread(llm.generate, prompt)
    except Exception as exc:
        msg = str(exc) or exc.__class__.__name__
        raise FinancialLLMUpstreamError(msg) from exc

    print("LLM Output preview:", raw[:200] if isinstance(raw, str) else raw)

    if not isinstance(raw, str):
        raw = str(raw) if raw is not None else ""

    if _financial_llm_output_ignores_instructions(raw):
        strict_prompt = _STRICT_JSON_ONLY_PREFIX + prompt
        logger.warning(
            "financial_llm_instruction_retry",
            extra={"provider": provider, "template_id": template_id},
        )
        try:
            raw = await asyncio.to_thread(llm.generate, strict_prompt)
        except Exception as exc:
            msg = str(exc) or exc.__class__.__name__
            raise FinancialLLMUpstreamError(msg) from exc
        print("LLM Output preview (strict retry):", raw[:200] if isinstance(raw, str) else raw)
        if not isinstance(raw, str):
            raw = str(raw) if raw is not None else ""
        if _financial_llm_output_ignores_instructions(raw):
            raise FinancialLLMUpstreamError(
                "LLM output ignored JSON instructions after strict retry.",
            )

    safe_env = parse_llm_json(raw)
    safe_payload = safe_env.get("data") if isinstance(safe_env, dict) else None
    if not isinstance(safe_payload, dict):
        safe_payload = {"ranked_products": []}

    sid = session_id or str(uuid4())
    response_id = str(uuid4())

    parsed_dump: str | None = None
    parse_error: str | None = None
    try:
        parsed = _parse_financial_response(raw)
        parsed_dump = parsed.model_dump_json()
    except FinancialLLMResponseParseError as exc:
        parse_error = str(exc)
        try:
            parsed_dump = json.dumps(safe_payload)
        except Exception:
            parsed_dump = None

    try:
        await asyncio.to_thread(
            tracking_store.record_llm_response,
            response_id=response_id,
            session_id=sid,
            run_index=run_index,
            template_id=template_id,
            user_query=user_query,
            raw_content=raw,
            parsed_json=parsed_dump,
            parse_error=parse_error,
            model=f"{provider}",
        )
    except Exception:
        logger.exception("llm_tracking_write_failed")

    return raw, safe_payload


async def query_financial_llm(
    user_query: str,
    template_id: str = "financial_ranking",
    *,
    session_id: str | None = None,
    run_index: int = 0,
) -> FinancialQueryResponse:
    settings = get_settings()
    if settings.use_llm_dev_mock:
        raw = json.dumps(_DEV_MOCK_PARSED)
        result = _parse_financial_response(raw)
        sid = session_id or str(uuid4())
        response_id = str(uuid4())
        try:
            await asyncio.to_thread(
                tracking_store.record_llm_response,
                response_id=response_id,
                session_id=sid,
                run_index=run_index,
                template_id=template_id,
                user_query=user_query,
                raw_content=raw,
                parsed_json=result.model_dump_json(),
                parse_error=None,
                model=f"{settings.llm_model}:dev_mock",
            )
        except Exception:
            logger.exception("llm_tracking_write_failed")
        logger.info("financial_llm_dev_mock", extra={"template_id": template_id})
        return result

    if not settings.llm_api_key:
        raise FinancialLLMConfigurationError(
            "LLM is not configured. Set LLM_API_KEY or OPENAI_API_KEY.",
        )

    prompt = build_financial_prompt(user_query, template_id=template_id)
    prompt = prompt[:500]
    url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }
    body: dict[str, Any] = {
        "model": settings.llm_model,
        "temperature": settings.llm_temperature,
        "messages": [
            {"role": "user", "content": prompt},
        ],
    }

    timeout = httpx.Timeout(settings.llm_timeout_seconds)

    @retry(
        stop=stop_after_attempt(settings.llm_max_retries),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        retry=retry_if_exception(_retryable),
        reraise=True,
    )
    async def _post(client: httpx.AsyncClient) -> dict[str, Any]:
        response = await client.post(url, headers=headers, json=body)
        response.raise_for_status()
        return response.json()

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            data = await _post(client)
    except httpx.HTTPStatusError as exc:
        raise FinancialLLMUpstreamError(f"LLM HTTP error: {exc.response.status_code}") from exc
    except httpx.HTTPError as exc:
        raise FinancialLLMUpstreamError("LLM request failed.") from exc

    try:
        raw = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise FinancialLLMResponseParseError("Unexpected LLM response shape.") from exc
    if not isinstance(raw, str) or not raw.strip():
        raise FinancialLLMResponseParseError("Empty LLM content.")

    sid = session_id or str(uuid4())
    response_id = str(uuid4())
    parse_error: str | None = None
    parsed_dump: str | None = None
    try:
        result = _parse_financial_response(raw)
        parsed_dump = result.model_dump_json()
    except FinancialLLMResponseParseError as exc:
        parse_error = str(exc)
        try:
            await asyncio.to_thread(
                tracking_store.record_llm_response,
                response_id=response_id,
                session_id=sid,
                run_index=run_index,
                template_id=template_id,
                user_query=user_query,
                raw_content=raw,
                parsed_json=None,
                parse_error=parse_error,
                model=settings.llm_model,
            )
        except Exception:
            logger.exception("llm_tracking_write_failed")
        raise

    try:
        await asyncio.to_thread(
            tracking_store.record_llm_response,
            response_id=response_id,
            session_id=sid,
            run_index=run_index,
            template_id=template_id,
            user_query=user_query,
            raw_content=raw,
            parsed_json=parsed_dump,
            parse_error=None,
            model=settings.llm_model,
        )
        logger.info(
            "llm_response_stored",
            extra={
                "session_id": sid,
                "run_index": run_index,
                "response_id": response_id,
                "raw_chars": len(raw),
            },
        )
    except Exception:
        logger.exception("llm_tracking_write_failed")

    return result
