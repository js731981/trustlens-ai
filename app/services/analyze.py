from __future__ import annotations

import asyncio
from typing import Any, cast

from app.core.logging import get_logger
from app.services import financial_llm as financial_llm_service
from app.models.analyze import (
    AnalyzeComparisonResponse,
    AnalyzeProviderError,
    AnalyzeProviderResult,
    AnalyzeResponse,
    HistoryEntry,
    ProviderName,
)
from app.services.tracking_store import list_analyze_history

logger = get_logger(__name__)


def list_history() -> list[HistoryEntry]:
    try:
        return list_analyze_history(limit=200)
    except Exception:
        logger.exception("history_read_failed")
        return []


async def run_analyze(
    user_query: str,
    *,
    provider: str = "ollama",
    prompt_override: str | None = None,
) -> AnalyzeResponse | AnalyzeComparisonResponse:
    if provider == "all":
        providers: tuple[ProviderName, ...] = ("ollama", "openai", "openrouter")

        async def _run_one(p: ProviderName, run_index: int) -> tuple[ProviderName, AnalyzeProviderResult]:
            try:
                raw, parsed = await financial_llm_service.query_financial_llm_multi(
                    user_query,
                    provider=p,
                    template_id="financial_ranking",
                    run_index=run_index,
                    prompt_override=prompt_override,
                )
                return p, AnalyzeResponse(provider_used=p, raw_output=raw, parsed_output=parsed)
            except Exception as exc:
                msg = str(exc) or exc.__class__.__name__
                return p, AnalyzeProviderError(error=msg, raw_output="", parsed_output={})

        tasks = [_run_one(p, idx) for idx, p in enumerate(providers)]
        pairs = await asyncio.gather(*tasks)
        return AnalyzeComparisonResponse(results=dict(pairs))

    requested = cast(ProviderName, provider)

    async def _call_llm(p: ProviderName) -> tuple[str, dict[str, Any]]:
        return await financial_llm_service.query_financial_llm_multi(
            user_query,
            provider=p,
            template_id="financial_ranking",
            prompt_override=prompt_override,
        )

    if requested == "ollama":
        raw, parsed = await _call_llm("ollama")
        return AnalyzeResponse(provider_used="ollama", fallback_used=False, raw_output=raw, parsed_output=parsed)

    try:
        raw, parsed = await _call_llm(requested)
    except (
        financial_llm_service.FinancialLLMUpstreamError,
        financial_llm_service.FinancialLLMConfigurationError,
    ) as exc:
        logger.warning(
            "analyze_provider_fallback",
            extra={
                "requested_provider": requested,
                "fallback_provider": "ollama",
                "error": str(exc),
            },
        )
        try:
            raw, parsed = await _call_llm("ollama")
        except Exception as ollama_exc:
            logger.error(
                "analyze_fallback_failed",
                extra={
                    "requested_provider": requested,
                    "fallback_provider": "ollama",
                    "primary_error": str(exc),
                    "fallback_error": str(ollama_exc) or ollama_exc.__class__.__name__,
                },
            )
            raise ollama_exc from exc

        return AnalyzeResponse(
            provider_used="ollama",
            fallback_used=True,
            raw_output=raw,
            parsed_output=parsed,
        )

    return AnalyzeResponse(
        provider_used=requested,
        fallback_used=False,
        raw_output=raw,
        parsed_output=parsed,
    )
