import json
from pathlib import Path
from typing import Any, Iterable, cast

from fastapi import APIRouter, HTTPException, status

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.analyze import (
    AnalyzeApiDebug,
    AnalyzeApiExplanation,
    AnalyzeApiMetrics,
    AnalyzeApiResponse,
    AnalyzeApiTrust,
    AnalyzeComparisonResponse,
    AnalyzeProviderError,
    AnalyzeProviderResult,
    AnalyzeRequest,
    AnalyzeResponse,
    HistoryEntry,
    ProviderName,
)
from app.prompts.registry import UnknownPromptTemplateError
from app.services import analyze as analyze_service
from app.services import financial_llm as financial_llm_service
from app.services import rag_client
from app.services.financial_llm import LLM_DEFLECTION_MARKER, build_financial_prompt
from app.services.query_intent import classify_query
from services.llm.prompt_builder import load_catalog_product_names
from services.trust.accuracy_scorer import (
    accuracy_score_vs_catalog,
    compute_accuracy,
    merged_provider_ranked_names,
)
from services.trust.ground_truth import load_ground_truth_for_query
from services.trust.explainer import explain_trust
from services.trust.ranking_comparator import compare_rankings
from services.trust.trust_scorer import compute_trust_score
from services.utils.normalizer import normalize_output
from services.utils.parser import parse_llm_json
from services.utils.validator import validate_products

router = APIRouter()
logger = get_logger(__name__)

_PROVIDER_ORDER: tuple[ProviderName, ...] = ("ollama", "openai", "openrouter")


def _providers_in_order(keys: Iterable[str]) -> list[ProviderName]:
    key_set = set(keys)
    return [p for p in _PROVIDER_ORDER if p in key_set]


def _ranked_products_empty(parsed_output: dict[str, Any]) -> bool:
    ranked = parsed_output.get("ranked_products")
    return not isinstance(ranked, list) or len(ranked) == 0


def _ranked_products_count(parsed_output: dict[str, Any]) -> int:
    ranked = parsed_output.get("ranked_products")
    return len(ranked) if isinstance(ranked, list) else 0


def _debug_for_result(entry: AnalyzeProviderResult, repair_applied: bool) -> AnalyzeApiDebug:
    return AnalyzeApiDebug(
        raw_length=len(entry.raw_output),
        parsed_items=_ranked_products_count(entry.parsed_output),
        repair_applied=repair_applied,
    )


def _ranking_catalog_path(intent: str, data_dir: Path) -> Path:
    if intent == "loan":
        return data_dir / "loan_providers.json"
    return data_dir / "insurance_products.json"


async def _retry_ranking_parse_if_empty(
    query: str,
    provider: ProviderName,
    shaped: AnalyzeResponse,
    original_prompt: str,
    catalog_names: tuple[str, ...],
) -> tuple[AnalyzeResponse, bool]:
    if not _ranked_products_empty(shaped.parsed_output):
        return shaped, False
    retry_prompt = "STRICT JSON ONLY. FIX FORMAT.\n\n" + original_prompt
    try:
        retry_result = await analyze_service.run_analyze(
            query,
            provider=provider,
            prompt_override=retry_prompt,
        )
    except Exception:
        logger.exception("analyze_format_retry_failed", extra={"provider": provider})
        return shaped, False
    if isinstance(retry_result, AnalyzeComparisonResponse):
        return shaped, False
    shaped_retry = _try_shape_analyze_response(retry_result, catalog_names)
    if isinstance(shaped_retry, AnalyzeProviderError):
        return shaped, False
    if not _ranked_products_empty(shaped_retry.parsed_output):
        return shaped_retry, True
    return shaped, False


def _explanation_single(parsed: dict[str, Any]) -> AnalyzeApiExplanation:
    exp = str(parsed.get("explanation") or "").strip()
    ranked = parsed.get("ranked_products") or []
    n = len(ranked) if isinstance(ranked, list) else 0
    if exp:
        summary = exp if len(exp) <= 2000 else f"{exp[:1999]}…"
        insights = [
            f"The model returned {n} ranked product(s) with an accompanying rationale.",
        ]
        return AnalyzeApiExplanation(summary=summary, insights=insights)
    summary = (
        "Single-provider response: run with provider \"all\" to compute overlap, "
        "stability, and aggregate trust across models."
    )
    insights = [f"The model returned {n} ranked product(s)."]
    return AnalyzeApiExplanation(summary=summary, insights=insights)


def _normalize_parsed_from_raw(raw_output: str, catalog_names: tuple[str, ...]) -> dict:
    try:
        parsed_output = parse_llm_json(raw_output)
    except Exception:
        logger.exception("parse_llm_json_unexpected")
        parsed_output = {"ranked_products": []}
    normalized_output = normalize_output(parsed_output)
    validated_output = validate_products(normalized_output, catalog_names)
    logger.debug(
        "parsed_output=%s",
        json.dumps(parsed_output, ensure_ascii=False, default=str),
    )
    logger.debug(
        "normalized_output=%s",
        json.dumps(normalized_output, ensure_ascii=False, default=str),
    )
    return validated_output


def _try_shape_analyze_response(
    result: AnalyzeResponse,
    catalog_names: tuple[str, ...],
) -> AnalyzeResponse | AnalyzeProviderError:
    """Parse and normalize one provider result; deflection becomes an error object (for multi-LLM)."""
    raw_output = result.raw_output
    if LLM_DEFLECTION_MARKER in raw_output:
        return AnalyzeProviderError(
            error="LLM did not process query correctly",
            raw_output=raw_output,
            parsed_output={},
        )
    normalized_output = _normalize_parsed_from_raw(raw_output, catalog_names)
    return AnalyzeResponse(
        provider_used=result.provider_used,
        fallback_used=result.fallback_used,
        raw_output=raw_output,
        parsed_output=normalized_output,
    )


def _ground_truth_accuracy_and_trust(
    query: str,
    data_dir: Path,
    results_for_compare: dict[str, Any],
    metrics_full: dict[str, float] | None,
) -> tuple[float | None, float | None]:
    """Compare merged LLM rankings to ``ground_truth.json``; return accuracy and blended trust_score."""
    gt = load_ground_truth_for_query(query, data_dir / "ground_truth.json")
    if gt is None:
        return None, None
    predicted = merged_provider_ranked_names(results_for_compare)
    acc = float(compute_accuracy(predicted, gt)["accuracy"])
    if metrics_full is not None:
        blended = dict(metrics_full)
        blended["accuracy_score"] = acc
        trust = float(compute_trust_score(blended)["trust_score"])
    else:
        trust = float(
            compute_trust_score(
                {
                    "overlap_score": 1.0,
                    "stability_score": 1.0,
                    "rank_variance": 0.0,
                    "accuracy_score": acc,
                }
            )["trust_score"]
        )
    return acc, trust


def _shape_analyze_response(result: AnalyzeResponse, catalog_names: tuple[str, ...]) -> AnalyzeResponse:
    shaped = _try_shape_analyze_response(result, catalog_names)
    if isinstance(shaped, AnalyzeProviderError):
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            detail=shaped.error,
        )
    return shaped


@router.post(
    "/analyze",
    response_model=AnalyzeApiResponse,
    status_code=status.HTTP_200_OK,
)
async def analyze(body: AnalyzeRequest) -> AnalyzeApiResponse:
    try:
        query = body.query
        intent = classify_query(query)
        settings = get_settings()
        catalog_path = _ranking_catalog_path(intent, settings.data_dir)
        rag_ctx = await rag_client.fetch_rag_context_for_query(query, settings)
        if rag_ctx is not None:
            catalog_names = rag_ctx.catalog_names
            retrieved_docs = rag_ctx.retrieved_documents
            logger.debug("analyze_catalog_source", extra={"source": "rag", "n": len(catalog_names)})
        else:
            catalog_names = load_catalog_product_names(catalog_path)
            retrieved_docs = None
            logger.debug(
                "analyze_catalog_source",
                extra={"source": "static_file", "path": str(catalog_path)},
            )
        original_prompt = build_financial_prompt(
            query,
            catalog_product_names=catalog_names,
            retrieved_documents=retrieved_docs,
        )
        result = await analyze_service.run_analyze(
            query,
            provider=body.provider,
            prompt_override=original_prompt,
        )
        if isinstance(result, AnalyzeComparisonResponse):
            shaped: dict[str, AnalyzeResponse | AnalyzeProviderError] = {}
            repair_applied_by_provider: dict[str, bool] = {}
            for key, entry in result.results.items():
                if isinstance(entry, AnalyzeResponse):
                    first = _try_shape_analyze_response(entry, catalog_names)
                    if isinstance(first, AnalyzeResponse):
                        retried, repair_applied = await _retry_ranking_parse_if_empty(
                            query,
                            cast(ProviderName, key),
                            first,
                            original_prompt,
                            catalog_names,
                        )
                        shaped[key] = retried
                        repair_applied_by_provider[key] = repair_applied
                    else:
                        shaped[key] = first
                        repair_applied_by_provider[key] = False
                else:
                    shaped[key] = entry
                    repair_applied_by_provider[key] = False
            results_for_compare: dict[str, Any] = {}
            for key, entry in shaped.items():
                if isinstance(entry, AnalyzeResponse):
                    results_for_compare[key] = entry.parsed_output
                else:
                    results_for_compare[key] = entry.parsed_output or {}
            metrics_full = compare_rankings(results_for_compare)
            metrics_full["accuracy_score"] = accuracy_score_vs_catalog(results_for_compare, catalog_names)
            trust_raw = compute_trust_score(metrics_full)
            gt_accuracy, gt_trust_score = _ground_truth_accuracy_and_trust(
                query, settings.data_dir, results_for_compare, metrics_full
            )
            explained = explain_trust(metrics_full, trust_raw)
            providers_used = _providers_in_order(shaped.keys())
            debug_by_provider: dict[ProviderName, AnalyzeApiDebug] = {
                cast(ProviderName, k): _debug_for_result(
                    shaped[k],
                    repair_applied_by_provider.get(k, False),
                )
                for k in providers_used
            }
            if len(debug_by_provider) == 0:
                debug_payload = AnalyzeApiDebug(
                    raw_length=0,
                    parsed_items=0,
                    repair_applied=False,
                )
            elif len(debug_by_provider) == 1:
                debug_payload = next(iter(debug_by_provider.values()))
            else:
                debug_payload = debug_by_provider
            return AnalyzeApiResponse(
                query=query,
                providers_used=providers_used,
                results=shaped,
                metrics=AnalyzeApiMetrics(
                    overlap_score=float(metrics_full["overlap_score"]),
                    stability_score=float(metrics_full["stability_score"]),
                    rank_variance=float(metrics_full["rank_variance"]),
                    accuracy_score=float(metrics_full["accuracy_score"]),
                ),
                trust=AnalyzeApiTrust(
                    score=float(trust_raw["trust_score"]),
                    confidence=trust_raw["confidence_level"],
                    stability_score=float(trust_raw["stability_score"]),
                    accuracy_score=float(trust_raw["accuracy_score"]),
                    overlap_score=float(trust_raw["overlap_score"]),
                    stability_component=float(trust_raw["stability_component"]),
                    accuracy_component=float(trust_raw["accuracy_component"]),
                    overlap_component=float(trust_raw["overlap_component"]),
                ),
                explanation=AnalyzeApiExplanation(
                    summary=explained["summary"],
                    insights=list(explained["insights"]),
                ),
                debug=debug_payload,
                accuracy=gt_accuracy,
                trust_score=gt_trust_score,
            )
        try_first = _try_shape_analyze_response(result, catalog_names)
        if isinstance(try_first, AnalyzeProviderError):
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY,
                detail=try_first.error,
            )
        shaped_one, repair_applied = await _retry_ranking_parse_if_empty(
            query,
            cast(ProviderName, body.provider),
            try_first,
            original_prompt,
            catalog_names,
        )
        p = shaped_one.provider_used
        results_one = {p: shaped_one.parsed_output}
        gt_accuracy, gt_trust_score = _ground_truth_accuracy_and_trust(
            query, settings.data_dir, results_one, None
        )
        return AnalyzeApiResponse(
            query=query,
            providers_used=[p],
            results={p: shaped_one},
            metrics=None,
            trust=None,
            explanation=_explanation_single(shaped_one.parsed_output),
            debug=_debug_for_result(shaped_one, repair_applied),
            accuracy=gt_accuracy,
            trust_score=gt_trust_score,
        )
    except UnknownPromptTemplateError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except financial_llm_service.FinancialLLMConfigurationError as exc:
        logger.warning("analyze_misconfigured", extra={"detail": str(exc)})
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except financial_llm_service.FinancialLLMUpstreamError as e:
        print(e)
        logger.error("analyze_upstream_error", extra={"detail": str(e)})
        raise HTTPException(status_code=502, detail=str(e)) from e


@router.get(
    "/history",
    response_model=list[HistoryEntry],
    status_code=status.HTTP_200_OK,
)
def history() -> list[HistoryEntry]:
    """Return past analyze queries from persistent storage (most recent first)."""
    return analyze_service.list_history()
