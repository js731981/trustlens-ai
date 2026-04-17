from __future__ import annotations

from typing import Any, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.geo.geo_service import analyze_geo
from app.services.history.history_service import save_query
from app.services.trust.accuracy_scorer import compute_accuracy, merged_provider_ranked_names
from app.services.trust.ground_truth import load_ground_truth_for_query
from app.services.trust.trust_scorer import compute_trust_score

logger = get_logger(__name__)

KNOWN_BRANDS = [
    "hdfc",
    "icici",
    "axis",
    "sbi",
    "bajaj",
    "kotak",
    "idfc",
    "pnb",
    "star health",
]

_FINANCIAL_KEYWORDS = ("bank", "insurance", "loan")
_CREDIBILITY_KEYWORDS = ("trusted", "reputed", "well-established", "leading")


def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _brands_in_name(product_name: str) -> list[str]:
    name = (product_name or "").lower()
    found: list[str] = []
    for brand in KNOWN_BRANDS:
        b = (brand or "").lower()
        if not b:
            continue
        if b in name:
            found.append(brand)
    # de-dupe but keep order
    out: list[str] = []
    seen: set[str] = set()
    for b in found:
        if b in seen:
            continue
        seen.add(b)
        out.append(b)
    return out


def _heuristic_trust_from_ranked(
    ranked_products: list[dict[str, Any]],
    *,
    explanation: str = "",
    rag_has_context: bool = False,
) -> tuple[float, str]:
    """
    Deterministic trust heuristic when ground truth is unavailable.

    Scored per product, then averaged across the ranked list:
    - brand in product name → +0.5
    - product contains financial keyword ("bank", "insurance", "loan") → +0.2
    - explanation contains credibility keywords ("trusted", "reputed", "well-established", "leading") → +0.2
    - retrieved context exists (RAG) → +0.1
    - per-product score capped at 1.0
    """
    items: list[str] = []
    scores: list[float] = []

    exp = (explanation or "").strip().lower()
    exp_has_cred = bool(exp) and any(k in exp for k in _CREDIBILITY_KEYWORDS)

    for item in ranked_products:
        if not isinstance(item, dict):
            continue
        product = str(item.get("name") or "").strip()
        if not product:
            continue

        trust = 0.0
        parts: list[str] = []

        brands = _brands_in_name(product)
        if brands:
            trust += 0.5
            parts.append(f"Detected {brands[0]} → +0.5")

        name_norm = _norm(product)
        matched_kw = next((k for k in _FINANCIAL_KEYWORDS if k in name_norm), None)
        if matched_kw is not None:
            trust += 0.2
            parts.append(f"keyword {matched_kw} → +0.2")

        if exp_has_cred:
            trust += 0.2
            parts.append("credibility wording in explanation → +0.2")

        if rag_has_context:
            trust += 0.1
            parts.append("RAG context → +0.1")

        trust = float(min(trust, 1.0))
        scores.append(trust)
        items.append(f"{product}: {', '.join(parts) if parts else 'no signals'}")

    if not scores:
        return 0.0, "No products to score"

    avg = float(sum(scores) / len(scores))
    # Keep reasons short but useful.
    sample = items[:5]
    suffix = "" if len(items) <= len(sample) else f" (and {len(items) - len(sample)} more)"
    return float(min(avg, 1.0)), "Avg trust across products. " + " | ".join(sample) + suffix


def trust_tool(query: str, ranking: dict[str, Any]) -> dict[str, Any]:
    """
    Compute trust score (ground-truth blended when available) and GEO score reusing existing services.

    Note: `/v1/analyze` single-provider responses historically keep `trust=None` and put
    ground-truth trust into `trust_score` when available; we preserve that behavior in orchestration.
    """
    settings = get_settings()
    # Ranking agent now outputs strict: { "ranked_products": [...] }
    ranked_products = ranking.get("ranked_products")
    if not isinstance(ranked_products, list):
        ranked_products = []
    parsed_output = {"ranked_products": ranked_products}

    explanation = str(ranking.get("explanation") or "")
    if explanation:
        parsed_output["explanation"] = explanation
    rag_has_context = bool(ranking.get("rag_has_context") or False)
    llm_valid = bool(ranking.get("llm_valid")) if "llm_valid" in ranking else True

    geo = analyze_geo(parsed_output, rag_has_context=rag_has_context, llm_valid=llm_valid)

    # Ground-truth based trust_score (optional)
    gt = load_ground_truth_for_query(query, settings.data_dir / "ground_truth.json")
    if gt is None:
        trust_score, trust_reason = _heuristic_trust_from_ranked(
            [x for x in ranked_products if isinstance(x, dict)],
            explanation=explanation,
            rag_has_context=rag_has_context,
        )
        if not llm_valid:
            trust_score = float(trust_score) * 0.5
            trust_reason = f"{trust_reason}; LLM output invalid → trust_score × 0.5"
        return {
            "geo": geo,
            "accuracy": None,
            "trust_score": float(trust_score),
            "trust_reason": trust_reason,
            "geo_reason": str(geo.get("geo_reason") or geo.get("reason") or ""),
            "llm_valid": llm_valid,
        }

    provider_used = str(ranking.get("provider_used") or "ollama")
    results_for_compare = {provider_used: parsed_output}
    predicted = merged_provider_ranked_names(results_for_compare)
    acc = float(compute_accuracy(predicted, gt)["accuracy"])
    # Even when ground truth exists, return the stronger per-product heuristic trust score
    # (accuracy remains available for debugging/monitoring).
    trust_score, trust_reason = _heuristic_trust_from_ranked(
        [x for x in ranked_products if isinstance(x, dict)],
        explanation=explanation,
        rag_has_context=rag_has_context,
    )
    if not llm_valid:
        trust_score = float(trust_score) * 0.5
        trust_reason = f"{trust_reason}; LLM output invalid → trust_score × 0.5"
    return {
        "geo": geo,
        "accuracy": acc,
        "trust_score": float(trust_score),
        "trust_reason": trust_reason,
        "geo_reason": str(geo.get("geo_reason") or geo.get("reason") or ""),
        "llm_valid": llm_valid,
    }


def analytics_tool(query: str, provider: str, trust_score: float | None, geo: dict[str, Any] | None) -> dict[str, Any]:
    """
    Persist query to DB using existing history service.
    """
    try:
        save_query(
            {
                "query": query,
                "provider": provider,
                "trust_score": float(trust_score or 0.0),
                "geo_score": float((geo or {}).get("score", 0.0)) if isinstance(geo, dict) else 0.0,
            }
        )
    except Exception:
        logger.exception("agents_history_save_failed")
    return {"saved": True}


class TrustInput(BaseModel):
    query: str = Field(..., description="User query")
    ranking: dict[str, Any] = Field(..., description="Ranking output payload")


class TrustTool(BaseTool):
    name: str = "Trust Scoring Tool"
    description: str = "Compute GEO/accuracy/trust score for a ranked response"

    args_schema: Type[BaseModel] = TrustInput

    def _run(self, query: str, ranking: dict[str, Any]) -> dict[str, Any]:
        return trust_tool(query=query, ranking=ranking)


class AnalyticsInput(BaseModel):
    query: str = Field(..., description="User query")
    provider: str = Field(..., description="Provider used")
    trust_score: float | None = Field(None, description="Trust score (if available)")
    geo: dict[str, Any] | None = Field(None, description="GEO analysis payload (if available)")


class AnalyticsTool(BaseTool):
    name: str = "Analytics Persistence Tool"
    description: str = "Persist the run metadata for monitoring/analytics"

    args_schema: Type[BaseModel] = AnalyticsInput

    def _run(self, query: str, provider: str, trust_score: float | None = None, geo: dict[str, Any] | None = None) -> dict[str, Any]:
        return analytics_tool(query=query, provider=provider, trust_score=trust_score, geo=geo)

