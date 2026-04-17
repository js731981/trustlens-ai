from __future__ import annotations

import asyncio
import json
from typing import Any, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.services import analyze as analyze_service
from app.services.financial_llm import build_financial_prompt
from app.services.utils.normalizer import normalize_output
from app.services.utils.parser import parse_llm_json
from app.services.utils.validator import validate_products

logger = get_logger(__name__)


def ranking_tool(query: str, provider: str, retrieval: dict[str, Any]) -> dict[str, Any]:
    """
    Use the existing LLM ranking pipeline (Ollama/OpenAI/OpenRouter via current services).
    Returns normalized+validated parsed_output plus raw output.
    """
    catalog_names = tuple(str(x) for x in (retrieval.get("catalog_names") or []))
    retrieved_docs = retrieval.get("retrieved_documents")
    if retrieved_docs is not None and not isinstance(retrieved_docs, list):
        retrieved_docs = None

    prompt = build_financial_prompt(
        query,
        catalog_product_names=catalog_names,
        retrieved_documents=retrieved_docs,
    )

    result = asyncio.run(analyze_service.run_analyze(query, provider=provider, prompt_override=prompt))
    # For the agent MVP we only orchestrate single-provider sequential flow.
    if hasattr(result, "results"):
        raise RuntimeError('CrewAI pipeline currently supports single-provider mode only (provider != "all").')

    raw_output = str(getattr(result, "raw_output", "") or "")
    try:
        parsed_env = parse_llm_json(raw_output)
    except Exception:
        logger.exception("agents_parse_llm_json_failed")
        parsed_env = {"data": {"ranked_products": []}, "llm_valid": False, "used_fallback": True, "parsing_success": False}

    parsed = parsed_env.get("data") if isinstance(parsed_env, dict) else None
    if not isinstance(parsed, dict):
        parsed = {"ranked_products": []}

    normalized = normalize_output(parsed)
    validated = validate_products(normalized, catalog_names)
    return {
        "provider_used": str(getattr(result, "provider_used", provider)),
        "fallback_used": bool(getattr(result, "fallback_used", False)),
        "raw_output": raw_output,
        # Raw (pre-normalization) JSON extracted from the model output.
        "raw_llm_json": parsed,
        "parsed_output": validated,
        "llm_valid": bool(parsed_env.get("llm_valid")) if isinstance(parsed_env, dict) else False,
        "used_fallback": bool(parsed_env.get("used_fallback")) if isinstance(parsed_env, dict) else (not bool(parsed_env.get("llm_valid"))),
        "parsing_success": bool(parsed_env.get("llm_valid")) if isinstance(parsed_env, dict) else False,
        "prompt_used": prompt,
    }


def explanation_tool(parsed_output: dict[str, Any]) -> dict[str, Any]:
    """
    Produce a user-facing explanation payload (keeps parity with existing single-provider behavior).
    """
    exp = str((parsed_output or {}).get("explanation") or "").strip()
    ranked = (parsed_output or {}).get("ranked_products") or []
    n = len(ranked) if isinstance(ranked, list) else 0
    if exp:
        summary = exp if len(exp) <= 2000 else f"{exp[:1999]}…"
        insights = [f"The model returned {n} ranked product(s) with an accompanying rationale."]
        return {"summary": summary, "insights": insights}
    summary = (
        'Single-provider response: run with provider "all" to compute overlap, stability, and aggregate trust across models.'
    )
    return {"summary": summary, "insights": [f"The model returned {n} ranked product(s)."]}


def to_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


class RankingInput(BaseModel):
    query: str = Field(..., description="User query")
    provider: str = Field(..., description="LLM provider identifier (e.g., ollama/openai/openrouter)")
    retrieval: dict[str, Any] = Field(..., description="Retrieval payload from RAG tool")


class RankingTool(BaseTool):
    name: str = "Financial Ranking Tool"
    description: str = "Rank financial products using LLM with retrieved context"

    args_schema: Type[BaseModel] = RankingInput

    def _run(self, query: str, provider: str, retrieval: dict[str, Any]) -> dict[str, Any]:
        return ranking_tool(query=query, provider=provider, retrieval=retrieval)


class ExplanationInput(BaseModel):
    parsed_output: dict[str, Any] = Field(..., description="Validated parsed output from ranking step")


class ExplanationTool(BaseTool):
    name: str = "Explanation Tool"
    description: str = "Generate a user-facing explanation from structured ranking output"

    args_schema: Type[BaseModel] = ExplanationInput

    def _run(self, parsed_output: dict[str, Any]) -> dict[str, Any]:
        return explanation_tool(parsed_output=parsed_output)

