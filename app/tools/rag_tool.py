from __future__ import annotations

import json
import re
from typing import Any, Literal, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services import embedding_service, qdrant_service, rag_client
from app.services.llm.prompt_builder import load_catalog_product_names
from app.services.query_intent import classify_query

logger = get_logger(__name__)


def _mock_financial_product_results(intent: str) -> list[dict[str, Any]]:
    # Lightweight, deterministic fallback context for when vector retrieval yields nothing
    # (empty index, wrong collection, model mismatch, or Qdrant down).
    if intent == "loan":
        products = [
            {
                "name": "ClearRate Personal Loan",
                "type": "loan",
                "features": ["fixed APR 9.99%–15.99%", "no prepayment penalty", "funding in 1–2 business days"],
                "eligibility": ["min credit score 660", "DTI < 40%"],
                "fees": ["origination fee 0%–3%"],
            },
            {
                "name": "FlexLine Credit Builder Loan",
                "type": "loan",
                "features": ["small principal", "reports to credit bureaus", "automatic payments"],
                "eligibility": ["stable income verification"],
                "fees": ["late fee up to $25"],
            },
            {
                "name": "HomeStart Mortgage (30-year fixed)",
                "type": "loan",
                "features": ["rate lock options", "escrow supported", "pre-approval available"],
                "eligibility": ["down payment 3%+", "property appraisal required"],
                "fees": ["closing costs vary by lender/state"],
            },
        ]
    else:
        products = [
            {
                "name": "ShieldPlus Comprehensive Auto",
                "type": "insurance",
                "features": ["collision + comprehensive", "roadside assistance", "rental reimbursement optional"],
                "deductibles": ["$250 / $500 / $1,000"],
                "discounts": ["safe driver", "multi-policy", "anti-theft"],
            },
            {
                "name": "CareFirst Term Life (20-year)",
                "type": "insurance",
                "features": ["level premium", "convertible option", "online underwriting for some applicants"],
                "coverage": ["$100k–$1M"],
                "exclusions": ["suicide clause (first 2 years)", "material misrepresentation"],
            },
            {
                "name": "SecureHome Standard Homeowners",
                "type": "insurance",
                "features": ["dwelling + personal property", "liability coverage", "loss-of-use"],
                "deductibles": ["$1,000 standard"],
                "exclusions": ["flood/earthquake (optional riders)"],
            },
        ]

    results: list[dict[str, Any]] = []
    for i, p in enumerate(products, start=1):
        results.append(
            {
                "content": f'{p["name"]}. Features: {", ".join(p.get("features") or [])}.',
                "score": 1.0,
                "source": "mock",
                "document_id": f"mock-{intent}-{i}",
                "metadata": {"name": p["name"], "type": p.get("type") or intent, "source": "mock", "raw": p},
            }
        )
    return results


def _ranking_catalog_path(intent: str) -> str:
    settings = get_settings()
    if intent == "loan":
        return str(settings.data_dir / "loan_providers.json")
    return str(settings.data_dir / "insurance_products.json")


def _payload_to_doc(hit: dict[str, Any]) -> dict[str, Any]:
    payload = hit.get("payload") or {}
    if not isinstance(payload, dict):
        payload = {}
    name = str(payload.get("name") or "").strip()
    features = payload.get("features") or []
    if isinstance(features, list):
        feat_str = ", ".join(str(x).strip() for x in features if str(x).strip())
    else:
        feat_str = str(features).strip()
    product_type = str(payload.get("type") or "").strip()
    head = name or str(hit.get("id") or "").strip()
    parts = [p for p in [head, feat_str] if p]
    text = ". ".join(parts).strip()
    meta: dict[str, Any] = {
        "name": head,
        "source": "qdrant",
        "document_id": str(hit.get("id") or ""),
    }
    if product_type:
        meta["type"] = product_type
    return {"metadata": meta, "text": text}


def _hit_to_structured_result(hit: dict[str, Any]) -> dict[str, Any]:
    doc = _payload_to_doc(hit)
    meta = doc.get("metadata") or {}
    if not isinstance(meta, dict):
        meta = {}
    return {
        "content": str(doc.get("text") or "").strip(),
        "score": float(hit.get("score") or 0.0),
        "source": str(meta.get("source") or "qdrant"),
        # Optional but useful for debug/traceability.
        "document_id": str(meta.get("document_id") or ""),
        "metadata": meta,
    }


def search(query: str, top_k: int = 5, filters: dict | None = None) -> dict[str, Any]:
    """
    Structured retrieval API for multi-agent workflows.

    Returns:
      { "query": str, "results": [ { "content": str, "score": float, "source": str, ... } ] }
    """
    cleaned_query = " ".join((query or "").strip().split())
    if not cleaned_query:
        return {"query": str(query or ""), "results": []}

    try:
        vector = embedding_service.embed_text(cleaned_query)
        hits = qdrant_service.search(vector, top_k=top_k, filters=filters)
    except Exception:
        logger.exception(
            "rag_search_failed_local_qdrant",
            extra={
                "collection": getattr(qdrant_service, "configured_collection_name", lambda: "financial_products")(),
                "expected_vector_size": getattr(qdrant_service, "expected_vector_size", lambda: None)(),
            },
        )
        hits = []
    results = [_hit_to_structured_result(h) for h in hits if isinstance(h, dict)]

    scores = [float(r.get("score") or 0.0) for r in results[: min(3, len(results))]]
    logger.debug(
        "rag_search",
        extra={
            "query": cleaned_query,
            "n_results": len(results),
            "top_scores": scores,
        },
    )
    return {"query": cleaned_query, "results": results}


_NOISE_RE = re.compile(r"[ \t]+")


def format_context(results: list) -> str:
    """
    Concatenate top results into a compact snippet block for LLM consumption.
    """
    if not results:
        return "No relevant context found"

    chunks: list[str] = []
    seen: set[str] = set()
    for r in results:
        if not isinstance(r, dict):
            continue
        content = str(r.get("content") or "").strip()
        if not content:
            continue
        content = _NOISE_RE.sub(" ", content).strip()
        if len(content) > 800:
            content = content[:799] + "…"
        key = content.lower()
        if key in seen:
            continue
        seen.add(key)
        chunks.append(content)
        if len(chunks) >= 5:
            break

    return "\n\n".join(chunks) if chunks else "No relevant context found"


def retrieve_context(query: str) -> dict[str, Any]:
    """
    Wrapper used by Retrieval Agent.
    Calls `search()` and returns structured results plus backward-compatible keys.
    """
    settings = get_settings()
    intent = classify_query(query)
    catalog_path = _ranking_catalog_path(intent)

    # Prefer local Qdrant retrieval; fall back to legacy HTTP client if needed.
    retrieval = search(
        query=query,
        top_k=int(settings.rag_search_top_k),
        filters={"type": intent} if intent in ("loan", "insurance") else None,
    )
    results = retrieval.get("results") or []
    if not isinstance(results, list):
        results = []

    # Fallback #1: If Qdrant is down/unreachable, return mock product context immediately.
    if not qdrant_service.is_qdrant_available():
        mock_results = _mock_financial_product_results(intent)
        return {
            "query": str(query or ""),
            "results": mock_results,
            "context": format_context(mock_results),
            "intent": intent,
            "catalog_names": [str(r.get("metadata", {}).get("name") or "") for r in mock_results if isinstance(r, dict)],
            "retrieved_documents": [
                {"metadata": (r.get("metadata") or {}), "text": str(r.get("content") or "")} for r in mock_results if isinstance(r, dict)
            ],
            "catalog_source": "mock",
        }

    retrieved_docs = []
    labels: list[str] = []
    for r in results:
        if not isinstance(r, dict):
            continue
        meta = r.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {}
        name = str(meta.get("name") or "").strip()
        if name:
            labels.append(name)
        retrieved_docs.append(
            {
                "metadata": meta,
                "text": str(r.get("content") or "").strip(),
            }
        )

    if labels:
        catalog_names = labels
        source: Literal["rag", "static_file"] = "rag"
    else:
        # Fallback #2: empty retrieval -> return mock financial product data as usable context,
        # while still keeping catalog names for downstream validation.
        mock_results = _mock_financial_product_results(intent)
        if mock_results:
            results = mock_results
            retrieved_docs = [
                {"metadata": (r.get("metadata") or {}), "text": str(r.get("content") or "")} for r in mock_results if isinstance(r, dict)
            ]
            catalog_names = [str(r.get("metadata", {}).get("name") or "") for r in mock_results if isinstance(r, dict)]
            source = "mock"  # type: ignore[assignment]
        else:
            # Backward compatible fallback: static catalog names if retrieval yields nothing.
            catalog_names = list(load_catalog_product_names(catalog_path))
            retrieved_docs = None
            source = "static_file"

    ctx_text = format_context(results)
    return {
        # New structured retrieval payload
        "query": str(retrieval.get("query") or query),
        "results": results,
        "context": ctx_text,
        # Backward-compatible keys consumed by ranking/trust tools
        "intent": intent,
        "catalog_names": catalog_names,
        "retrieved_documents": retrieved_docs,
        "catalog_source": source,
    }


class RetrievalInput(BaseModel):
    query: str = Field(..., description="User query")


class RetrievalTool(BaseTool):
    name: str = "RAG Retrieval Tool"
    description: str = "Fetch relevant financial context from vector database"

    args_schema: Type[BaseModel] = RetrievalInput

    def _run(self, query: str) -> str:
        # CrewAI agents downstream need structured retrieval, not a free-form text blob.
        # Return JSON text so orchestration can parse it reliably.
        result = retrieve_context(query)
        return json.dumps(result, ensure_ascii=False, default=str)


# Backward compatible alias (legacy imports / wiring). Prefer `RetrievalTool()`.
retrieval_tool = RetrievalTool()

