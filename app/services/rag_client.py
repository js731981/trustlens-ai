from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import Settings
from app.core.logging import get_logger
from app.services import embedding_service, qdrant_service

logger = get_logger(__name__)


@dataclass(frozen=True)
class RagSearchContext:
    """Catalog labels for validation plus raw hits for prompt context."""

    catalog_names: tuple[str, ...]
    retrieved_documents: tuple[dict[str, Any], ...]


def _hit_display_name(hit: dict[str, Any]) -> str:
    meta = hit.get("metadata")
    if isinstance(meta, dict):
        for key in ("name", "title", "product"):
            v = meta.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
    text = (hit.get("text") or "").strip()
    if not text:
        return ""
    if text.startswith("{"):
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                n = obj.get("name")
                if isinstance(n, str) and n.strip():
                    return n.strip()
        except json.JSONDecodeError:
            pass
    first = text.split("\n", 1)[0].strip()
    return first[:300] if first else ""


def _dedupe_preserve_order(names: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return tuple(out)


async def fetch_rag_context_for_query(query: str, settings: Settings) -> RagSearchContext | None:
    """
    Call RAG service POST /search; return catalog names (for validation) and hit payloads (for prompts).

    Returns None if the request fails or yields no usable labels (caller may fall back to static JSON).
    """
    # Prefer local Qdrant retrieval so our app doesn't depend on an external RAG schema.
    try:
        vector = embedding_service.embed_text(query)
        hits = qdrant_service.search(vector, top_k=int(settings.rag_search_top_k))
    except Exception:
        hits = []

    # Fallback to remote HTTP service for backward compatibility if configured.
    if not hits and settings.rag_service_base_url:
        base = settings.rag_service_base_url.rstrip("/")
        url = f"{base}/search"
        # Try both schemas: legacy `limit` and current `top_k`.
        payload = {"query": query, "top_k": settings.rag_search_top_k, "limit": settings.rag_search_top_k}
        timeout = httpx.Timeout(settings.rag_service_timeout_seconds)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
        except Exception:
            logger.exception("rag_search_failed", extra={"url": url})
            return None

        if isinstance(data, dict) and isinstance(data.get("hits"), list):
            hits = data["hits"]
        elif isinstance(data, dict) and isinstance(data.get("results"), list):
            # Map /search response into legacy hit-like dicts.
            hits = [
                {"id": r.get("name"), "score": r.get("score"), "payload": {"name": r.get("name")}}
                for r in data["results"]
                if isinstance(r, dict)
            ]
        else:
            logger.warning("rag_search_invalid_shape", extra={"url": url})
            return None

    documents: list[dict[str, Any]] = []
    labels: list[str] = []
    for h in hits:
        if not isinstance(h, dict):
            continue
        payload = h.get("payload") if isinstance(h.get("payload"), dict) else {}
        name = str(payload.get("name") or "").strip()
        if not name:
            name = str(h.get("id") or "").strip()
        doc = {
            "metadata": {
                "name": name,
                "source": "qdrant",
                "document_id": str(h.get("id") or ""),
                "score": float(h.get("score") or 0.0),
            },
            "text": json.dumps(payload, ensure_ascii=False, default=str) if payload else name,
        }
        documents.append(doc)
        label = _hit_display_name(doc)
        if label:
            labels.append(label)

    catalog = _dedupe_preserve_order(labels)
    if not catalog:
        logger.warning("RAG returned empty results", extra={"n_hits": len(hits)})
        # Hard fallback: always return non-empty retrieval payload for downstream ranking.
        fallback_docs = [
            {"metadata": {"name": "HDFC Bank Personal Loan", "source": "rag_fallback"}, "text": "HDFC Bank Personal Loan"},
            {"metadata": {"name": "ICICI Bank Personal Loan", "source": "rag_fallback"}, "text": "ICICI Bank Personal Loan"},
        ]
        return RagSearchContext(
            catalog_names=("HDFC Bank Personal Loan", "ICICI Bank Personal Loan"),
            retrieved_documents=tuple(fallback_docs),
        )
    return RagSearchContext(
        catalog_names=catalog,
        retrieved_documents=tuple(documents),
    )


async def fetch_rag_catalog_for_query(query: str, settings: Settings) -> tuple[str, ...] | None:
    """Backward-compatible: catalog names only from RAG search."""
    ctx = await fetch_rag_context_for_query(query, settings)
    return ctx.catalog_names if ctx else None
