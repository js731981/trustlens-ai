from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import Settings
from app.core.logging import get_logger

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
    base = settings.rag_service_base_url.rstrip("/")
    url = f"{base}/search"
    payload = {"query": query, "limit": settings.rag_search_top_k}
    timeout = httpx.Timeout(settings.rag_service_timeout_seconds)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
    except Exception:
        logger.exception("rag_search_failed", extra={"url": url})
        return None

    hits = data.get("hits")
    if not isinstance(hits, list):
        logger.warning("rag_search_invalid_shape", extra={"url": url})
        return None

    documents: list[dict[str, Any]] = []
    labels: list[str] = []
    for h in hits:
        if not isinstance(h, dict):
            continue
        documents.append(h)
        label = _hit_display_name(h)
        if label:
            labels.append(label)

    catalog = _dedupe_preserve_order(labels)
    if not catalog:
        logger.warning("rag_search_empty_catalog", extra={"url": url, "n_hits": len(hits)})
        return None
    return RagSearchContext(
        catalog_names=catalog,
        retrieved_documents=tuple(documents),
    )


async def fetch_rag_catalog_for_query(query: str, settings: Settings) -> tuple[str, ...] | None:
    """Backward-compatible: catalog names only from RAG search."""
    ctx = await fetch_rag_context_for_query(query, settings)
    return ctx.catalog_names if ctx else None
