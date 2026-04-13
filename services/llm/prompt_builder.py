from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

_MAX_PROMPT_WORDS = 500

_PRODUCTS_PATH = Path(__file__).resolve().parents[2] / "data" / "insurance_products.json"


def _names_from_ranking_json(data: Any) -> tuple[str, ...]:
    if not isinstance(data, list):
        return ()
    names: list[str] = []
    for item in data:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            n = item["name"].strip()
            if n:
                names.append(n)
    return tuple(names)


def load_catalog_product_names(json_path: Path) -> tuple[str, ...]:
    """Load ``name`` fields from a ranking catalog JSON file (list of objects)."""
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)
    return _names_from_ranking_json(data)


@lru_cache(maxsize=1)
def _default_catalog_product_names() -> tuple[str, ...]:
    with _PRODUCTS_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    return _names_from_ranking_json(data)


def _normalize_option_names(product_names: Iterable[str]) -> tuple[str, ...]:
    return tuple(str(n).strip() for n in product_names if str(n).strip())


def _format_available_options(names: tuple[str, ...]) -> str:
    lines = "\n".join(f"- {name}" for name in names)
    return f"Available Options:\n{lines}" if lines else "Available Options:\n"


def insurance_catalog_product_names() -> tuple[str, ...]:
    """Default catalog ``name`` values from ``data/insurance_products.json``."""
    return _default_catalog_product_names()


def format_available_products_block(product_names: Iterable[str] | None = None) -> str:
    """One option per line under ``Available Options:`` for LLM prompts."""
    if product_names is None:
        names = _default_catalog_product_names()
    else:
        names = _normalize_option_names(product_names)
    return _format_available_options(names)


def format_context_product_bullets(product_names: Iterable[str] | None = None) -> str:
    """Bullet lines only (no header) for ``Context:`` blocks in ranking prompts."""
    if product_names is None:
        names = _default_catalog_product_names()
    else:
        names = _normalize_option_names(tuple(product_names))
    return "\n".join(f"- {name}" for name in names) if names else ""


def documents_from_product_names(product_names: Iterable[str]) -> list[dict[str, Any]]:
    """Minimal retrieval-shaped documents when only catalog names are available."""
    out: list[dict[str, Any]] = []
    for n in _normalize_option_names(product_names):
        out.append({"metadata": {"name": n}, "text": n})
    return out


def _label_for_retrieved_document(doc: dict[str, Any]) -> str:
    meta = doc.get("metadata")
    if isinstance(meta, dict):
        for key in ("name", "title", "product"):
            v = meta.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
    text = (doc.get("text") or "").strip()
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


def format_top_results_block(
    retrieved_documents: Iterable[dict[str, Any]],
    *,
    max_items: int | None = None,
    max_text_chars: int = 480,
) -> str:
    """
    Render retrieved hits (or synthetic name-only docs) for the ``Context:`` section.

    Each item is numbered with a display label and an optional text snippet.
    """
    docs = [d for d in retrieved_documents if isinstance(d, dict)]
    if max_items is not None:
        docs = docs[: max(0, max_items)]
    lines: list[str] = []
    for i, doc in enumerate(docs, start=1):
        label = _label_for_retrieved_document(doc)
        head = f"{i}. {label}" if label else f"{i}."
        lines.append(head)
        raw = (doc.get("text") or "").strip()
        if raw:
            body = raw if len(raw) <= max_text_chars else f"{raw[: max_text_chars - 1]}…"
            lines.append(f"   {body}")
    return "\n".join(lines) if lines else "(no context items)"


def _word_count(text: str) -> int:
    return len(text.split()) if text.strip() else 0


def _escape_query_for_quoted_line(query: str) -> str:
    collapsed = " ".join((query or "").strip().split())
    return collapsed.replace("\\", "\\\\").replace('"', '\\"')


def _build_ranking_prompt(escaped_query: str, top_results_block: str) -> str:
    return (
        "---\n\n"
        "Context:\n"
        f"{top_results_block}\n\n"
        f'Query: "{escaped_query}"\n\n'
        "Instructions:\n"
        "- Use ONLY context items\n"
        "- Rank top 5\n"
        "- Return ONLY valid JSON (no markdown fences)\n\n"
        "Output format:\n"
        "{\n"
        '  "ranked_products": [\n'
        "    {\n"
        '      "name": "string",\n'
        '      "rank": 1,\n'
        '      "reason": "string"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "STRICT:\n"
        "- Each \"name\" must match a context item (verbatim where a clear label is shown)\n"
        "- Do NOT invent items outside the context list\n"
        "- Return exactly 5 items\n\n"
        "---"
    )


def build_ranking_prompt_from_retrieval(
    query: str,
    retrieved_documents: Sequence[dict[str, Any]],
    *,
    max_items: int | None = None,
) -> str:
    """Ranking prompt from ``query`` and retrieval payloads (RAG hits or synthetic dicts)."""
    top_results_block = format_top_results_block(retrieved_documents, max_items=max_items)
    cleaned = (query or "").strip()
    words = cleaned.split()
    escaped = _escape_query_for_quoted_line(cleaned)
    body = _build_ranking_prompt(escaped, top_results_block)
    while _word_count(body) > _MAX_PROMPT_WORDS and words:
        words = words[:-1]
        escaped = _escape_query_for_quoted_line(" ".join(words))
        body = _build_ranking_prompt(escaped, top_results_block)
    return body


def build_ranking_prompt(query: str, product_names: Iterable[str]) -> str:
    """Build a JSON-only ranking prompt using catalog names as minimal context rows (word-capped)."""
    docs = documents_from_product_names(product_names)
    return build_ranking_prompt_from_retrieval(query, docs)
