from __future__ import annotations

import json
from collections.abc import Iterable
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


def _word_count(text: str) -> int:
    return len(text.split()) if text.strip() else 0


def _escape_query_for_quoted_line(query: str) -> str:
    collapsed = " ".join((query or "").strip().split())
    return collapsed.replace("\\", "\\\\").replace('"', '\\"')


def _build_ranking_prompt(escaped_query: str, option_names: tuple[str, ...]) -> str:
    options_block = _format_available_options(option_names)
    return (
        "---\n\n"
        "Return ONLY valid JSON.\n\n"
        f'Query: "{escaped_query}"\n\n'
        f"{options_block}\n\n"
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
        "- If a product is not in Available Options, DO NOT include it\n"
        "- Always choose best match from the given list only\n\n"
        "Rules:\n"
        "- Select ONLY from available options\n"
        "- Do NOT invent new products\n"
        "- Return exactly 5 items\n"
        "- No markdown\n\n"
        "---"
    )


def build_ranking_prompt(query: str, product_names: Iterable[str]) -> str:
    """Build a JSON-only ranking prompt with ``query`` and a dynamic option list (word-capped)."""
    option_names = _normalize_option_names(product_names)
    cleaned = (query or "").strip()
    words = cleaned.split()
    escaped = _escape_query_for_quoted_line(cleaned)
    body = _build_ranking_prompt(escaped, option_names)
    while _word_count(body) > _MAX_PROMPT_WORDS and words:
        words = words[:-1]
        escaped = _escape_query_for_quoted_line(" ".join(words))
        body = _build_ranking_prompt(escaped, option_names)
    return body
