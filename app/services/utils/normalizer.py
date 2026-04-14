from __future__ import annotations

from typing import Any

_PROVIDERS: tuple[str, ...] = ("ollama", "openai", "openrouter")


def _coerce_rank(value: Any, fallback: int) -> int:
    if value is None:
        return fallback
    if isinstance(value, bool):
        return fallback
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return fallback
        try:
            return int(float(s))
        except ValueError:
            return fallback
    return fallback


def _string_field(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip() if isinstance(value, str) else str(value)


def _first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _first_nonempty_mapped_name(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        if key not in row:
            continue
        val = row[key]
        if val is None:
            continue
        s = _string_field(val)
        if s:
            return s
    return ""


def normalize_output(data: dict) -> dict[str, Any]:
    """
    Normalize LLM JSON dicts so ``ranked_products`` is always a list of objects
    with ``name``, ``rank`` (int), and ``reason``. Other top-level keys are
    preserved. Never raises.

    Missing ``ranked_products`` becomes ``[]``. Each row maps ``name`` from
    ``name`` or ``service_name``, ``reason`` from ``reason`` or ``reasoning``.
    Invalid rows (e.g. empty name) are dropped. Results are sorted by ``rank``
    ascending and capped at five items.
    """
    src: dict[str, Any] = data if isinstance(data, dict) else {}

    raw = src.get("ranked_products")
    if raw is None or not isinstance(raw, list):
        raw = []

    ranked_products: list[dict[str, Any]] = []
    for idx, item in enumerate(raw):
        row = item if isinstance(item, dict) else {}

        name = _first_nonempty_mapped_name(
            row, ("name", "service_name", "product_name")
        )

        reason_val = _first_present(row, ("reason", "reasoning", "notes"))
        reason = _string_field(reason_val)

        rank = _coerce_rank(row.get("rank"), idx + 1)

        if not name:
            continue

        ranked_products.append(
            {
                "name": name,
                "rank": int(rank),
                "reason": reason,
            }
        )

    ranked_products.sort(key=lambda r: int(r["rank"]))
    ranked_products = ranked_products[:5]

    out: dict[str, Any] = {k: v for k, v in src.items() if k != "ranked_products"}
    out["ranked_products"] = ranked_products
    return out


def _provider_payload(entry: Any) -> dict[str, Any]:
    """Use nested ``parsed_output`` when present (e.g. serialized analyze results); otherwise treat as LLM JSON."""
    if not isinstance(entry, dict):
        return {}
    parsed = entry.get("parsed_output")
    if isinstance(parsed, dict):
        return parsed
    return entry


def normalize_multi_output(results: dict) -> dict[str, dict[str, Any]]:
    """
    For each known provider, ensure ``ranked_products`` exists with ``name``,
    ``rank`` (int), and ``reason``, then sort products by ``rank``.

    Return value always includes ``ollama``, ``openai``, and ``openrouter``.
    Missing or invalid provider entries are normalized to empty rankings.
    """
    src = results if isinstance(results, dict) else {}
    out: dict[str, dict[str, Any]] = {}
    for key in _PROVIDERS:
        normalized = normalize_output(_provider_payload(src.get(key)))
        ranked = normalized.get("ranked_products")
        if not isinstance(ranked, list):
            ranked = []
        ranked_sorted = sorted(
            (row for row in ranked if isinstance(row, dict)),
            key=lambda row: _coerce_rank(row.get("rank"), 0),
        )
        normalized = {**normalized, "ranked_products": ranked_sorted}
        out[key] = normalized
    return out
