"""Compare rankings from multiple LLM providers (overlap, rank spread, stability)."""

from __future__ import annotations

import statistics
from typing import Any


def _normalize_product_name(name: str) -> str:
    return " ".join(str(name).strip().lower().split())


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


def _product_items(payload: Any) -> list[Any]:
    if payload is None:
        return []
    if isinstance(payload, dict):
        raw = payload.get("ranked_products")
        if isinstance(raw, list):
            return raw
        return []
    if isinstance(payload, list):
        return payload
    return []


def _name_from_row(row: Any) -> str:
    if not isinstance(row, dict):
        return ""
    name_val = row.get("name")
    if name_val is None:
        name_val = row.get("product_name")
    return str(name_val).strip() if name_val is not None else ""


def _rank_map_for_provider(payload: Any) -> dict[str, int]:
    """
    Map normalized product name -> best (minimum) rank for that provider.
    """
    items = _product_items(payload)
    best: dict[str, int] = {}
    for idx, item in enumerate(items):
        if isinstance(item, str):
            key = _normalize_product_name(item)
            rank = idx + 1
        else:
            name = _name_from_row(item)
            if not name:
                continue
            key = _normalize_product_name(name)
            rank = _coerce_rank(item.get("rank") if isinstance(item, dict) else None, idx + 1)
        if key not in best or rank < best[key]:
            best[key] = rank
    return best


def compare_rankings(results: dict[str, Any]) -> dict[str, float]:
    """
    Compare provider rankings for the same underlying query.

    ``results`` values are each either a list of ranked rows (``name`` / ``product_name``,
    optional ``rank``) or a dict containing ``ranked_products`` in that shape.

    Returns:
        - ``overlap_score``: |∩ provider name sets| / |∪| among providers that returned
          at least one name (1.0 if there is nothing to union).
        - ``rank_variance``: mean sample variance of rank across providers, per product,
          taken only over products that appear in at least two providers' lists.
        - ``stability_score``: ``overlap_score * (1 / (1 + rank_variance))``, clipped to [0, 1].
    """
    order = ("ollama", "openai", "openrouter")
    keys: list[str] = [k for k in order if k in results]
    for k in sorted(results.keys()):
        if k not in keys:
            keys.append(k)

    rank_maps = [_rank_map_for_provider(results[k]) for k in keys]
    non_empty = [m for m in rank_maps if m]

    if not non_empty:
        return {"overlap_score": 1.0, "rank_variance": 0.0, "stability_score": 1.0}

    union: set[str] = set()
    for m in non_empty:
        union |= set(m.keys())

    intersection = set(non_empty[0].keys())
    for m in non_empty[1:]:
        intersection &= set(m.keys())

    if not union:
        overlap_score = 1.0
    else:
        overlap_score = len(intersection) / len(union)

    variances: list[float] = []
    for name in union:
        ranks = [m[name] for m in rank_maps if name in m]
        if len(ranks) >= 2:
            variances.append(float(statistics.variance(ranks)))

    rank_variance = float(statistics.mean(variances)) if variances else 0.0

    stability_raw = overlap_score * (1.0 / (1.0 + rank_variance))
    stability_score = max(0.0, min(1.0, stability_raw))

    return {
        "overlap_score": float(overlap_score),
        "rank_variance": float(rank_variance),
        "stability_score": float(stability_score),
    }
