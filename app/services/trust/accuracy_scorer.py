"""Accuracy-based trust scoring against a ground-truth label or product list."""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence, TypedDict

from app.services.trust.ranking_comparator import _name_from_row, _product_items


def _normalize_item(value: Any) -> str:
    if isinstance(value, str):
        return " ".join(value.strip().lower().split())
    return str(value).strip().lower()


class AccuracyResult(TypedDict):
    accuracy: float
    matched: list[Any]
    missed: list[Any]


def compute_accuracy(
    predicted: Sequence[Any],
    ground_truth: Sequence[Any],
) -> AccuracyResult:
    """
    Score how well ``predicted`` covers ``ground_truth``.

    Matching is multiset-aware: each ground-truth entry consumes at most one
    equal prediction (equality uses normalized string form for strings).

    ``accuracy`` is ``matches / len(ground_truth)``. If ``ground_truth`` is
    empty, ``accuracy`` is ``1.0`` and both lists are empty.
    """
    gt = list(ground_truth)
    if not gt:
        return {"accuracy": 1.0, "matched": [], "missed": []}

    pred_counts = Counter(_normalize_item(p) for p in predicted)

    matched: list[Any] = []
    missed: list[Any] = []

    for item in gt:
        key = _normalize_item(item)
        if pred_counts[key] > 0:
            matched.append(item)
            pred_counts[key] -= 1
        else:
            missed.append(item)

    accuracy = len(matched) / len(gt)
    return {"accuracy": float(accuracy), "matched": matched, "missed": missed}


def _ranked_names_in_order(payload: Any) -> list[str]:
    names: list[str] = []
    for item in _product_items(payload):
        if isinstance(item, str):
            s = item.strip()
            if s:
                names.append(s)
        else:
            n = _name_from_row(item)
            if n:
                names.append(n)
    return names


def merged_provider_ranked_names(provider_results: dict[str, Any]) -> list[str]:
    """Concatenate ranked product names from providers in canonical order (for evaluation)."""
    order = ("ollama", "openai", "openrouter")
    keys = [k for k in order if k in provider_results]
    for k in sorted(provider_results.keys()):
        if k not in keys:
            keys.append(k)
    predicted: list[str] = []
    for k in keys:
        predicted.extend(_ranked_names_in_order(provider_results.get(k)))
    return predicted


def accuracy_score_vs_catalog(
    provider_results: dict[str, Any],
    catalog_names: Sequence[str],
    *,
    catalog_ground_truth_cap: int = 200,
) -> float:
    """
    Recall-style alignment: fraction of catalog sample (in order) matched by any
    ranked name from any provider (multiset-aware via ``compute_accuracy``).
    """
    predicted = merged_provider_ranked_names(provider_results)
    cap = catalog_ground_truth_cap if catalog_ground_truth_cap > 0 else len(catalog_names)
    gt = list(catalog_names[:cap])
    return float(compute_accuracy(predicted, gt)["accuracy"])
