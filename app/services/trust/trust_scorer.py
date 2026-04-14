"""Compute aggregate trust score from ranking comparison and accuracy metrics."""

from __future__ import annotations

from typing import Any, Literal, TypedDict


def _coerce_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return default
        try:
            return float(s)
        except ValueError:
            return default
    return default


class TrustScoreResult(TypedDict):
    """Aggregate trust and the signals that produced it."""

    trust_score: float
    confidence_level: Literal["low", "medium", "high"]
    stability_score: float
    accuracy_score: float
    overlap_score: float
    rank_variance: float
    stability_weight: float
    accuracy_weight: float
    overlap_weight: float
    stability_component: float
    accuracy_component: float
    overlap_component: float


def compute_trust_score(comparison_metrics: dict[str, Any]) -> TrustScoreResult:
    """
    Blend ranking stability, catalog alignment accuracy, and set overlap.

    ``comparison_metrics`` should include ``overlap_score``, ``rank_variance``,
    ``stability_score`` (from ``compare_rankings``), and ``accuracy_score`` in
    ``[0, 1]`` (e.g. catalog alignment from ``accuracy_score_vs_catalog``).
    Missing numeric values are coerced to ``0.0`` except ``accuracy_score``,
    which defaults to ``1.0`` when absent so legacy callers that only pass
    ranking metrics are not over-penalized.

    Final (components sum to this before clamp):
        trust_score =
            (0.4 * stability_score) +
            (0.4 * accuracy_score) +
            (0.2 * overlap_score)

    ``rank_variance`` is echoed from inputs for explanations; it does not enter
    the aggregate formula.

    The result is clamped to ``[0, 1]``.

    ``confidence_level``:
        - high: score >= 0.7
        - medium: 0.4 <= score < 0.7
        - low: score < 0.4
    """
    overlap = _coerce_float(comparison_metrics.get("overlap_score"))
    rank_variance = _coerce_float(comparison_metrics.get("rank_variance"))
    stability = _coerce_float(comparison_metrics.get("stability_score"))
    if "accuracy_score" not in comparison_metrics:
        accuracy = 1.0
    else:
        accuracy = _coerce_float(comparison_metrics.get("accuracy_score"))

    w_stab, w_acc, w_ovl = 0.4, 0.4, 0.2
    c_stab = w_stab * stability
    c_acc = w_acc * accuracy
    c_ovl = w_ovl * overlap
    raw = c_stab + c_acc + c_ovl
    trust_score = max(0.0, min(1.0, raw))

    if trust_score >= 0.7:
        level: Literal["low", "medium", "high"] = "high"
    elif trust_score >= 0.4:
        level = "medium"
    else:
        level = "low"

    return {
        "trust_score": trust_score,
        "confidence_level": level,
        "stability_score": stability,
        "accuracy_score": accuracy,
        "overlap_score": overlap,
        "rank_variance": rank_variance,
        "stability_weight": w_stab,
        "accuracy_weight": w_acc,
        "overlap_weight": w_ovl,
        "stability_component": c_stab,
        "accuracy_component": c_acc,
        "overlap_component": c_ovl,
    }
