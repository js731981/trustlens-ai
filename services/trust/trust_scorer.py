"""Compute aggregate trust score from ranking comparison metrics."""

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
    trust_score: float
    confidence_level: Literal["low", "medium", "high"]


def compute_trust_score(comparison_metrics: dict[str, Any]) -> TrustScoreResult:
    """
    Combine overlap, rank spread, and stability into a single trust score.

    ``comparison_metrics`` should include ``overlap_score``, ``rank_variance``,
    and ``stability_score`` (e.g. output of ``compare_rankings``). Missing or
    invalid values are treated as ``0.0``.

    Formula (before normalization):
        0.5 * stability_score + 0.3 * overlap_score - 0.2 * rank_variance

    The result is clamped to the inclusive range ``[0, 1]``.

    ``confidence_level`` is derived from the final ``trust_score``:
        - high: score >= 0.7
        - medium: 0.4 <= score < 0.7
        - low: score < 0.4
    """
    overlap = _coerce_float(comparison_metrics.get("overlap_score"))
    rank_variance = _coerce_float(comparison_metrics.get("rank_variance"))
    stability = _coerce_float(comparison_metrics.get("stability_score"))

    raw = (0.5 * stability) + (0.3 * overlap) - (0.2 * rank_variance)
    trust_score = max(0.0, min(1.0, raw))

    if trust_score >= 0.7:
        level: Literal["low", "medium", "high"] = "high"
    elif trust_score >= 0.4:
        level = "medium"
    else:
        level = "low"

    return {"trust_score": trust_score, "confidence_level": level}
