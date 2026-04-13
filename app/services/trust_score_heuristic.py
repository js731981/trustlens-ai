"""Deterministic trust score from the same four signals as `TrustScoreMLP` (see `app/ml/trust_score_model.py`)."""

from __future__ import annotations

from app.models.insights import ExplanationInsights


def _sentiment_to_unit(sentiment: str) -> float:
    s = sentiment.strip().lower()
    if s == "positive":
        return 0.88
    if s == "negative":
        return 0.18
    return 0.52


def compute_trust_score(
    ranking_consistency: float,
    explanation_insights: ExplanationInsights,
) -> float:
    """
    Map signals to [0, 1] using weights aligned with the synthetic training prior in the MLP module.

    - ranking_consistency: 0–1 from repeated-run stability metrics
    - sentiment: discretized to a 0–1 proxy
    - feature_coverage: fraction of price/trust/coverage factors detected in the explanation
    - llm_confidence: sentiment pipeline confidence on the rationale
    """
    sentiment_u = _sentiment_to_unit(explanation_insights.sentiment)
    feature_coverage = len(set(explanation_insights.features) & {"price", "trust", "coverage"}) / 3.0
    llm_confidence = float(explanation_insights.confidence)
    raw = (
        0.35 * ranking_consistency
        + 0.25 * sentiment_u
        + 0.20 * feature_coverage
        + 0.20 * llm_confidence
    )
    return max(0.0, min(1.0, raw))
