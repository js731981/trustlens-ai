"""Human-readable explanations for aggregate trust scores."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from services.trust.trust_scorer import _coerce_float


class TrustExplanation(TypedDict):
    summary: str
    insights: list[str]


def _confidence_level(trust: Any) -> Literal["low", "medium", "high"]:
    raw: Any
    if isinstance(trust, dict):
        raw = trust.get("confidence_level")
    else:
        raw = getattr(trust, "confidence_level", None)
    if raw in ("low", "medium", "high"):
        return raw
    return "medium"


def _trust_score_value(trust: Any) -> float:
    if isinstance(trust, dict):
        return _coerce_float(trust.get("trust_score"))
    return _coerce_float(getattr(trust, "trust_score", None))


def explain_trust(metrics: dict[str, Any], trust: dict[str, Any] | Any) -> TrustExplanation:
    """
    Produce a short narrative summary and bullet insights from comparison metrics.

    Uses ``overlap_score`` (how much product sets agree) and ``rank_variance``
    (mean spread of ranks per shared product). ``trust`` should include at least
    ``trust_score`` and ideally ``confidence_level`` (as returned by
    ``compute_trust_score`` or ``AnalyzeTrustScore``).
    """
    overlap = _coerce_float(metrics.get("overlap_score"))
    rank_variance = _coerce_float(metrics.get("rank_variance"))
    trust_score = _trust_score_value(trust)
    level = _confidence_level(trust)
    if isinstance(trust, dict) and "accuracy_score" in trust:
        accuracy_value = _coerce_float(trust.get("accuracy_score"))
    elif "accuracy_score" in metrics:
        accuracy_value = _coerce_float(metrics.get("accuracy_score"))
    else:
        accuracy_value = None

    overlap_high = overlap >= 0.65
    overlap_low = overlap < 0.35
    variance_low = rank_variance < 0.5
    variance_high = rank_variance > 1.5

    if overlap_high and variance_low:
        summary_lead = (
            "High agreement across models indicates reliable recommendations: "
            "they surface similar products and largely agree on their ordering."
        )
        overlap_insight = (
            f"Overlap is strong ({overlap:.0%} of the combined product set appears "
            "in every model), so recommendations are anchored on a shared shortlist."
        )
        variance_insight = (
            "Mean rank variance across models is low, so orderings reinforce each other "
            "rather than pulling in different directions."
        )
    elif overlap_high and not variance_low:
        summary_lead = (
            "Models agree on which products matter, but ranks diverge—overlap is high "
            "while rank variance is elevated, so treat top positions as less stable."
        )
        overlap_insight = (
            f"High overlap ({overlap:.0%}) shows the models are looking at a similar basket "
            "of candidates even when they reshuffle priorities."
        )
        variance_insight = (
            f"Higher mean rank variance ({rank_variance:.2f}) means the same products "
            "can appear in very different slots depending on the model."
        )
    elif overlap_low:
        summary_lead = (
            "Low overlap across models suggests each surface different products; "
            "consolidated trust is limited until you reconcile those differences."
        )
        overlap_insight = (
            f"Overlap is modest ({overlap:.0%}), so the union of recommendations is wider "
            "than what any single model alone would emphasize."
        )
        if variance_high:
            variance_insight = (
                f"Rank variance is also high ({rank_variance:.2f}), so even shared products "
                "are ordered inconsistently."
            )
        elif variance_low:
            variance_insight = (
                "Where products do overlap, ranks are fairly consistent—disagreement is "
                "mostly about coverage, not ordering."
            )
        else:
            variance_insight = (
                f"Mean rank variance is moderate ({rank_variance:.2f}), mixing some ordering "
                "agreement with meaningful spread."
            )
    else:
        summary_lead = (
            "Agreement across models is mixed: overlap and rank spread sit in the middle, "
            "so recommendations are usable but not strongly convergent."
        )
        overlap_insight = (
            f"Overlap is moderate ({overlap:.0%}), indicating partial—not unanimous—"
            "agreement on which products belong in scope."
        )
        variance_insight = (
            f"Mean rank variance is {rank_variance:.2f}, reflecting a typical spread in how "
            "models order overlapping picks."
        )

    if level == "high":
        summary_close = (
            f" The aggregate trust score ({trust_score:.2f}) maps to high confidence given "
            "these signals."
        )
    elif level == "medium":
        summary_close = (
            f" The aggregate trust score ({trust_score:.2f}) indicates medium confidence—"
            "use the list as guidance and sanity-check top choices."
        )
    else:
        summary_close = (
            f" The aggregate trust score ({trust_score:.2f}) is in the low-confidence band; "
            "prioritize independent validation before relying on ordering."
        )

    summary = summary_lead + summary_close
    insights = [overlap_insight, variance_insight]
    if accuracy_value is not None:
        if accuracy_value >= 0.75:
            acc_insight = (
                f"Catalog alignment is strong ({accuracy_value:.0%} of the catalog sample is reflected "
                "in model rankings), which supports factual grounding."
            )
        elif accuracy_value >= 0.4:
            acc_insight = (
                f"Catalog alignment is moderate ({accuracy_value:.0%}); some catalog references may be "
                "missing from or reshuffled across model outputs."
            )
        else:
            acc_insight = (
                f"Catalog alignment is weak ({accuracy_value:.0%}), so names in rankings may drift from "
                "the supplied product list—verify picks against the catalog."
            )
        insights.append(acc_insight)

    return {"summary": summary, "insights": insights}
