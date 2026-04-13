"""Heuristic bias detection for LLM product rankings vs dataset ground truth."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from difflib import SequenceMatcher

from app.models.financial import BiasType, RecommendationBiasResult
from app.services.ranking_consistency import normalize_ranking_key

_FUZZY_RATIO = 0.72
_TOP_GROUND_TRUTH = 3
_BRAND_SHARE_SINGLE_RUN = 0.6
_BRAND_SHARE_REPEAT_RANK1 = 0.66
_MIN_REPEAT_RUNS = 3


def _pair_similar(a: str, b: str, *, threshold: float = _FUZZY_RATIO) -> bool:
    na = normalize_ranking_key(a)
    nb = normalize_ranking_key(b)
    if not na or not nb:
        return False
    if na == nb or na in nb or nb in na:
        return True
    return SequenceMatcher(None, na, nb).ratio() >= threshold


def _matches_any(name: str, candidates: Sequence[str], *, threshold: float = _FUZZY_RATIO) -> bool:
    return any(_pair_similar(name, c, threshold=threshold) for c in candidates)


def _infer_brand_token(name: str) -> str:
    parts = normalize_ranking_key(name).split()
    return parts[0] if parts else ""


def _hallucination_bias(llm_names: Sequence[str], catalog: Sequence[str]) -> bool:
    if not catalog or not llm_names:
        return False
    return any(not _matches_any(x, catalog) for x in llm_names if normalize_ranking_key(x))


def _missing_ground_truth_top(
    llm_names: Sequence[str],
    ground_truth: Sequence[str],
    *,
    top_k: int = _TOP_GROUND_TRUTH,
) -> bool:
    if not ground_truth or not llm_names:
        return False
    k = min(top_k, len(ground_truth))
    top_slice = list(ground_truth[:k])
    return any(not _matches_any(g, llm_names) for g in top_slice)


def _brand_bias_single_run(llm_names: Sequence[str]) -> bool:
    if len(llm_names) < 2:
        return False
    tokens = [_infer_brand_token(x) for x in llm_names if _infer_brand_token(x)]
    if not tokens:
        return False
    top_count = Counter(tokens).most_common(1)[0][1]
    return top_count / len(llm_names) >= _BRAND_SHARE_SINGLE_RUN


def _brand_bias_repeat_rank1(rank_ones: Sequence[str]) -> bool:
    if len(rank_ones) < _MIN_REPEAT_RUNS:
        return False
    tokens = [_infer_brand_token(x) for x in rank_ones if _infer_brand_token(x)]
    if not tokens:
        return False
    top_count = Counter(tokens).most_common(1)[0][1]
    return top_count / len(tokens) >= _BRAND_SHARE_REPEAT_RANK1


def detect_recommendation_bias(
    ranked_product_names: Sequence[str],
    ground_truth_product_names: Sequence[str],
    *,
    repeat_run_rank_one_names: Sequence[str] | None = None,
) -> RecommendationBiasResult:
    """
    Compare LLM picks to dataset ordering and optional repeated rank-1 samples.

    Priority when multiple signals fire: hallucination, then brand, then popularity.
    """
    llm = [str(x) for x in ranked_product_names if normalize_ranking_key(str(x))]
    catalog = [str(x) for x in ground_truth_product_names]

    if not llm:
        return RecommendationBiasResult(bias_detected=False, bias_type=None)

    hallucination = _hallucination_bias(llm, catalog) if catalog else False
    brand = _brand_bias_repeat_rank1(repeat_run_rank_one_names or ()) or _brand_bias_single_run(llm)
    popularity = (
        bool(catalog)
        and not hallucination
        and _missing_ground_truth_top(llm, catalog)
    )

    if hallucination:
        return RecommendationBiasResult(bias_detected=True, bias_type="hallucination")
    if brand:
        return RecommendationBiasResult(bias_detected=True, bias_type="brand")
    if popularity:
        return RecommendationBiasResult(bias_detected=True, bias_type="popularity")
    return RecommendationBiasResult(bias_detected=False, bias_type=None)
