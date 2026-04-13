"""Repeated ranking queries: variance, position shifts, stability (Kendall's W), consistency."""

from __future__ import annotations

import asyncio
import statistics
from collections.abc import Sequence

from app.models.financial import FinancialQueryResponse, RankedProduct, RankingConsistencyResult


def normalize_ranking_key(name: str) -> str:
    return " ".join(name.strip().lower().split())


def _rank_by_key_for_run(products: Sequence[RankedProduct]) -> dict[str, int]:
    """Minimum rank if duplicate keys appear in one run."""
    best: dict[str, int] = {}
    for p in products:
        k = normalize_ranking_key(p.name)
        r = p.rank
        if k not in best or r < best[k]:
            best[k] = r
    return best


def _union_keys(rank_maps: Sequence[dict[str, int]]) -> list[str]:
    keys: set[str] = set()
    for m in rank_maps:
        keys.update(m.keys())
    return sorted(keys)


def _intersection_keys(rank_maps: Sequence[dict[str, int]]) -> list[str]:
    if not rank_maps:
        return []
    inter = set(rank_maps[0].keys())
    for m in rank_maps[1:]:
        inter &= set(m.keys())
    return sorted(inter)


def _imputed_ranks_for_union(
    rank_maps: Sequence[dict[str, int]],
    run_lengths: Sequence[int],
    union_keys: Sequence[str],
) -> list[list[float]]:
    """One row per item (union order), one column per run: rank with missing => len(run)+1."""
    cols: list[list[float]] = []
    for t, m in enumerate(rank_maps):
        n_t = run_lengths[t]
        miss = float(n_t + 1)
        cols.append([float(m.get(k, miss)) for k in union_keys])
    # transpose to rows=item, cols=run
    m_runs = len(rank_maps)
    if m_runs == 0:
        return []
    return [[cols[t][i] for t in range(m_runs)] for i in range(len(union_keys))]


def _mean_ranking_variance(rank_matrix: list[list[float]]) -> float:
    if not rank_matrix:
        return 0.0
    variances: list[float] = []
    for row in rank_matrix:
        if len(row) < 2:
            variances.append(0.0)
        else:
            variances.append(statistics.variance(row))
    return float(statistics.mean(variances)) if variances else 0.0


def _mean_position_shifts(rank_matrix: list[list[float]]) -> float:
    """Mean over items of mean absolute deviation from that item's mean rank across runs."""
    if not rank_matrix:
        return 0.0
    shifts: list[float] = []
    for row in rank_matrix:
        mu = statistics.mean(row)
        mad = statistics.mean(abs(x - mu) for x in row)
        shifts.append(mad)
    return float(statistics.mean(shifts)) if shifts else 0.0


def kendalls_w(rank_matrix: list[list[float]]) -> float:
    """
    Kendall's coefficient of concordance W.

    rank_matrix[i][j] = rank of object i by rater j (complete ranking 1..n per rater).
    Returns 1.0 when n < 2 (no disagreement possible).
    """
    n = len(rank_matrix)
    if n < 2:
        return 1.0
    m = len(rank_matrix[0])
    if m < 2:
        return 1.0
    sums = [sum(row[j] for j in range(m)) for row in rank_matrix]
    r_mean = m * (n + 1) / 2
    s = sum((ri - r_mean) ** 2 for ri in sums)
    denom = m * m * (n**3 - n)
    if denom <= 0:
        return 1.0
    return max(0.0, min(1.0, (12.0 * s) / denom))


def _intersection_subrank_matrix(
    rank_maps: Sequence[dict[str, int]],
    intersection_keys: Sequence[str],
) -> list[list[float]]:
    """For each run, relative ranks 1..k among intersection items only (by original rank)."""
    k = len(intersection_keys)
    if k == 0:
        return []
    columns: list[list[float]] = []
    keys_tuple = tuple(intersection_keys)
    for m in rank_maps:
        sorted_keys = sorted(keys_tuple, key=lambda key: m[key])
        sub_by_key = {key: float(r) for r, key in enumerate(sorted_keys, start=1)}
        columns.append([sub_by_key[key] for key in keys_tuple])
    m_runs = len(rank_maps)
    return [[columns[t][i] for t in range(m_runs)] for i in range(k)]


def _reference_scales(n_union: int) -> tuple[float, float]:
    """Loose upper scales for normalizing variance and MAD of ranks in 1..n (+ imputed tail)."""
    n_union = max(n_union, 1)
    # Variance of discrete uniform on 1..n
    v_ref = max((n_union**2 - 1) / 12.0, 1e-9)
    # Typical MAD scale for ranks 1..n
    p_ref = max((n_union - 1) / 2.0, 1e-9)
    return v_ref, p_ref


def _consistency_blend(
    stability: float,
    ranking_variance: float,
    position_shifts: float,
    n_union: int,
) -> float:
    v_ref, p_ref = _reference_scales(n_union)
    v_part = 1.0 - min(1.0, ranking_variance / v_ref)
    p_part = 1.0 - min(1.0, position_shifts / p_ref)
    raw = 0.5 * stability + 0.25 * v_part + 0.25 * p_part
    return max(0.0, min(1.0, raw))


def evaluate_ranking_consistency(responses: Sequence[FinancialQueryResponse]) -> RankingConsistencyResult:
    """
    Compute metrics from multiple `FinancialQueryResponse` instances (same underlying query).

    Uses the union of normalized product names across runs; missing names in a run get
    rank len(that_run)+1 for variance / shift calculations. Kendall's W uses only names
    present in every run, with ranks re-computed within that intersection per run.
    """
    if not responses:
        return RankingConsistencyResult(
            ranking_variance=0.0,
            position_shifts=0.0,
            stability_score=0.0,
            consistency_score=0.0,
            n_runs=0,
            n_items_union=0,
            n_items_intersection=0,
        )

    rank_maps = [_rank_by_key_for_run(r.ranked_products) for r in responses]
    run_lengths = [len(r.ranked_products) for r in responses]
    union_keys = _union_keys(rank_maps)
    inter_keys = _intersection_keys(rank_maps)

    union_matrix = _imputed_ranks_for_union(rank_maps, run_lengths, union_keys)
    ranking_variance = _mean_ranking_variance(union_matrix)
    position_shifts = _mean_position_shifts(union_matrix)

    n_union = len(union_keys)
    n_inter = len(inter_keys)

    if n_inter >= 2:
        inter_matrix = _intersection_subrank_matrix(rank_maps, inter_keys)
        stability = kendalls_w(inter_matrix)
    else:
        v_ref, _ = _reference_scales(max(n_union, 1))
        stability = max(0.0, 1.0 - min(1.0, ranking_variance / v_ref))

    consistency = _consistency_blend(stability, ranking_variance, position_shifts, max(n_union, 1))

    return RankingConsistencyResult(
        ranking_variance=ranking_variance,
        position_shifts=position_shifts,
        stability_score=stability,
        consistency_score=consistency,
        n_runs=len(responses),
        n_items_union=n_union,
        n_items_intersection=n_inter,
    )


async def run_financial_query_consistency(
    user_query: str,
    *,
    n_runs: int = 5,
    template_id: str = "financial_ranking",
) -> RankingConsistencyResult:
    """Call the financial LLM `n_runs` times with the same query and aggregate consistency metrics."""
    from app.services.financial_llm import query_financial_llm

    if n_runs < 1:
        raise ValueError("n_runs must be at least 1.")
    responses = await asyncio.gather(
        *[query_financial_llm(user_query, template_id=template_id) for _ in range(n_runs)],
    )
    return evaluate_ranking_consistency(list(responses))
