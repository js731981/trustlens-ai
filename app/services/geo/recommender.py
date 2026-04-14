from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Dict, List, Tuple


def _norm_str(x: Any) -> str:
    s = "" if x is None else str(x)
    return " ".join(s.strip().split())


def _dedupe_keep_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in items:
        s = _norm_str(raw)
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _as_query_rankings(rankings: Any) -> list[tuple[str, list[str]]]:
    """
    Normalizes various possible shapes into: [(query, [ranked_items...]), ...]

    Accepted shapes:
    - {"query": ["A", "B"]} (dict mapping)
    - [{"query": "...", "ranking": [...]}, ...]
    - ["A", "B", ...] (single ranking without query)
    """
    if rankings is None:
        return []

    if isinstance(rankings, dict):
        out: list[tuple[str, list[str]]] = []
        for q, r in rankings.items():
            qn = _norm_str(q) or "query"
            if isinstance(r, list):
                out.append((qn, _dedupe_keep_order(r)))
            else:
                out.append((qn, _dedupe_keep_order([r])))
        return out

    if isinstance(rankings, list):
        if not rankings:
            return []
        if all(isinstance(x, str) or x is None for x in rankings):
            return [("query", _dedupe_keep_order(rankings))]

        out2: list[tuple[str, list[str]]] = []
        for item in rankings:
            if isinstance(item, dict):
                qn = _norm_str(item.get("query")) or "query"
                r = item.get("ranking", item.get("rankings", []))
                if isinstance(r, list):
                    out2.append((qn, _dedupe_keep_order(r)))
                else:
                    out2.append((qn, _dedupe_keep_order([r])))
        return out2

    return [("query", _dedupe_keep_order([rankings]))]


def _as_query_list(value: Any) -> dict[str, list[str]]:
    """
    Normalizes possible shapes into {query: [items...]}.

    Accepted shapes:
    - ["A", "B"] (applies to "query")
    - {"query": ["A", "B"]}
    - [{"query": "...", "items": [...]}, ...]
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        out: dict[str, list[str]] = {}
        for q, v in value.items():
            qn = _norm_str(q) or "query"
            if isinstance(v, list):
                out[qn] = _dedupe_keep_order(v)
            else:
                out[qn] = _dedupe_keep_order([v])
        return out
    if isinstance(value, list):
        if not value:
            return {}
        if all(isinstance(x, str) or x is None for x in value):
            return {"query": _dedupe_keep_order(value)}
        out2: dict[str, list[str]] = {}
        for item in value:
            if not isinstance(item, dict):
                continue
            qn = _norm_str(item.get("query")) or "query"
            items = item.get("items", item.get("products", item.get("missing", [])))
            if isinstance(items, list):
                out2[qn] = _dedupe_keep_order(items)
            else:
                out2[qn] = _dedupe_keep_order([items])
        return out2
    return {"query": _dedupe_keep_order([value])}


def _query_flags(query: str) -> dict[str, bool]:
    q = _norm_str(query).lower()
    return {
        "family": "family" in q,
        "health": ("health" in q) or ("insurance" in q),
        "loan": "loan" in q,
        "keyword_health": "health" in q,
        "keyword_insurance": "insurance" in q,
    }


def _rank_index(ranking: list[str], product: str) -> int | None:
    """0-based index if present, else None (case-insensitive match)."""
    target = _norm_str(product).casefold()
    for i, item in enumerate(ranking):
        if _norm_str(item).casefold() == target:
            return i
    return None


def generate_geo_recommendations(data: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    GEO optimisation recommendations.

    Inputs on `data`:
      - rankings
      - missing_products
      - ground_truth

    Output:
      {
        "issues": [...],
        "recommendations": [...]
      }
    """
    src = data if isinstance(data, dict) else {}
    rankings = _as_query_rankings(src.get("rankings"))
    missing_by_q = _as_query_list(src.get("missing_products"))
    gt_by_q = _as_query_list(src.get("ground_truth"))

    issues: list[str] = []
    recs: list[str] = []

    # Heuristics
    LOW_RANK_THRESHOLD = 3  # ranks 4+ are considered "low visibility"

    for query, ranking in rankings:
        flags = _query_flags(query)
        gt = gt_by_q.get(query) or gt_by_q.get("query") or []
        missing = missing_by_q.get(query) or missing_by_q.get("query") or []

        if missing:
            if flags["family"]:
                issues.append("Low visibility in family queries")
            else:
                issues.append("Missing product visibility in key queries")

            if flags["health"] and flags["family"]:
                recs.append("Improve content around family insurance")
            elif flags["health"]:
                recs.append("Improve coverage and benefits messaging for your insurance products")
            elif flags["loan"]:
                recs.append("Improve landing pages and FAQs for personal loan intent queries")
            else:
                recs.append("Increase coverage across high-intent queries where products are missing")

        # Low rank signals (only if we have a ground-truth set to compare against)
        if gt and ranking:
            worst_idx: int | None = None
            for p in gt:
                idx = _rank_index(ranking, p)
                if idx is None:
                    continue
                worst_idx = idx if worst_idx is None else max(worst_idx, idx)

            if worst_idx is not None and worst_idx >= LOW_RANK_THRESHOLD:
                if flags["family"]:
                    issues.append("Low visibility in family queries")
                    recs.append("Strengthen family-focused keyword coverage and comparison content")
                elif flags["health"]:
                    issues.append("Low visibility for health-related queries")
                    recs.append("Increase presence in health-related keywords")
                elif flags["loan"]:
                    issues.append("Low visibility for loan-related queries")
                    recs.append("Target loan-related keywords with clearer eligibility and rate content")
                else:
                    issues.append("Low visibility in competitive queries")
                    recs.append("Improve relevance signals (titles, headings, structured data) for priority queries")

    return {
        "issues": _dedupe_keep_order(issues),
        "recommendations": _dedupe_keep_order(recs),
    }

