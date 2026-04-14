from __future__ import annotations

import statistics
from typing import Any

from app.services import analyze as analyze_service


def _norm_name(name: Any) -> str:
    s = str(name or "").strip()
    return " ".join(s.split())


def _display_name(raw: str) -> str:
    s = _norm_name(raw)
    if not s:
        return ""
    compact = s.replace(" ", "")
    if compact.isalpha() and len(compact) <= 6:
        return compact.upper()
    parts = [p for p in s.split(" ") if p]
    return " ".join(p[:1].upper() + p[1:].lower() if p else "" for p in parts).strip()


def _rank_items(payload: Any) -> list[Any]:
    if isinstance(payload, dict):
        raw = payload.get("ranked_companies")
        if isinstance(raw, list):
            return raw
        raw = payload.get("ranked_products")
        if isinstance(raw, list):
            return raw
        return []
    if isinstance(payload, list):
        return payload
    return []


def _coerce_rank(value: Any, fallback: int) -> int:
    if value is None or isinstance(value, bool):
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


def _name_from_row(row: Any) -> str:
    if isinstance(row, str):
        return _norm_name(row)
    if not isinstance(row, dict):
        return ""
    for key in ("name", "company", "provider", "brand", "product_name"):
        if key in row and row[key] is not None:
            s = _norm_name(row[key])
            if s:
                return s
    return ""


def _rank_map(payload: Any) -> dict[str, tuple[int, str]]:
    """
    Map normalized company name -> best (minimum) rank for that payload.
    Accepts both:
      - {"ranked_companies": [{"name": "...", "rank": 1}, ...]}
      - {"ranked_products": [{"name": "...", "rank": 1}, ...]} (fallback)
      - ["A", "B", "C"]
    """
    best: dict[str, tuple[int, str]] = {}
    for idx, row in enumerate(_rank_items(payload)):
        raw_name = _name_from_row(row)
        if not raw_name:
            continue
        display = _display_name(raw_name)
        if isinstance(row, dict):
            rank = _coerce_rank(row.get("rank"), idx + 1)
        else:
            rank = idx + 1
        key = raw_name.lower()
        if key not in best or rank < best[key][0]:
            best[key] = (int(rank), display)
    return best


def _aggregate_rank_maps(rank_maps: list[dict[str, tuple[int, str]]]) -> list[tuple[str, int, str]]:
    """
    Aggregate per-provider ranks into a single consensus ranking using median rank
    (ties broken by mean rank then name).
    """
    keys: set[str] = set()
    for m in rank_maps:
        keys |= set(m.keys())
    if not keys:
        return []

    rows: list[tuple[str, float, float]] = []
    for key in keys:
        ranks = [m[key][0] for m in rank_maps if key in m]
        med = float(statistics.median(ranks))
        mean = float(statistics.mean(ranks))
        rows.append((key, med, mean))

    rows.sort(key=lambda r: (r[1], r[2], r[0]))
    out: list[tuple[str, int, str]] = []
    for i, (key, _, __) in enumerate(rows):
        display = ""
        for m in rank_maps:
            if key in m and m[key][1]:
                display = m[key][1]
                break
        out.append((key, i + 1, display or _display_name(key)))
    return out


def _competitor_prompt(query: str, company: str, top_k: int) -> str:
    q = (query or "").strip()
    c = _norm_name(company)
    k = max(2, min(10, int(top_k)))
    return "\n".join(
        [
            "You are ranking competing financial companies for a user query.",
            "Return STRICT JSON ONLY. No markdown, no prose.",
            "",
            "Input:",
            f'- query: "{q}"',
            f'- company: "{c}"',
            "",
            "Output JSON schema (exact keys):",
            "{",
            '  "ranked_companies": [',
            '    {"name": "Company", "rank": 1}',
            "  ]",
            "}",
            "",
            f"Rules:",
            f"- Include the user's company ({c}) somewhere in the list.",
            f"- Provide exactly {k} companies total (company + competitors).",
            "- Ranks must be 1..N without gaps.",
            "- Names should be short brand names (e.g., HDFC, ICICI, Star).",
        ]
    ).strip()


async def competitor_comparison(
    *,
    query: str,
    company: str,
    provider: str = "all",
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Returns:
      {
        "your_rank": int,
        "competitors": [{"name": str, "rank": int}, ...]
      }
    """
    prompt = _competitor_prompt(query=query, company=company, top_k=top_k)
    result = await analyze_service.run_analyze(
        query,
        provider=provider,
        prompt_override=prompt,
    )

    payloads: list[dict[str, Any]] = []
    if isinstance(result, dict) is False and hasattr(result, "results"):
        # AnalyzeComparisonResponse
        for entry in (getattr(result, "results", {}) or {}).values():
            parsed = getattr(entry, "parsed_output", {}) if entry is not None else {}
            payloads.append(parsed if isinstance(parsed, dict) else {})
    else:
        parsed = getattr(result, "parsed_output", {}) if result is not None else {}
        payloads.append(parsed if isinstance(parsed, dict) else {})

    rank_maps = [_rank_map(p) for p in payloads if p]
    aggregated = _aggregate_rank_maps(rank_maps) if rank_maps else []

    company_key = _norm_name(company).lower()
    your_rank = 0
    competitors: list[dict[str, Any]] = []

    for key, rank, display in aggregated:
        name = display or _display_name(key)
        if key == company_key:
            your_rank = int(rank)
            continue
        competitors.append({"name": name, "rank": int(rank)})

    # Fallback: if the model didn't include the company, derive from the first payload list order.
    if your_rank == 0:
        for m in rank_maps:
            if company_key in m:
                your_rank = int(m[company_key][0])
                break

    return {"your_rank": int(your_rank), "competitors": competitors}

