from __future__ import annotations

import hashlib
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple


KNOWN_BRANDS: set[str] = {
    "HDFC",
    "HDFC Ergo",
    "HDFC Bank",
    "ICICI",
    "ICICI Lombard",
    "SBI",
    "Star Health",
    "Bajaj Finserv",
    "Axis Bank",
    "Kotak",
}

INSURANCE_FAMILY: List[str] = [
    "HDFC Ergo Family Floater Health Insurance",
    "Star Health Family Health Optima",
    "ICICI Lombard Complete Health Insurance",
    "SBI General Arogya Supreme (Family)",
    "Bajaj Finserv Health Guard Family Plan",
]

INSURANCE_BUDGET: List[str] = [
    "Star Health Smart Health Pro (Budget)",
    "HDFC Ergo Optima Secure (Value)",
    "ICICI Lombard Health Booster (Entry)",
    "SBI General Hospital Cash (Add-on)",
]

LOAN_FAMILY: List[str] = [
    "SBI Personal Loan (Flexible EMI)",
    "HDFC Bank Personal Loan (Quick Disbursal)",
    "ICICI Personal Loan (Digital Journey)",
    "Axis Bank Personal Loan (Pre-approved)",
    "Kotak Personal Loan (Same-day)",
]

LOAN_BUDGET: List[str] = [
    "SBI Personal Loan (Low-rate segment)",
    "ICICI Personal Loan (Offers for salaried)",
    "HDFC Bank Personal Loan (Preferred customers)",
    "Bajaj Finserv Personal Loan (Quick approval)",
]

GENERIC_OPTIONS: List[str] = [
    "Trusted Health Cover Plan",
    "Affordable Family Protection Plan",
    "Instant Personal Loan Plan",
    "Budget-Friendly Insurance Plan",
    "Comprehensive Coverage Plan",
]

TRACE_STEP_LABELS: List[str] = [
    "🔍 Retrieval agent fetching candidates...",
    "📊 Ranking agent sorting products...",
    "🛡 Trust agent computing score...",
    "🌍 GEO agent analysing relevance...",
    "💡 Explanation agent generating reasoning...",
]

AGENT_KEYS: List[str] = ["retrieval", "ranking", "trust", "geo", "explanation"]


def _stable_random(query: str, extra: str = "") -> random.Random:
    digest = hashlib.md5(f"{query.strip().lower()}\n{extra}".encode("utf-8")).hexdigest()
    seed = int(digest[:8], 16)
    return random.Random(seed)


def _normalize_query(query: str) -> str:
    return " ".join((query or "").strip().lower().split())


def _query_has_brand(q: str) -> bool:
    ql = (q or "").lower()
    return any(b.lower() in ql for b in KNOWN_BRANDS)


def _query_is_generic(ql: str, flags: Dict[str, bool]) -> bool:
    if len(ql.split()) <= 2:
        return True
    if not (flags.get("insurance") or flags.get("loan")):
        return True
    vague = ("best" in ql or "good" in ql or "compare" in ql) and len(ql) < 28
    return vague


def _clamp_int(x: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(x))))


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _pick_domain_candidates(q: str, rng: random.Random) -> Tuple[str, List[str], Dict[str, bool]]:
    ql = _normalize_query(q)
    is_insurance = ("insurance" in ql) or ("health" in ql)
    is_loan = "loan" in ql
    wants_family = "family" in ql
    wants_cheap = ("cheap" in ql) or ("affordable" in ql) or ("budget" in ql) or ("low cost" in ql)

    flags: Dict[str, bool] = {
        "insurance": is_insurance,
        "loan": is_loan,
        "family": wants_family,
        "cheap": wants_cheap,
    }

    if is_insurance and wants_family:
        pool = INSURANCE_FAMILY[:]
        domain = "health insurance"
    elif is_insurance and wants_cheap:
        pool = INSURANCE_BUDGET[:]
        domain = "health insurance"
    elif is_insurance:
        pool = [
            "HDFC Ergo Health Insurance",
            "ICICI Lombard Health Insurance",
            "Star Health Insurance",
            "SBI General Health Insurance",
            "Bajaj Finserv Health Insurance",
        ]
        domain = "health insurance"
    elif is_loan and wants_family:
        pool = LOAN_FAMILY[:]
        domain = "personal loan"
    elif is_loan and wants_cheap:
        pool = LOAN_BUDGET[:]
        domain = "personal loan"
    elif is_loan:
        pool = [
            "HDFC Bank Personal Loan",
            "ICICI Personal Loan",
            "SBI Personal Loan",
            "Axis Bank Personal Loan",
            "Kotak Personal Loan",
        ]
        domain = "personal loan"
    else:
        pool = [
            "HDFC Ergo Health Insurance",
            "ICICI Lombard Health Insurance",
            "Star Health Insurance",
            "HDFC Bank Personal Loan",
            "SBI Personal Loan",
            "ICICI Personal Loan",
        ] + GENERIC_OPTIONS
        rng.shuffle(pool)
        domain = "financial products"

    n = rng.randint(3, min(5, len(pool)))
    candidates = pool[:n]
    if len(candidates) >= 3 and rng.random() < 0.35:
        candidates[1], candidates[2] = candidates[2], candidates[1]
    return domain, candidates, flags


def _brandiness_score(names: List[str]) -> float:
    if not names:
        return 0.0
    branded = sum(1 for r in names if r and any(b.lower() in r.lower() for b in KNOWN_BRANDS))
    return branded / max(1, len(names))


def _specificity_score(names: List[str]) -> float:
    if not names:
        return 0.0
    vals: list[float] = []
    for r in names:
        s = " ".join((r or "").strip().split())
        if not s:
            continue
        tok = len(s.split())
        has_brand = 1.0 if any(b.lower() in s.lower() for b in KNOWN_BRANDS) else 0.0
        vals.append(min(1.0, (tok / 6.0) * 0.55 + has_brand * 0.45))
    return sum(vals) / max(1, len(vals))


def _pick_failed_agent_index(simulate_failure: bool, rng: random.Random) -> int | None:
    if not simulate_failure:
        return None
    return rng.randint(0, len(AGENT_KEYS) - 1)


def simulate_agents(query: str, simulate_failure: bool = False) -> Dict[str, Any]:
    """
    Simulated CrewAI-style multi-agent pipeline. No external APIs.

    Returns scores in 0..1 for UI bars, integers 0..100 for trust_score_int/geo/confidence,
    trace as list of dicts for HTML renderer, plus breakdowns and debug payload.
    """
    q = (query or "").strip()
    rng = _stable_random(q, f"fail={simulate_failure}")
    ql = _normalize_query(q)

    domain, retrieval, flags = _pick_domain_candidates(q, rng)
    ranking = list(retrieval)

    has_best = "best" in ql
    has_brand_in_query = _query_has_brand(q)
    generic_q = _query_is_generic(ql, flags)

    brandiness = _brandiness_score(ranking)
    specificity = _specificity_score(ranking)

    trust_int = _clamp_int(72 + 18 * brandiness + 8 * specificity + (6 if has_best else 0) + (8 if has_brand_in_query else 0), 60, 95)
    geo_int = _clamp_int(68 + 16 * specificity + 10 * brandiness - (10 if generic_q else 0) - (4 if flags.get("cheap") else 0), 50, 90)

    low_results = len(ranking) < 3
    if low_results:
        trust_int = _clamp_int(trust_int - 8, 60, 95)
        geo_int = _clamp_int(geo_int - 10, 50, 90)

    is_fallback_domain = domain == "financial products" and brandiness < 0.34
    used_fallback = bool(is_fallback_domain)

    fail_idx = _pick_failed_agent_index(simulate_failure, rng)
    agent_failed = fail_idx is not None
    failed_key = AGENT_KEYS[fail_idx] if fail_idx is not None else None

    if agent_failed:
        if failed_key == "trust":
            trust_int = _clamp_int(trust_int - 12, 60, 95)
        elif failed_key == "geo":
            geo_int = _clamp_int(geo_int - 14, 50, 90)
        elif failed_key == "retrieval" and len(ranking) > 3:
            ranking = ranking[: max(3, len(ranking) - 1)]
        used_fallback = True

    trust_f = round(trust_int / 100.0, 2)
    geo_f = round(geo_int / 100.0, 2)

    brand_pts = _clamp_int(12 + 22 * brandiness, 0, 35)
    rel_pts = _clamp_int(18 + 18 * (1.0 if has_best or len(ql) > 6 else 0.7), 0, 35)
    data_pts = _clamp_int(10 + 15 * specificity, 0, 30)
    trust_breakdown = {
        "Brand strength": brand_pts,
        "Query relevance": rel_pts,
        "Data confidence": data_pts,
    }

    region_pts = _clamp_int(22 + 28 * (0.55 if generic_q else 1.0), 0, 50)
    avail_pts = _clamp_int(14 + 18 * brandiness + (6 if not generic_q else 0), 0, 35)
    geo_breakdown = {
        "Region match": region_pts,
        "Availability": avail_pts,
    }

    conf_base = (trust_int + geo_int) / 2 + rng.randint(-4, 6)
    confidence = _clamp_int(conf_base, 40, 100)
    if used_fallback:
        confidence = _clamp_int(confidence - 12, 40, 100)
    if agent_failed:
        confidence = _clamp_int(confidence - 15, 40, 100)
    if low_results:
        confidence = _clamp_int(confidence - 8, 40, 100)

    if confidence >= 90:
        conf_tier, conf_label = "🟢", "High"
    elif confidence >= 60:
        conf_tier, conf_label = "🟡", "Medium"
    else:
        conf_tier, conf_label = "🔴", "Low"

    explanation_lines = [
        f"**Domain inferred:** {domain.replace('_', ' ').title()}",
        f"**Ranking rationale:** Prioritized recognizable issuers and concrete product names aligned with your query.",
        f"**Trust ({trust_int}%):** {'Strong brand signals' if brandiness > 0.55 else 'Mixed signals'}; "
        f"{'boosted for comparative intent (\"best\")' if has_best else 'standard weighting'}.",
        f"**GEO ({geo_int}%):** {'Regional fit moderated (generic intent)' if generic_q else 'Strong intent-to-offer alignment'} "
        f"with availability-weighted coverage.",
        f"**Confidence ({confidence}% — {conf_label} {conf_tier}):** Synthesized from agent agreement"
        f"{' with fallback path engaged' if used_fallback else ''}.",
    ]
    explanation = "\n\n".join(explanation_lines)

    trace: List[Dict[str, Any]] = []
    for i, label in enumerate(TRACE_STEP_LABELS):
        key = AGENT_KEYS[i]
        ms = int(rng.uniform(420, 980))
        if fail_idx == i:
            name = f"{label} ⚠ {key.title()} agent failed → using fallback"
            trace.append({"name": name, "duration_ms": ms, "status": "fallback"})
        else:
            trace.append({"name": label, "duration_ms": ms, "status": "success"})

    product_notes: Dict[str, str] = {}
    for name in ranking:
        if "HDFC" in name:
            product_notes[name] = "Strong retail footprint; fast digital servicing."
        elif "ICICI" in name:
            product_notes[name] = "Broad branch + app-led journey."
        elif "SBI" in name:
            product_notes[name] = "Public-sector scale; predictable policies."
        elif "Axis" in name or "Kotak" in name:
            product_notes[name] = "Competitive salaried segments."
        elif "Star" in name:
            product_notes[name] = "Health-focused underwriting."
        else:
            product_notes[name] = "Demo corpus match; verify on issuer site."

    retrieval_output = {
        "query": q,
        "domain_inferred": domain,
        "retrieved_documents": [
            {
                "id": f"doc_{i+1}",
                "title": retrieval[i] if i < len(retrieval) else f"{domain} candidate",
                "source": "simulated-corpus",
                "score": round(0.9 - i * 0.07 + rng.uniform(-0.02, 0.02), 3),
                "snippet": f"Simulated excerpt for '{q[:48]}…'." if len(q) > 48 else f"Simulated excerpt for '{q}'.",
            }
            for i in range(min(5, max(3, len(retrieval))))
        ],
        "notes": "Fallback" if used_fallback else "Simulated retrieval (no external DB).",
        "ts_utc": datetime.now(timezone.utc).isoformat(),
    }

    ranking_raw_llm_output = "\n".join(
        ["SYSTEM: You are a ranking agent.", f"USER: Rank options for: {q}", "ASSISTANT:"]
        + [f"{j+1}) {r}" for j, r in enumerate(ranking)]
        + ["", "Rationale: Prefer established brands and specific product names."]
    ).strip()

    trust_calculation_steps = {
        "inputs": {"brandiness": round(brandiness, 3), "specificity": round(specificity, 3), "flags": flags},
        "modifiers": {"best_query": has_best, "brand_in_query": has_brand_in_query, "generic_query": generic_q},
        "breakdown_points": trust_breakdown,
        "result": {"trust_score": trust_f, "trust_int": trust_int},
    }
    geo_calculation_steps = {
        "inputs": {"brandiness": round(brandiness, 3), "specificity": round(specificity, 3), "flags": flags},
        "breakdown_points": geo_breakdown,
        "result": {"geo_score": geo_f, "geo_int": geo_int},
    }
    explanation_prompt = "\n".join(
        [
            "SYSTEM: Explanation agent.",
            f"QUERY: {q}",
            f"RANKING: {ranking}",
            f"TRUST: {trust_int} GEO: {geo_int} CONF: {confidence}",
        ]
    )

    return {
        "query": q,
        "domain": domain,
        "retrieval": retrieval,
        "ranking": ranking,
        "product_notes": product_notes,
        "trust_score": trust_f,
        "geo_score": geo_f,
        "trust_score_int": trust_int,
        "geo_score_int": geo_int,
        "trust_breakdown": trust_breakdown,
        "geo_breakdown": geo_breakdown,
        "explanation": explanation,
        "trace": trace,
        "confidence_score": confidence,
        "confidence_tier": conf_label,
        "confidence_tier_emoji": conf_tier,
        "used_fallback": used_fallback,
        "agent_failed": agent_failed,
        "failed_agent_key": failed_key,
        "low_results": low_results,
        "debug": {
            "retrieval_output": retrieval_output,
            "ranking_raw_llm_output": {"raw_text": ranking_raw_llm_output},
            "trust_calculation_steps": trust_calculation_steps,
            "geo_calculation_steps": geo_calculation_steps,
            "explanation_prompt_output": {"prompt": explanation_prompt, "output": explanation},
        },
    }


def run_trustlens(query: str) -> Dict[str, Any]:
    """Backward-compatible wrapper returning legacy keys used by older demos."""
    q = (query or "").strip()
    r = simulate_agents(q, simulate_failure=False)
    trend = {
        "trust": _trend_from_target(float(r["trust_score"]), _stable_random(q)),
        "geo": _trend_from_target(float(r["geo_score"]), _stable_random(q + "|geo")),
    }
    out = {k: r[k] for k in ("ranking", "trust_score", "geo_score", "explanation", "trace", "debug")}
    out["trend"] = trend
    return out


def _trend_from_target(target: float, rng: random.Random) -> List[float]:
    t = _clamp01(target)
    start = _clamp01(max(0.10, t - rng.uniform(0.25, 0.45)))
    mid1 = _clamp01(start + rng.uniform(0.08, 0.18))
    mid2 = _clamp01(mid1 + rng.uniform(0.06, 0.14))
    end = t
    mid2 = min(mid2, end - 0.02) if end > 0.04 else min(mid2, end)
    mid1 = min(mid1, mid2 - 0.02) if mid2 > 0.04 else min(mid1, mid2)
    start = min(start, mid1 - 0.02) if mid1 > 0.04 else min(start, mid1)
    return [round(_clamp01(start), 2), round(_clamp01(mid1), 2), round(_clamp01(mid2), 2), round(_clamp01(end), 2)]
