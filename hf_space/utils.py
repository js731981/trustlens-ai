from __future__ import annotations

import hashlib
import random
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
]

INSURANCE_BUDGET: List[str] = [
    "Star Health Smart Health Pro (Budget)",
    "HDFC Ergo Optima Secure (Value)",
    "ICICI Lombard Health Booster (Entry)",
]

LOAN_FAMILY: List[str] = [
    "SBI Personal Loan (Flexible EMI)",
    "HDFC Bank Personal Loan (Quick Disbursal)",
    "ICICI Personal Loan (Digital Journey)",
]

LOAN_BUDGET: List[str] = [
    "SBI Personal Loan (Low-rate segment)",
    "ICICI Personal Loan (Offers for salaried)",
    "HDFC Bank Personal Loan (Preferred customers)",
]

GENERIC_OPTIONS: List[str] = [
    "Trusted Health Cover Plan",
    "Affordable Family Protection Plan",
    "Instant Personal Loan Plan",
    "Budget-Friendly Insurance Plan",
    "Comprehensive Coverage Plan",
]


def _stable_random(query: str) -> random.Random:
    """
    Deterministic RNG seeded from the query, so the demo feels consistent
    for the same input while still having slight variation across inputs.
    """

    digest = hashlib.md5(query.strip().lower().encode("utf-8")).hexdigest()
    seed = int(digest[:8], 16)
    return random.Random(seed)


def _normalize_query(query: str) -> str:
    return " ".join((query or "").strip().lower().split())


def _pick_domain_and_ranking(q: str, rng: random.Random) -> Tuple[str, List[str], Dict[str, bool]]:
    ql = _normalize_query(q)
    is_insurance = ("insurance" in ql) or ("health" in ql)
    is_loan = "loan" in ql
    wants_family = "family" in ql
    wants_cheap = ("cheap" in ql) or ("affordable" in ql) or ("budget" in ql) or ("low cost" in ql)

    flags = {"insurance": is_insurance, "loan": is_loan, "family": wants_family, "cheap": wants_cheap}

    if is_insurance and wants_family:
        ranking = INSURANCE_FAMILY[:]
        domain = "health insurance"
    elif is_insurance and wants_cheap:
        ranking = INSURANCE_BUDGET[:]
        domain = "health insurance"
    elif is_insurance:
        ranking = [
            "HDFC Ergo Health Insurance",
            "ICICI Lombard Health Insurance",
            "Star Health Insurance",
        ]
        domain = "health insurance"
    elif is_loan and wants_family:
        ranking = LOAN_FAMILY[:]
        domain = "personal loan"
    elif is_loan and wants_cheap:
        ranking = LOAN_BUDGET[:]
        domain = "personal loan"
    elif is_loan:
        ranking = [
            "HDFC Bank Personal Loan",
            "SBI Personal Loan",
            "ICICI Personal Loan",
        ]
        domain = "personal loan"
    else:
        # Unknown intent: blend branded + generic to demonstrate scoring behavior.
        pool = [
            "HDFC Ergo Health Insurance",
            "ICICI Lombard Health Insurance",
            "Star Health Insurance",
            "HDFC Bank Personal Loan",
            "SBI Personal Loan",
            "ICICI Personal Loan",
        ] + GENERIC_OPTIONS
        rng.shuffle(pool)
        ranking = pool[:3]
        domain = "financial products"

    # Slight deterministic "LLM variation": shuffle 2nd/3rd occasionally
    if len(ranking) >= 3 and rng.random() < 0.35:
        ranking[1], ranking[2] = ranking[2], ranking[1]
    return domain, ranking[:3], flags


def _brandiness_score(ranking: List[str]) -> float:
    if not ranking:
        return 0.0
    branded = 0
    for r in ranking:
        s = (r or "").strip()
        if not s:
            continue
        if any(b.lower() in s.lower() for b in KNOWN_BRANDS):
            branded += 1
    return branded / max(1, len(ranking))


def _specificity_score(ranking: List[str]) -> float:
    # Heuristic: more tokens + brand presence => more "specific"
    if not ranking:
        return 0.0
    vals: list[float] = []
    for r in ranking:
        s = " ".join((r or "").strip().split())
        if not s:
            continue
        tok = len(s.split(" "))
        has_brand = 1.0 if any(b.lower() in s.lower() for b in KNOWN_BRANDS) else 0.0
        vals.append(min(1.0, (tok / 6.0) * 0.55 + has_brand * 0.45))
    return sum(vals) / max(1, len(vals))


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _trend_from_target(target: float, rng: random.Random) -> List[float]:
    """
    Produce 4-point rising trend ending at target, deterministic per query.
    """
    t = _clamp01(target)
    start = _clamp01(max(0.10, t - rng.uniform(0.25, 0.45)))
    mid1 = _clamp01(start + rng.uniform(0.08, 0.18))
    mid2 = _clamp01(mid1 + rng.uniform(0.06, 0.14))
    end = _clamp01(max(mid2, t))

    # If end overshoots target, gently pull it back but keep monotonic.
    end = t
    mid2 = min(mid2, end - 0.02) if end > 0.04 else min(mid2, end)
    mid1 = min(mid1, mid2 - 0.02) if mid2 > 0.04 else min(mid1, mid2)
    start = min(start, mid1 - 0.02) if mid1 > 0.04 else min(start, mid1)

    return [round(_clamp01(start), 2), round(_clamp01(mid1), 2), round(_clamp01(mid2), 2), round(_clamp01(end), 2)]


def run_trustlens(query: str) -> Dict[str, Any]:
    """
    Demo-only TrustLens simulation.

    Returns:
        {
          "ranking": [...],
          "trust_score": 0.82,
          "geo_score": 0.74,
          "explanation": "...",
          "trend": {"trust": [...], "geo": [...]}
        }
    """
    q = (query or "").strip()
    rng = _stable_random(q)

    domain, ranking, flags = _pick_domain_and_ranking(q, rng)
    brandiness = _brandiness_score(ranking)  # known brands -> higher trust
    specificity = _specificity_score(ranking)  # specific + branded -> higher GEO

    # Trust: brandiness-heavy; penalize vague/generic
    trust_base = 0.55 + 0.35 * brandiness + 0.10 * specificity
    # GEO: specificity-heavy; still benefits from brands
    geo_base = 0.50 + 0.40 * specificity + 0.10 * brandiness

    # Query intent modifiers
    if flags.get("cheap"):
        # "cheap" queries often bring tradeoffs; trust is slightly lower, GEO slightly lower if generic
        trust_base -= 0.04
        geo_base -= 0.03
    if flags.get("family"):
        trust_base += 0.02
        geo_base += 0.01

    # Small deterministic jitter (keeps it "alive" but stable)
    trust = _clamp01(trust_base + rng.uniform(-0.02, 0.02))
    geo = _clamp01(geo_base + rng.uniform(-0.02, 0.02))

    trust = round(trust, 2)
    geo = round(geo, 2)

    def tone_label(x: float) -> str:
        if x >= 0.85:
            return "High"
        if x >= 0.70:
            return "Medium"
        return "Low"

    # Explanation: concise, product-y, AI-style.
    explanation_bits: list[str] = []
    explanation_bits.append(
        f"I ranked these options for **{domain}** by balancing brand reputation, specificity (how *concrete* the product is), and query intent."
    )
    if flags.get("cheap"):
        explanation_bits.append(
            "Because you asked for *affordability*, the list leans toward value-oriented options—this can slightly reduce trust if names are less established."
        )
    if flags.get("family"):
        explanation_bits.append(
            "For *family* intent, the ranking favors plans/products that typically signal broader coverage or flexible repayment."
        )
    explanation_bits.append(
        f"Overall: **{tone_label(trust)} Trust** / **{tone_label(geo)} GEO**. Branded, clearly named products push both metrics up; vague/generic options pull them down."
    )

    explanation = " ".join(explanation_bits).strip()

    trend = {
        "trust": _trend_from_target(trust, rng),
        "geo": _trend_from_target(geo, rng),
    }

    return {
        "ranking": ranking,
        "trust_score": float(trust),
        "geo_score": float(geo),
        "explanation": explanation,
        "trend": trend,
    }
