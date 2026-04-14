from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List


INSURANCE_PRODUCTS: List[str] = [
    "HDFC Ergo Health Insurance",
    "ICICI Lombard Health Insurance",
    "Star Health Insurance",
]

LOAN_PRODUCTS: List[str] = [
    "HDFC Bank Personal Loan",
    "SBI Personal Loan",
    "ICICI Personal Loan",
]


def _stable_random(query: str) -> random.Random:
    """
    Deterministic RNG seeded from the query, so the demo feels consistent
    for the same input while still having slight variation across inputs.
    """

    digest = hashlib.md5(query.strip().lower().encode("utf-8")).hexdigest()
    seed = int(digest[:8], 16)
    return random.Random(seed)


def run_trustlens(query: str) -> Dict[str, Any]:
    """
    Lightweight TrustLens demo logic.

    Returns:
        {
          "ranking": [str, str, str],
          "score": float,          # 0-1
          "explanation": str
        }
    """

    q = (query or "").strip()
    q_lower = q.lower()
    rng = _stable_random(q)

    is_insurance = ("health" in q_lower) or ("insurance" in q_lower)
    is_loan = "loan" in q_lower
    is_family = "family" in q_lower

    if is_insurance:
        ranking = INSURANCE_PRODUCTS.copy()
        if is_family:
            # Family queries typically favor broader/benefit-rich plans first.
            ranking = [
                "HDFC Ergo Health Insurance",
                "Star Health Insurance",
                "ICICI Lombard Health Insurance",
            ]
        score_min, score_max = 0.84, 0.95
        domain_hint = "health insurance coverage"
    elif is_loan:
        ranking = LOAN_PRODUCTS.copy()
        if is_family:
            # Family needs often correlate with flexibility/repayment comfort.
            ranking = [
                "SBI Personal Loan",
                "HDFC Bank Personal Loan",
                "ICICI Personal Loan",
            ]
        score_min, score_max = 0.80, 0.92
        domain_hint = "loan suitability"
    else:
        mixed = [
            "HDFC Ergo Health Insurance",
            "HDFC Bank Personal Loan",
            "ICICI Lombard Health Insurance",
            "SBI Personal Loan",
            "Star Health Insurance",
            "ICICI Personal Loan",
        ]
        rng.shuffle(mixed)
        ranking = mixed[:3]
        score_min, score_max = 0.75, 0.90
        domain_hint = "overall fit"

    score = round(rng.uniform(score_min, score_max), 4)

    explanation = (
        "Based on your query, these products are ranked considering coverage, reliability, "
        "and market reputation. The top recommendation is prioritized for strong benefits "
        f"aligned with your needs around {domain_hint}, while the remaining options provide "
        "competitive alternatives with different strengths."
    )

    return {"ranking": ranking, "score": score, "explanation": explanation}


def _stable_random_for_model(query: str, model_key: str) -> random.Random:
    """
    Deterministic RNG seeded from (query, model_key) so each model gets a
    repeatable but slightly different output for the same query.
    """
    key = f"{(query or '').strip().lower()}|{(model_key or '').strip().lower()}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    seed = int(digest[:8], 16)
    return random.Random(seed)


def _slight_shuffle(items: List[str], rng: random.Random) -> List[str]:
    """
    Make a minimal, "plausible" variation:
    - mostly keep the same order
    - occasionally swap two nearby items
    """
    out = list(items or [])
    if len(out) <= 1:
        return out
    if len(out) == 2:
        if rng.random() < 0.35:
            out[0], out[1] = out[1], out[0]
        return out

    # For 3+ items: swap two adjacent positions with modest probability.
    if rng.random() < 0.70:
        i = rng.randrange(0, len(out) - 1)
        out[i], out[i + 1] = out[i + 1], out[i]
    return out


def run_multi_llm(query: str) -> dict:
    """
    Simulate outputs from multiple LLMs and return side-by-side comparable results.

    Returns:
        {
          "gpt4": {"ranking": [...], "score": 0.9},
          "claude": {"ranking": [...], "score": 0.85},
          "llama": {"ranking": [...], "score": 0.8}
        }
    """
    base = run_trustlens(query)
    base_ranking: List[str] = list(base.get("ranking") or [])
    base_score = float(base.get("score") or 0.0)

    def model_result(model_key: str, score_bias: float) -> Dict[str, Any]:
        rng = _stable_random_for_model(query, model_key)
        ranking = _slight_shuffle(base_ranking, rng)
        jitter = rng.uniform(-0.02, 0.02)
        score = max(0.0, min(1.0, base_score + score_bias + jitter))
        score = round(score, 4)
        return {"ranking": ranking, "score": score}

    return {
        "gpt4": model_result("gpt4", score_bias=0.015),
        "claude": model_result("claude", score_bias=0.0),
        "llama": model_result("llama", score_bias=-0.015),
    }
