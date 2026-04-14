from __future__ import annotations

from typing import Any


def analyze_geo(data: dict[str, Any]) -> dict[str, Any]:
    products = data.get("ranked_products", []) or []
    explanation = data.get("explanation", "") or ""

    issues: list[str] = []
    recommendations: list[str] = []

    if not isinstance(products, list):
        products = []

    names = [
        (p.get("name", "") if isinstance(p, dict) else "").strip().lower()
        for p in products
    ]

    # Rule 1: Brand Coverage
    top_brands = ["hdfc", "icici", "sbi", "axis", "kotak"]
    if names and not any(any(b in name for b in top_brands) for name in names):
        issues.append("Missing major financial brands (HDFC, ICICI, SBI, etc.)")
        recommendations.append(
            "Include top Indian financial institutions to improve trust and visibility"
        )

    # Rule 2: Generic Names Detection
    generic_words = ["plan", "product", "insurance", "loan"]
    generic_count = sum(
        1 for name in names if any(word in name for word in generic_words)
    )
    if names and generic_count == len(names):
        issues.append("All results are generic product names")
        recommendations.append(
            "Use specific product or company names instead of generic labels"
        )

    # Rule 3: Regional Context
    if "india" not in explanation.lower():
        issues.append("Lack of regional relevance (India not mentioned)")
        recommendations.append(
            "Include region-specific context (e.g., India) in explanation"
        )

    # Rule 4: Ranking Depth
    if len(products) < 3:
        issues.append("Too few ranked results")
        recommendations.append("Provide at least 3–5 ranked options for better comparison")

    # Rule 5: Explanation Quality
    if len(explanation.split()) < 20:
        issues.append("Explanation is too short")
        recommendations.append("Provide more detailed reasoning for ranking")

    # Score calculation (deterministic)
    total_checks = 5
    penalty = len(issues)
    score = max(0.0, 1.0 - (penalty / total_checks))

    return {
        "issues": issues,
        "recommendations": recommendations,
        "score": round(score, 2),
    }

