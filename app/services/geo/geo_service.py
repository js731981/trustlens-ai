from __future__ import annotations

from typing import Any


KNOWN_BRANDS = [
    "hdfc",
    "icici",
    "axis",
    "sbi",
    "bajaj",
    "kotak",
    "idfc",
    "pnb",
    "star health",
]

_GEO_KEYWORDS = ("bank", "insurance", "loan")


def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _brands_in_text(text: str) -> list[str]:
    # Robust partial matching on normalized product.lower()
    t = (text or "").lower()
    found: list[str] = []
    for b in KNOWN_BRANDS:
        bb = (b or "").lower()
        if not bb:
            continue
        if bb in t:
            found.append(b)
    # de-dupe but keep order
    out: list[str] = []
    seen: set[str] = set()
    for b in found:
        if b in seen:
            continue
        seen.add(b)
        out.append(b)
    return out


def analyze_geo(
    data: dict[str, Any],
    *,
    rag_has_context: bool = False,
    llm_valid: bool = True,
) -> dict[str, Any]:
    products = data.get("ranked_products", []) or []
    explanation = data.get("explanation", "") or ""

    issues: list[str] = []
    recommendations: list[str] = []

    if not isinstance(products, list):
        products = []

    raw_names = [(p.get("name", "") if isinstance(p, dict) else "").strip() for p in products]
    names = [n.lower() for n in raw_names if n]

    def _is_generic_name(name: str) -> bool:
        s = (name or "").strip().lower()
        if not s:
            return True
        if s in ("provider a", "provider b", "provider c"):
            return True
        if s.startswith("provider "):
            return True
        return False

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

    # GEO scoring (financial credibility + specificity), per requirements:
    # Per product:
    # - known brand → +0.4
    # - product name length > 12 → +0.2
    # - contains keyword (bank/loan/insurance) → +0.2
    # - RAG context used → +0.2
    # Then average across ranked list; cap at 1.0.
    geo_reason_parts: list[str] = []
    if not raw_names:
        score = 0.0
        geo_reason_parts.append("No products to score")
    else:
        scores: list[float] = []
        item_reasons: list[str] = []
        for product in raw_names:
            if not product:
                continue
            if _is_generic_name(product):
                # Preserve existing "generic placeholders are bad" behavior.
                scores.append(0.0)
                item_reasons.append(f"{product}: generic placeholder → 0.0")
                continue

            geo = 0.0
            parts: list[str] = []
            brands = _brands_in_text(product)
            if brands:
                geo += 0.4
                parts.append(f"brand {brands[0]} → +0.4")

            if len(product) > 12:
                geo += 0.2
                parts.append("length>12 → +0.2")

            norm = _norm(product)
            matched_kw = next((k for k in _GEO_KEYWORDS if k in norm), None)
            if matched_kw is not None:
                geo += 0.2
                parts.append(f"keyword {matched_kw} → +0.2")

            if rag_has_context:
                geo += 0.2
                parts.append("RAG context → +0.2")

            geo = float(min(geo, 1.0))
            scores.append(geo)
            item_reasons.append(f"{product}: {', '.join(parts) if parts else 'no signals'}")

        score = float(min(sum(scores) / max(len(scores), 1), 1.0))
        sample = item_reasons[:5]
        suffix = "" if len(item_reasons) <= len(sample) else f" (and {len(item_reasons) - len(sample)} more)"
        geo_reason_parts.append("Avg geo across products. " + " | ".join(sample) + suffix)

    score_f = float(round(float(score), 2))
    if not llm_valid:
        score_f = float(round(score_f * 0.7, 2))
        geo_reason_parts.append("LLM output invalid → geo_score × 0.7")

    return {
        "issues": issues,
        "recommendations": recommendations,
        "score": score_f,
        "geo_reason": "; ".join([p for p in geo_reason_parts if p]) if geo_reason_parts else "GEO score computed",
    }

