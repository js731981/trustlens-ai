from __future__ import annotations


def classify_query(query: str) -> str:
    """Classify user query intent using keyword matching (MVP).

    Precedence: insurance > loan > general.
    Matching is case-insensitive.
    """
    text = query.lower()
    if "insurance" in text:
        return "insurance"
    if "loan" in text:
        return "loan"
    return "general"
