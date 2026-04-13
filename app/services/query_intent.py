from __future__ import annotations

from typing import Literal


Intent = Literal["insurance", "loan"]


def classify_query(query: str) -> Intent:
    """Coarse intent for ranking dataset selection (insurance vs loan)."""
    q = (query or "").lower()
    loan_terms = (
        "loan",
        "emi",
        "borrow",
        "lending",
        "mortgage",
        "personal loan",
        "disbursal",
        "interest rate",
        "credit",
    )
    insurance_terms = (
        "insurance",
        "policy",
        "premium",
        "coverage",
        "claim",
        "health plan",
        "insured",
        "sum insured",
    )
    loan_score = sum(1 for t in loan_terms if t in q)
    ins_score = sum(1 for t in insurance_terms if t in q)
    if loan_score > ins_score:
        return "loan"
    return "insurance"
