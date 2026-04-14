from __future__ import annotations

import difflib
from typing import Any, Iterable

# difflib cutoff: higher = stricter (default in library is 0.6)
_CLOSE_MATCH_CUTOFF = 0.55


def _canonical_allowed(allowed_products: Iterable[str]) -> tuple[frozenset[str], list[str], dict[str, str]]:
    """Exact set, ordered list for difflib, and casefold -> canonical name."""
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in allowed_products:
        if not isinstance(raw, str):
            continue
        name = raw.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    by_casefold = {n.casefold(): n for n in ordered}
    return frozenset(ordered), ordered, by_casefold


def _resolve_name(
    name: str,
    allowed_exact: frozenset[str],
    allowed_list: list[str],
    by_casefold: dict[str, str],
) -> str | None:
    """Return canonical catalog name, or None if no acceptable match."""
    trimmed = name.strip()
    if not trimmed:
        return None
    if trimmed in allowed_exact:
        return trimmed
    canon = by_casefold.get(trimmed.casefold())
    if canon is not None:
        return canon
    matches = difflib.get_close_matches(
        trimmed, allowed_list, n=1, cutoff=_CLOSE_MATCH_CUTOFF
    )
    if matches:
        return matches[0]
    return None


def validate_products(output: dict, allowed_products: Iterable[str]) -> dict[str, Any]:
    """
    Ensure each ``ranked_products`` entry uses a name from the dataset.

    Names that match exactly (or case-insensitively) are kept with canonical
    spelling from ``allowed_products``. Otherwise the closest difflib match
    above ``_CLOSE_MATCH_CUTOFF`` replaces the name; if none qualifies, the
    row is removed. Other top-level keys on ``output`` are preserved.
    """
    allowed_exact, allowed_list, by_casefold = _canonical_allowed(allowed_products)
    src: dict[str, Any] = output if isinstance(output, dict) else {}

    raw = src.get("ranked_products")
    if raw is None or not isinstance(raw, list):
        raw = []

    validated: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name_val = item.get("name")
        if name_val is None:
            continue
        name_str = str(name_val).strip()
        resolved = _resolve_name(name_str, allowed_exact, allowed_list, by_casefold)
        if resolved is None:
            continue
        row = {**item, "name": resolved}
        validated.append(row)

    out: dict[str, Any] = {k: v for k, v in src.items() if k != "ranked_products"}
    out["ranked_products"] = validated
    return out
