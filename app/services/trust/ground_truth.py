"""Load query-keyed ground truth product rankings from JSON."""

from __future__ import annotations

import json
from pathlib import Path


def _normalize_query_key(query: str) -> str:
    return " ".join(query.strip().lower().split())


def load_ground_truth_for_query(query: str, ground_truth_path: Path) -> list[str] | None:
    """
    Return the ordered ground-truth product names for ``query`` if the file exists
    and contains a matching key (exact or whitespace-normalized case-insensitive).
    """
    if not ground_truth_path.is_file():
        return None
    try:
        raw = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None

    wanted = _normalize_query_key(query)
    for key, value in raw.items():
        if not isinstance(key, str):
            continue
        if _normalize_query_key(key) != wanted:
            continue
        if not isinstance(value, list):
            return None
        out: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif item is not None:
                s = str(item).strip()
                if s:
                    out.append(s)
        return out if out else None
    return None
