from __future__ import annotations

from typing import Any

from app.services.history.history_service import get_history


def compute_dashboard_metrics(limit: int = 100) -> dict[str, Any]:
    history = get_history(limit)

    trust_vals: list[float] = []
    geo_vals: list[float] = []
    trust_series: list[float] = []
    geo_series: list[float] = []

    for h in history:
        try:
            t = float(h.get("trust_score", 0) or 0.0)
        except Exception:
            t = 0.0
        try:
            g = float(h.get("geo_score", 0) or 0.0)
        except Exception:
            g = 0.0

        # Clamp for safety/backward compatibility.
        t = float(max(0.0, min(1.0, t)))
        g = float(max(0.0, min(1.0, g)))

        trust_series.append(t)
        geo_series.append(g)
        trust_vals.append(t)
        geo_vals.append(g)

    n = max(len(history), 1)
    avg_trust = sum(trust_vals) / n
    avg_geo = sum(geo_vals) / n

    # A simple proxy: fraction of rows with a non-null trust_score value.
    ok = 0
    for h in history:
        if h.get("trust_score", None) is not None:
            ok += 1
    visibility = float(ok / len(history)) if history else 0.0

    return {
        "avg_trust": float(avg_trust),
        "avg_geo": float(avg_geo),
        "trust_series": trust_series,
        "geo_series": geo_series,
        "visibility": float(max(0.0, min(1.0, visibility))),
        "queries": len(history),
    }

