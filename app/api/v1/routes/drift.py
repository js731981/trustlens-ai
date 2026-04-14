from __future__ import annotations

from fastapi import APIRouter, Query

from app.models.drift import DriftResponse
from app.services.drift.drift_tracker import get_drift

router = APIRouter()


@router.get("/drift", response_model=DriftResponse)
def drift(query: str = Query(min_length=1, max_length=8000)) -> DriftResponse:
    history, score = get_drift(query)
    return DriftResponse(history=history, drift_score=score, meta={"query": query, "n_points": len(history)})

