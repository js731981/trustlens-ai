from __future__ import annotations

from fastapi import APIRouter, Query

from app.models.history import QueryHistoryEntry
from app.services.history.history_service import get_history

router = APIRouter()


@router.get("/history", response_model=list[QueryHistoryEntry])
async def fetch_history(limit: int = Query(default=10, ge=1, le=100)) -> list[QueryHistoryEntry]:
    rows = get_history(limit=limit)
    return [QueryHistoryEntry.model_validate(r) for r in rows]

