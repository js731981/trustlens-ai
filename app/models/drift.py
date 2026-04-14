from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DriftHistoryItem(BaseModel):
    query: str
    product: str
    rank: int = Field(ge=1)
    timestamp: str
    rank_change: int | None = None

    # Allow forward compatibility if we later add fields in the DB history dicts.
    model_config = {"extra": "allow"}


class DriftResponse(BaseModel):
    history: list[DriftHistoryItem]
    drift_score: float = Field(ge=0.0, le=1.0)
    meta: dict[str, Any] | None = None

