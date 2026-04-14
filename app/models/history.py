from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class QueryHistoryEntry(BaseModel):
    id: int
    query: str
    provider: str
    timestamp: datetime
    trust_score: float | None = Field(default=None, ge=0.0, le=1.0)
    geo_score: float | None = Field(default=None, ge=0.0, le=1.0)

