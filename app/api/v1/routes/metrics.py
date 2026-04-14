from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.metrics.metrics_service import compute_dashboard_metrics

router = APIRouter()


class MetricsResponse(BaseModel):
    avg_trust: float = Field(ge=0.0, le=1.0)
    avg_geo: float = Field(ge=0.0, le=1.0)
    trust_series: list[float] = Field(default_factory=list)
    geo_series: list[float] = Field(default_factory=list)
    visibility: float = Field(ge=0.0, le=1.0)
    queries: int = Field(ge=0)


@router.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    """
    Enterprise dashboard rollups.

    - avg_trust: average of recorded analyze run trust_score (0..1)
    - visibility: fraction of LLM responses with parsed_json and no parse_error (0..1)
    - queries: total stored LLM responses (proxy for query volume)
    """
    data = compute_dashboard_metrics(limit=100)
    return MetricsResponse(
        avg_trust=float(data.get("avg_trust") or 0.0),
        avg_geo=float(data.get("avg_geo") or 0.0),
        trust_series=list(data.get("trust_series") or []),
        geo_series=list(data.get("geo_series") or []),
        visibility=float(data.get("visibility") or 0.0),
        queries=int(data.get("queries") or 0),
    )

