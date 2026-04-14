from __future__ import annotations

from fastapi import APIRouter, status

from app.models.comparison import CompetitorComparisonRequest, CompetitorComparisonResponse
from app.services.comparison.comparator import competitor_comparison

router = APIRouter()


@router.post(
    "/comparison/competitors",
    response_model=CompetitorComparisonResponse,
    status_code=status.HTTP_200_OK,
)
async def compare_competitors(body: CompetitorComparisonRequest) -> CompetitorComparisonResponse:
    result = await competitor_comparison(query=body.query, company=body.company)
    return CompetitorComparisonResponse.model_validate(result)

