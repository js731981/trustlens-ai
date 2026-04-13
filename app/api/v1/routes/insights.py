import asyncio

from fastapi import APIRouter, status

from app.models.insights import ExplanationInsights, ExplanationInsightsRequest
from app.services import explanation_insights as explanation_insights_service

router = APIRouter()


@router.post(
    "/insights/explanation",
    response_model=ExplanationInsights,
    status_code=status.HTTP_200_OK,
)
async def explain_insights(body: ExplanationInsightsRequest) -> ExplanationInsights:
    return await asyncio.to_thread(
        explanation_insights_service.analyze_explanation,
        body.explanation,
    )
