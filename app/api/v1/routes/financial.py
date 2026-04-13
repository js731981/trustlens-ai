from fastapi import APIRouter, HTTPException, status

from app.core.logging import get_logger
from app.models.financial import (
    FinancialQueryRequest,
    FinancialQueryResponse,
    RecommendationBiasRequest,
    RecommendationBiasResult,
)
from app.prompts.registry import UnknownPromptTemplateError
from app.services import financial_llm as financial_llm_service
from app.services.recommendation_bias import detect_recommendation_bias

router = APIRouter()
logger = get_logger(__name__)


@router.post(
    "/financial/query",
    response_model=FinancialQueryResponse,
    status_code=status.HTTP_200_OK,
)
async def financial_query(body: FinancialQueryRequest) -> FinancialQueryResponse:
    try:
        return await financial_llm_service.query_financial_llm(
            body.user_query,
            template_id=body.template_id,
        )
    except UnknownPromptTemplateError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except financial_llm_service.FinancialLLMConfigurationError as exc:
        logger.warning("financial_query_misconfigured", extra={"detail": str(exc)})
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except financial_llm_service.FinancialLLMResponseParseError as exc:
        logger.error("financial_query_parse_error", extra={"detail": str(exc)})
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            detail="The model response could not be parsed.",
        ) from exc
    except financial_llm_service.FinancialLLMUpstreamError as exc:
        logger.error("financial_query_upstream_error", extra={"detail": str(exc)})
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            detail="The LLM provider request failed.",
        ) from exc


@router.post(
    "/financial/recommendation-bias",
    response_model=RecommendationBiasResult,
    status_code=status.HTTP_200_OK,
)
def recommendation_bias(body: RecommendationBiasRequest) -> RecommendationBiasResult:
    """Compare LLM-ranked names to dataset ground truth and optional repeated rank-1 samples."""
    return detect_recommendation_bias(
        body.ranked_product_names,
        body.ground_truth_product_names,
        repeat_run_rank_one_names=body.repeat_run_rank_one_names,
    )
