from fastapi import APIRouter

from app.core.logging import get_logger
from app.models.health import HealthStatus
from app.services import health as health_service

router = APIRouter()
logger = get_logger(__name__)


@router.get("/health", response_model=HealthStatus)
def health() -> HealthStatus:
    logger.debug("health_check")
    return health_service.get_health_status()
