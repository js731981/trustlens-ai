from app.models.health import HealthStatus
from app.utils.version import API_VERSION


def get_health_status() -> HealthStatus:
    return HealthStatus(status="ok", version=API_VERSION)
