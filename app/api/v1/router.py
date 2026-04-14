from fastapi import APIRouter

from app.api.v1.routes import (
    analyze,
    comparison,
    drift,
    financial,
    geo,
    health,
    history,
    insights,
)

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(analyze.router, tags=["analyze"])
api_router.include_router(financial.router, tags=["financial"])
api_router.include_router(insights.router, tags=["insights"])
api_router.include_router(comparison.router, tags=["comparison"])
api_router.include_router(drift.router, tags=["drift"])
api_router.include_router(history.router, tags=["history"])
api_router.include_router(geo.router, tags=["geo"])
