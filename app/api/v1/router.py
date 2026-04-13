from fastapi import APIRouter

from app.api.v1.routes import analyze, financial, health, insights

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(analyze.router, tags=["analyze"])
api_router.include_router(financial.router, tags=["financial"])
api_router.include_router(insights.router, tags=["insights"])
