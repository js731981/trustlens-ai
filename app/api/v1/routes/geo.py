from __future__ import annotations

from fastapi import APIRouter

from app.services.geo.geo_service import analyze_geo

router = APIRouter()


@router.post("/geo")
async def geo_analysis(data: dict) -> dict:
    return analyze_geo(data)

