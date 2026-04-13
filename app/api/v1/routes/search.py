from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.core.logging import get_logger
from app.models.search import SearchRequest, SearchResponse, SearchResultItem
from app.services import embedding_service, qdrant_service

router = APIRouter()
logger = get_logger(__name__)


@router.post("/search", response_model=SearchResponse)
def search(body: SearchRequest) -> SearchResponse:
    try:
        vector = embedding_service.embed_text(body.query)
        hits = qdrant_service.search(vector, top_k=body.top_k)
    except Exception as exc:
        logger.exception("search_failed")
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Search failed: {exc}",
        ) from exc

    results: list[SearchResultItem] = []
    for hit in hits:
        payload = hit.get("payload") or {}
        name = str(payload.get("name", "")).strip()
        if not name:
            name = str(hit.get("id", ""))
        results.append(SearchResultItem(name=name, score=float(hit["score"])))
    return SearchResponse(results=results)
