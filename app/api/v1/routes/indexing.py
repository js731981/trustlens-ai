from __future__ import annotations

import json
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import PlainTextResponse

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services import embedding_service, qdrant_service

router = APIRouter()
logger = get_logger(__name__)

def _load_json_array(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("expected a JSON array")
    return data


def _item_to_embed_text(item: dict[str, Any]) -> str:
    name = str(item.get("name", "")).strip()
    features = item.get("features") or []
    if isinstance(features, list):
        feat_str = ", ".join(str(x) for x in features)
    else:
        feat_str = str(features)
    return f"{name}. {feat_str}".strip()


def _build_docs(items: list[dict[str, Any]], product_type: str) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for item in items:
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        features = item.get("features")
        if features is not None and not isinstance(features, list):
            features = [str(features)]
        elif features is None:
            features = []
        text = _item_to_embed_text(item)
        vector = embedding_service.embed_text(text)
        point_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"trust-lens/financial-products/{product_type}/{name}")
        )
        docs.append(
            {
                "id": point_id,
                "vector": vector,
                "payload": {
                    "name": name,
                    "type": product_type,
                    "features": features,
                },
            }
        )
    return docs


@router.post("/index", response_class=PlainTextResponse)
def index_datasets() -> str:
    settings = get_settings()
    base = settings.data_dir
    insurance_path = base / "insurance_products.json"
    loans_path = base / "loan_providers.json"
    if not insurance_path.is_file():
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail=f"Missing dataset: {insurance_path}",
        )
    if not loans_path.is_file():
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail=f"Missing dataset: {loans_path}",
        )
    try:
        insurance_items = _load_json_array(str(insurance_path))
        loan_items = _load_json_array(str(loans_path))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.exception("index_datasets_load_failed")
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=f"Could not load datasets: {exc}",
        ) from exc

    docs: list[dict[str, Any]] = []
    docs.extend(_build_docs(insurance_items, "insurance"))
    docs.extend(_build_docs(loan_items, "loan"))
    if not docs:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail="No valid items to index.",
        )
    try:
        qdrant_service.upsert_documents(docs)
    except Exception as exc:
        logger.exception("index_datasets_qdrant_failed")
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Qdrant upsert failed: {exc}",
        ) from exc

    return "Indexed successfully"
