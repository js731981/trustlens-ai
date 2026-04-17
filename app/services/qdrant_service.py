from __future__ import annotations

import os
from threading import Lock
from typing import Any, Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

COLLECTION_NAME = "financial_products"
# Matches `all-MiniLM-L6-v2` in `app/services/embedding_service.py`
DEFAULT_VECTOR_SIZE = 384

_client: QdrantClient | None = None
_lock = Lock()


def _collection_name() -> str:
    # Prefer a single env var used across components.
    return (os.getenv("QDRANT_COLLECTION") or os.getenv("QDRANT_COLLECTION_NAME") or COLLECTION_NAME).strip() or COLLECTION_NAME


def _vector_size() -> int:
    return int(os.getenv("QDRANT_FINANCIAL_PRODUCTS_VECTOR_SIZE", str(DEFAULT_VECTOR_SIZE)))


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                host = os.getenv("QDRANT_HOST", "localhost")
                port = int(os.getenv("QDRANT_PORT", "6333"))
                _client = QdrantClient(host=host, port=port)
    return _client


def _ensure_collection(client: QdrantClient) -> None:
    collection = _collection_name()
    if client.collection_exists(collection):
        return
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=_vector_size(), distance=Distance.COSINE),
    )


def configured_collection_name() -> str:
    return _collection_name()


def expected_vector_size() -> int:
    return _vector_size()


def is_qdrant_available() -> bool:
    try:
        client = _get_client()
        client.get_collections()
        return True
    except Exception:
        return False


def upsert_documents(docs: Sequence[dict[str, Any]]) -> None:
    """
    Upsert points into `financial_products`.

    Each document must provide:
      - id: int or str (UUID string supported)
      - vector: list[float]
      - payload: dict with at least name, type, features (stored as Qdrant payload)
    """
    client = _get_client()
    _ensure_collection(client)
    points = [
        PointStruct(
            id=doc["id"],
            vector=doc["vector"],
            payload=doc["payload"],
        )
        for doc in docs
    ]
    client.upsert(collection_name=_collection_name(), points=points)


def _filters_to_qdrant(filters: dict[str, Any] | None) -> Filter | None:
    if not filters:
        return None
    must: list[FieldCondition] = []
    for key, value in filters.items():
        if value is None:
            continue
        # Keep filter surface simple: exact match on payload fields.
        must.append(
            FieldCondition(
                key=str(key),
                match=MatchValue(value=value),
            )
        )
    return Filter(must=must) if must else None


def search(
    query_vector: list[float],
    top_k: int = 5,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Vector search over `financial_products`.

    Returns scored hits: { "id", "score", "payload" }.
    """
    client = _get_client()
    _ensure_collection(client)
    qfilter = _filters_to_qdrant(filters)
    hits = client.query_points(
        collection_name=_collection_name(),
        query=query_vector,
        limit=top_k,
        with_payload=True,
        query_filter=qfilter,
    ).points
    return [
        {
            "id": hit.id,
            "score": hit.score,
            "payload": hit.payload or {},
        }
        for hit in hits
    ]
