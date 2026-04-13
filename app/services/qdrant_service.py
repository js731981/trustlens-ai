from __future__ import annotations

import os
from threading import Lock
from typing import Any, Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

COLLECTION_NAME = "financial_products"
# Matches `all-MiniLM-L6-v2` in `app/services/embedding_service.py`
DEFAULT_VECTOR_SIZE = 384

_client: QdrantClient | None = None
_lock = Lock()


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
    if client.collection_exists(COLLECTION_NAME):
        return
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=_vector_size(), distance=Distance.COSINE),
    )


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
    client.upsert(collection_name=COLLECTION_NAME, points=points)


def search(query_vector: list[float], top_k: int = 5) -> list[dict[str, Any]]:
    """
    Vector search over `financial_products`.

    Returns scored hits: { "id", "score", "payload" }.
    """
    client = _get_client()
    _ensure_collection(client)
    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points
    return [
        {
            "id": hit.id,
            "score": hit.score,
            "payload": hit.payload or {},
        }
        for hit in hits
    ]
