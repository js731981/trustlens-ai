import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# Deterministic UUID namespace for mapping arbitrary external ids to Qdrant point ids
_EXTERNAL_ID_NAMESPACE = uuid.UUID("8c5d6f2a-4e1b-4c0d-9a8e-726167730001")


def _point_id(external_id: str | None) -> str:
    if external_id is None:
        return str(uuid.uuid4())
    try:
        uuid.UUID(external_id)
        return external_id
    except ValueError:
        return str(uuid.uuid5(_EXTERNAL_ID_NAMESPACE, external_id))


class QdrantDocumentStore:
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        vector_size: int,
    ) -> None:
        self._client = client
        self._collection = collection_name
        self._vector_size = vector_size

    def ensure_collection(self) -> None:
        collections = self._client.get_collections().collections
        names = {c.name for c in collections}
        if self._collection in names:
            return

        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(
                size=self._vector_size,
                distance=Distance.COSINE,
            ),
        )

    def upsert_documents(
        self,
        items: list[tuple[str, list[float], str, dict[str, Any]]],
    ) -> list[str]:
        """
        items: list of (point_id, vector, text, metadata)
        Returns point ids in order.
        """
        points = [
            PointStruct(
                id=pid,
                vector=vec,
                payload={"text": text, "metadata": meta},
            )
            for pid, vec, text, meta in items
        ]
        self._client.upsert(collection_name=self._collection, points=points)
        return [p.id for p in points]  # type: ignore[misc]

    def search(self, vector: list[float], limit: int) -> list[dict[str, Any]]:
        response = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=limit,
            with_payload=True,
        )
        results = response.points
        hits: list[dict[str, Any]] = []
        for r in results:
            payload = r.payload or {}
            hits.append(
                {
                    "id": str(r.id),
                    "score": float(r.score),
                    "text": str(payload.get("text", "")),
                    "metadata": payload.get("metadata") or {},
                }
            )
        return hits


def build_qdrant_client(url: str, api_key: str | None) -> QdrantClient:
    if url.strip() == ":memory:":
        return QdrantClient(location=":memory:", api_key=api_key or None)
    return QdrantClient(url=url, api_key=api_key or None)


def external_to_point_id(external_id: str | None) -> str:
    return _point_id(external_id)
