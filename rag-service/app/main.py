from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from httpx import ConnectError
from qdrant_client.http.exceptions import ResponseHandlingException

from app.models.schemas import (
    DocumentInput,
    IndexRequest,
    IndexResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from app.services.embeddings import EmbeddingService
from app.services.qdrant_store import (
    QdrantDocumentStore,
    build_qdrant_client,
    external_to_point_id,
)
from app.utils.config import Settings, get_settings

_embedder: EmbeddingService | None = None
_store: QdrantDocumentStore | None = None


def get_embedder() -> EmbeddingService:
    if _embedder is None:
        raise HTTPException(status_code=503, detail="Embeddings not ready")
    return _embedder


def get_store() -> QdrantDocumentStore:
    if _store is None:
        raise HTTPException(status_code=503, detail="Vector store not ready")
    return _store


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _embedder, _store
    settings = get_settings()
    embedder = EmbeddingService(
        settings.embedding_model,
        vector_size=settings.embedding_vector_size,
    )
    if settings.embedding_load_on_startup:
        embedder.load()
    client = build_qdrant_client(settings.qdrant_url, settings.qdrant_api_key)
    store = QdrantDocumentStore(
        client=client,
        collection_name=settings.qdrant_collection,
        vector_size=settings.embedding_vector_size,
    )
    try:
        store.ensure_collection()
    except (ConnectError, ResponseHandlingException) as e:
        url = settings.qdrant_url
        hint = (
            f"Cannot reach Qdrant at {url!r}. Start Qdrant (e.g. "
            "`docker run -p 6333:6333 qdrant/qdrant`) or set QDRANT_URL=:memory: "
            "in rag-service/.env for local in-process storage (no persistence). "
            "See rag-service/.env.example."
        )
        raise RuntimeError(hint) from e
    _embedder = embedder
    _store = store
    yield
    _embedder = None
    _store = None


def create_app(settings: Settings | None = None) -> FastAPI:
    s = settings or get_settings()
    docs_url = "/docs" if s.environment != "production" else None
    redoc_url = "/redoc" if s.environment != "production" else None
    return FastAPI(
        title="RAG Service",
        version="1.0.0",
        lifespan=lifespan,
        docs_url=docs_url,
        redoc_url=redoc_url,
    )


app = create_app()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "rag-service",
        "embeddings_loaded": _embedder is not None and _embedder._model is not None,  # type: ignore[attr-defined]
    }


@app.post("/index", response_model=IndexResponse)
def index_documents(
    body: IndexRequest,
    embedder: EmbeddingService = Depends(get_embedder),
    store: QdrantDocumentStore = Depends(get_store),
) -> IndexResponse:
    texts = [d.text for d in body.documents]
    try:
        vectors = embedder.embed(texts)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=(
                "Embeddings model is not available yet (download/load failed). "
                "Check network access to Hugging Face or pre-download the model. "
                "You can also avoid loading at startup by keeping "
                "EMBEDDING_LOAD_ON_STARTUP=false (default)."
            ),
        ) from e
    items: list[tuple[str, list[float], str, dict[str, Any]]] = []
    returned_ids: list[str] = []
    for doc, vec in zip(body.documents, vectors, strict=True):
        pid = external_to_point_id(doc.id)
        items.append((pid, vec, doc.text, doc.metadata))
        returned_ids.append(pid)
    store.upsert_documents(items)
    return IndexResponse(indexed=len(returned_ids), ids=returned_ids)


@app.post("/search", response_model=SearchResponse)
def search_documents(
    body: SearchRequest,
    embedder: EmbeddingService = Depends(get_embedder),
    store: QdrantDocumentStore = Depends(get_store),
) -> SearchResponse:
    try:
        (qvec,) = embedder.embed([body.query])
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=(
                "Embeddings model is not available yet (download/load failed). "
                "Check network access to Hugging Face or pre-download the model."
            ),
        ) from e
    raw = store.search(qvec, body.limit)
    hits = [
        SearchHit(
            id=h["id"],
            score=h["score"],
            text=h["text"],
            metadata=h["metadata"],
        )
        for h in raw
    ]
    return SearchResponse(query=body.query, hits=hits)
