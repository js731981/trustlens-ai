from typing import Any

from pydantic import BaseModel, Field


class DocumentInput(BaseModel):
    """One document to index."""

    id: str | None = Field(
        default=None,
        description="Optional stable id; if omitted, server generates one.",
    )
    text: str = Field(..., min_length=1, description="Body text to embed and store.")
    metadata: dict[str, Any] = Field(default_factory=dict)


class IndexRequest(BaseModel):
    documents: list[DocumentInput] = Field(..., min_length=1)


class IndexResponse(BaseModel):
    indexed: int
    ids: list[str]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    limit: int = Field(default=5, ge=1, le=50)


class SearchHit(BaseModel):
    id: str
    score: float
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]
