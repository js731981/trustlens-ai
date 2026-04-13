from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)


class SearchResultItem(BaseModel):
    name: str
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
