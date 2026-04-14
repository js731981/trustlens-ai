from __future__ import annotations

from pydantic import BaseModel, Field


class CompetitorComparisonRequest(BaseModel):
    query: str = Field(min_length=1)
    company: str = Field(min_length=1)


class CompetitorRank(BaseModel):
    name: str
    rank: int = Field(ge=1)


class CompetitorComparisonResponse(BaseModel):
    your_rank: int = Field(ge=0, description="0 means the company was not found in the ranking.")
    competitors: list[CompetitorRank]

