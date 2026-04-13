from typing import Literal

from pydantic import BaseModel, Field

BiasType = Literal["brand", "popularity", "hallucination"]


class RankedProduct(BaseModel):
    rank: int = Field(ge=1, description="1 is highest priority / best fit")
    name: str
    notes: str | None = None


class FinancialQueryRequest(BaseModel):
    user_query: str = Field(min_length=1, max_length=8000)
    template_id: str = Field(
        default="financial_ranking",
        description="Prompt template folder name under app/prompts/templates",
    )


class FinancialQueryResponse(BaseModel):
    ranked_products: list[RankedProduct]
    explanation: str


class RankingConsistencyResult(BaseModel):
    """Metrics from repeated identical ranking queries (e.g. LLM stochasticity)."""

    ranking_variance: float = Field(
        description="Mean across items of the sample variance of rank across runs.",
    )
    position_shifts: float = Field(
        description="Mean absolute deviation of each item's rank from its mean rank, averaged over items.",
    )
    stability_score: float = Field(
        ge=0,
        le=1,
        description="Kendall's W on items present in every run when |intersection| >= 2; else a variance-based proxy.",
    )
    consistency_score: float = Field(
        ge=0,
        le=1,
        description="Blended 0–1 score from stability, variance, and position shifts.",
    )
    n_runs: int
    n_items_union: int
    n_items_intersection: int


class RecommendationBiasResult(BaseModel):
    """Heuristic bias flags for LLM rankings vs dataset ground truth."""

    bias_detected: bool
    bias_type: BiasType | None = Field(
        default=None,
        description="Set when bias_detected is true: brand, popularity, or hallucination.",
    )


class RecommendationBiasRequest(BaseModel):
    """Ordered LLM picks vs dataset ordering; optional repeated rank-1s for brand bias."""

    ranked_product_names: list[str] = Field(
        min_length=1,
        description="LLM output best-first (same order as ranks 1..n).",
    )
    ground_truth_product_names: list[str] = Field(
        description="Dataset products best-first (ground truth ordering).",
    )
    repeat_run_rank_one_names: list[str] | None = Field(
        default=None,
        description="Optional: rank-1 product name from each repeated query (same prompt).",
    )
