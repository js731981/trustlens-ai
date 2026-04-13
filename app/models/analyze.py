from datetime import datetime
from typing import Any
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, computed_field

from app.models.financial import RankedProduct, RecommendationBiasResult
from app.models.insights import ExplanationInsights


ProviderName = Literal["ollama", "openai", "openrouter"]
ProviderSelector = Literal["ollama", "openai", "openrouter", "all"]


class AnalyzeRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8000, description="User question for ranking and trust analysis.")
    provider: ProviderSelector = Field(
        default="ollama",
        description='LLM provider to use (default: "ollama"). Use "all" to compare providers.',
    )


class AnalyzeExplanations(BaseModel):
    """Structured explanations for the primary ranking."""

    ranking_rationale: str = Field(description="LLM explanation accompanying the ranked products.")
    insights: ExplanationInsights = Field(
        description="Sentiment and factor coverage derived from the rationale text.",
    )


class AnalyzeTracking(BaseModel):
    """Persisted session context and ranking drift vs the last stored run for the same normalized query."""

    session_id: UUID = Field(description="Correlates parallel LLM calls and rows in the tracking database.")
    kendall_tau_vs_prior: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Kendall's tau between primary ranking and the prior run's ranking (intersection of product names).",
    )
    ranking_drift_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="0 = same pairwise order as prior; 1 = maximal disagreement ((1 - tau) / 2).",
    )
    items_compared_for_drift: int = Field(
        default=0,
        ge=0,
        description="Number of product names in the intersection used for tau / drift.",
    )
    prior_trust_score: float | None = Field(default=None, description="Trust score from the previous stored run for this query, if any.")
    prior_run_id: UUID | None = None
    prior_recorded_at: datetime | None = None


class AnalyzeResponse(BaseModel):
    provider_used: ProviderName = Field(
        description="The provider that produced this response (after any automatic fallback).",
    )
    fallback_used: bool = Field(
        default=False,
        description="True when the requested provider failed and the response was served from Ollama instead.",
    )
    raw_output: str = Field(description="Raw text returned by the selected provider.")
    parsed_output: dict[str, Any] = Field(
        description="JSON object extracted from the raw response (safe parse; may be partial or empty on failure).",
    )

    @computed_field
    @property
    def provider(self) -> ProviderName:
        """Same as ``provider_used``; kept for older clients that read ``provider``."""
        return self.provider_used


class AnalyzeProviderError(BaseModel):
    error: str = Field(description="Provider-specific error message.")
    raw_output: str = Field(
        default="",
        description="Raw model text when available; empty when the call failed before a response body.",
    )
    parsed_output: dict[str, Any] = Field(
        default_factory=dict,
        description="Normalized parse of raw_output when applicable; empty on hard failures.",
    )


AnalyzeProviderResult = AnalyzeResponse | AnalyzeProviderError


class AnalyzeTrustScore(BaseModel):
    trust_score: float = Field(ge=0.0, le=1.0)
    confidence_level: Literal["low", "medium", "high"]


class AnalyzeApiMetrics(BaseModel):
    """Cross-provider ranking agreement (present when multiple providers are compared)."""

    overlap_score: float = Field(ge=0.0, le=1.0)
    stability_score: float = Field(ge=0.0, le=1.0)
    rank_variance: float = Field(ge=0.0, description="Mean variance of rank for products listed by multiple providers.")
    accuracy_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Catalog alignment: recall of a catalog sample against merged provider rankings.",
    )


class AnalyzeApiTrust(BaseModel):
    """Aggregate trust from stability, catalog accuracy, and overlap (present when multiple providers are compared)."""

    score: float = Field(ge=0.0, le=1.0)
    confidence: Literal["low", "medium", "high"]
    stability_score: float = Field(ge=0.0, le=1.0)
    accuracy_score: float = Field(ge=0.0, le=1.0)
    overlap_score: float = Field(ge=0.0, le=1.0)
    stability_component: float = Field(ge=0.0, le=1.0, description="0.4 × stability_score (pre-sum contribution).")
    accuracy_component: float = Field(ge=0.0, le=1.0, description="0.4 × accuracy_score (pre-sum contribution).")
    overlap_component: float = Field(ge=0.0, le=1.0, description="0.2 × overlap_score (pre-sum contribution).")


class AnalyzeApiExplanation(BaseModel):
    summary: str
    insights: list[str]


class AnalyzeApiDebug(BaseModel):
    """Parse and retry diagnostics for an analyze run (per provider when comparing models)."""

    raw_length: int = Field(ge=0, description="Character length of the provider raw model output.")
    parsed_items: int = Field(ge=0, description="Count of normalized ranked_products after parse.")
    repair_applied: bool = Field(
        description="True when an empty ranking triggered the strict-JSON retry and the retry response was used.",
    )


class AnalyzeApiResponse(BaseModel):
    """
    Unified `/v1/analyze` JSON shape: always includes the original query, which providers
    contributed, per-provider payloads, optional cross-provider metrics and trust, and
    a human-readable explanation block.
    """

    query: str
    providers_used: list[ProviderName] = Field(
        description="Providers that were invoked, in canonical order (ollama → openai → openrouter).",
    )
    results: dict[ProviderName, AnalyzeProviderResult] = Field(
        description="Per-provider outcome: success payload or structured error.",
    )
    metrics: AnalyzeApiMetrics | None = Field(
        default=None,
        description="Set when comparing multiple providers; null for a single-provider run.",
    )
    trust: AnalyzeApiTrust | None = Field(
        default=None,
        description="Aggregate trust when metrics are available; null for single-provider.",
    )
    explanation: AnalyzeApiExplanation
    debug: AnalyzeApiDebug | dict[ProviderName, AnalyzeApiDebug] = Field(
        description="Single-provider: one object. Multi-provider: map of provider name to the same fields.",
    )
    accuracy: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Multiset recall of model-ranked names vs labeled ground truth for this query (when available).",
    )
    trust_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Aggregate trust using stability/overlap from ranking comparison and ground-truth accuracy when set.",
    )


class AnalyzeComparisonResponse(BaseModel):
    results: dict[ProviderName, AnalyzeProviderResult]
    metrics: dict[str, float] | None = Field(
        default=None,
        description="Ranking agreement metrics when comparing all providers (overlap, rank variance, stability).",
    )
    trust: AnalyzeTrustScore | None = Field(
        default=None,
        description="Aggregate trust derived from comparison metrics when provider is 'all'.",
    )


class HistoryEntry(BaseModel):
    """A stored analyze request for listing."""

    id: UUID
    query: str
    trust_score: float
    created_at: datetime
    snapshot: dict[str, Any] | None = Field(
        default=None,
        description="Optional full AnalyzeResponse as JSON-compatible dict.",
    )
