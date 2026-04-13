from typing import Literal

from pydantic import BaseModel, Field

FeatureId = Literal["price", "trust", "coverage"]
SentimentLabel = Literal["positive", "negative", "neutral"]


class ExplanationInsightsRequest(BaseModel):
    explanation: str = Field(
        min_length=1,
        max_length=16000,
        description="Free-form explanation text to analyze.",
    )


class ExplanationInsights(BaseModel):
    features: list[FeatureId] = Field(
        description="Subset of price, trust, coverage mentioned in the text.",
    )
    sentiment: SentimentLabel = Field(
        description="Overall sentiment: positive, negative, or neutral.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Model confidence for the predicted sentiment label.",
    )
