from __future__ import annotations

import re
import threading
from typing import Any

from app.models.insights import ExplanationInsights

_lock = threading.Lock()
_sentiment_pipe: Any = None
_zero_shot_pipe: Any = None

_SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
_ZERO_SHOT_MODEL = "typeform/distilbert-base-uncased-mnli"

_FEATURE_KEYS = ("price", "trust", "coverage")
_CANDIDATE_LABELS = (
    "topics about price, premiums, fees, cost, affordability, or value for money",
    "topics about trust, reputation, brand strength, reviews, ratings, or reliability",
    "topics about coverage, benefits, policy limits, exclusions, or what is protected",
)
_FEATURE_THRESHOLD = 0.32
_MAX_SENTENCES_FOR_FACTORS = 10


def _get_sentiment_pipe() -> Any:
    global _sentiment_pipe
    if _sentiment_pipe is None:
        from transformers import pipeline

        _sentiment_pipe = pipeline(
            task="sentiment-analysis",
            model=_SENTIMENT_MODEL,
        )
    return _sentiment_pipe


def _get_zero_shot_pipe() -> Any:
    global _zero_shot_pipe
    if _zero_shot_pipe is None:
        from transformers import pipeline

        _zero_shot_pipe = pipeline(
            "zero-shot-classification",
            model=_ZERO_SHOT_MODEL,
        )
    return _zero_shot_pipe


def _split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    if parts:
        return parts
    return [text.strip()] if text.strip() else []


def _aggregate_feature_scores(sentences: list[str], clf: Any) -> dict[str, float]:
    """Pool zero-shot scores across sentences so distinct factors surface."""
    agg: dict[str, float] = {k: 0.0 for k in _FEATURE_KEYS}
    for sent in sentences[:_MAX_SENTENCES_FOR_FACTORS]:
        zs = clf(
            sent[:512],
            candidate_labels=list(_CANDIDATE_LABELS),
            multi_label=True,
            hypothesis_template="This text is about {}.",
        )
        labels_order: list[str] = list(zs["labels"])
        scores_map = dict(zip(labels_order, zs["scores"], strict=True))
        for key, cand in zip(_FEATURE_KEYS, _CANDIDATE_LABELS, strict=True):
            agg[key] = max(agg[key], float(scores_map.get(cand, 0.0)))
    return agg


def _normalize_sentiment_label(raw: str) -> str:
    label = raw.strip().lower()
    if label in ("pos", "positive", "label_2"):
        return "positive"
    if label in ("neg", "negative", "label_0"):
        return "negative"
    if label in ("neu", "neutral", "label_1"):
        return "neutral"
    if "positive" in label:
        return "positive"
    if "negative" in label:
        return "negative"
    if "neutral" in label:
        return "neutral"
    return "neutral"


def analyze_explanation(explanation: str) -> ExplanationInsights:
    """
    Use Hugging Face transformers for overall sentiment and zero-shot NLI
    over sentences to infer which of price / trust / coverage are discussed
    (key factors are pooled across sentence-level signals).
    """
    text = explanation.strip()
    if not text:
        return ExplanationInsights(features=[], sentiment="neutral", confidence=0.0)

    sentences = _split_sentences(text)

    with _lock:
        sentiment_pipe = _get_sentiment_pipe()
        zs_pipe = _get_zero_shot_pipe()
        sentiment_out = sentiment_pipe(text[:512], truncation=True)[0]
        if len(sentences) <= 1:
            zs = zs_pipe(
                text[:2000],
                candidate_labels=list(_CANDIDATE_LABELS),
                multi_label=True,
                hypothesis_template="This text is about {}.",
            )
            labels_order: list[str] = list(zs["labels"])
            scores_map = dict(zip(labels_order, zs["scores"], strict=True))
            feature_scores = {
                key: float(scores_map.get(cand, 0.0))
                for key, cand in zip(_FEATURE_KEYS, _CANDIDATE_LABELS, strict=True)
            }
        else:
            feature_scores = _aggregate_feature_scores(sentences, zs_pipe)

    sentiment = _normalize_sentiment_label(str(sentiment_out["label"]))
    confidence = float(sentiment_out["score"])

    features = [k for k, v in feature_scores.items() if v >= _FEATURE_THRESHOLD]

    return ExplanationInsights(
        features=features,
        sentiment=sentiment,
        confidence=confidence,
    )
