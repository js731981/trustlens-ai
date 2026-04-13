from __future__ import annotations

from threading import Lock

from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None
_lock = Lock()


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed_text(text: str) -> list[float]:
    """Return a dense embedding vector for a single string."""
    model = _get_model()
    vec = model.encode(
        text,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vec.flatten().astype("float64", copy=False).tolist()
