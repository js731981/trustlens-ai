from sentence_transformers import SentenceTransformer


class EmbeddingService:
    def __init__(self, model_name: str, *, vector_size: int | None = None) -> None:
        self._model_name = model_name
        self._model: SentenceTransformer | None = None
        self._vector_size = vector_size

    def load(self) -> None:
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)

    @property
    def vector_size(self) -> int:
        if self._model is None:
            if self._vector_size is None:
                raise RuntimeError("Embedding model not loaded and vector size not configured")
            return int(self._vector_size)
        return int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            self.load()
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()
