from __future__ import annotations

import os
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer


DATASET = [
    "HDFC Bank Personal Loan - low interest, fast approval",
    "ICICI Bank Personal Loan - flexible tenure options",
    "Axis Bank Personal Loan - instant disbursal",
    "Bajaj Finserv Personal Loan - minimal documentation",
    "SBI Personal Loan - trusted public sector bank",
    "HDFC Ergo Health Insurance - comprehensive coverage",
    "ICICI Lombard Health Insurance - strong hospital network",
    "Star Health Insurance - affordable plans",
]


def _payload_from_text(text: str) -> dict[str, Any]:
    name = (text.split("-", 1)[0] if "-" in text else text).strip()
    t = text.lower()
    if "insurance" in t:
        ptype = "insurance"
    elif "loan" in t:
        ptype = "loan"
    else:
        ptype = "product"
    return {"name": name, "type": ptype, "text": text, "features": [x.strip() for x in text.split("-")[1:]]}


def main() -> None:
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    collection = (os.getenv("QDRANT_COLLECTION") or os.getenv("QDRANT_COLLECTION_NAME") or "financial_products").strip() or "financial_products"

    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode(DATASET, normalize_embeddings=True).tolist()
    if not vectors:
        raise RuntimeError("No embeddings generated.")
    dim = len(vectors[0])

    client = QdrantClient(host=host, port=port)
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    points: list[PointStruct] = []
    for i, (text, vec) in enumerate(zip(DATASET, vectors, strict=True), start=1):
        points.append(PointStruct(id=i, vector=vec, payload=_payload_from_text(text)))

    client.upsert(collection_name=collection, points=points)
    print(f"Seeded Qdrant collection '{collection}' with {len(points)} financial products.")


if __name__ == "__main__":
    main()

