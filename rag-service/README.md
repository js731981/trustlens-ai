# RAG Service

Independent FastAPI app for document indexing and semantic search using **Qdrant** and **sentence-transformers**.

## Setup

```powershell
cd f:\CursorWorkspace\Projects\trust-lens-ai\rag-service
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

The example `.env` uses **`QDRANT_URL=:memory:`** (no separate Qdrant process; vectors are cleared when the app exits). For a persistent server, start **Qdrant** and set `QDRANT_URL=http://127.0.0.1:6333` in `.env`:

```powershell
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

## Run (port 8002)

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8002
```

Open [http://127.0.0.1:8002/docs](http://127.0.0.1:8002/docs) when `ENVIRONMENT` is not `production`.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/index` | Embed and upsert documents into Qdrant |
| `POST` | `/search` | Embed the query and return nearest neighbors |
| `GET` | `/health` | Liveness |

### Example bodies

**`POST /index`**

```json
{
  "documents": [
    { "id": "doc-1", "text": "Qdrant is a vector database.", "metadata": { "topic": "infra" } },
    { "text": "sentence-transformers produces dense embeddings." }
  ]
}
```

**`POST /search`**

```json
{
  "query": "What stores vectors?",
  "limit": 5
}
```

## Layout

- `app/main.py` — FastAPI app and routes
- `app/services/` — embeddings and Qdrant store
- `app/models/` — Pydantic request/response models
- `app/utils/` — settings
