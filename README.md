![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.11+-green)
![FastAPI](https://img.shields.io/badge/backend-FastAPI-teal)
![LLM](https://img.shields.io/badge/LLM-Ollama%20%7C%20OpenAI-orange)
![Status](https://img.shields.io/badge/status-MVP-brightgreen)


# Trust Lens AI

Trust Lens is a **FastAPI** service that asks large language models (LLMs) to rank financial products from a **catalog** (static JSON files and/or **retrieval** from an optional RAG microservice), then **parses, validates, and scores** those answers. You can run a single provider (Ollama, OpenAI, or OpenRouter) or compare all three in parallel to get **overlap**, **stability**, **catalog alignment**, and an aggregate **trust score** with a short explanation.

A **Streamlit** demo calls the analyze API and visualizes rankings and cross-model metrics.

## Features

- **Multi-provider LLM ranking** — Ollama (local), OpenAI, OpenRouter; optional parallel `"all"` comparison.
- **Catalog-grounded prompts** — Insurance vs loan intent picks `data/insurance_products.json` or `data/loan_providers.json`; when `RAG_SERVICE_BASE_URL` is reachable, `/v1/analyze` can use **RAG search hits** as the catalog and prompt context instead of the full static file list.
- **Optional local vector index** — `POST /index` (load JSON catalogs into Qdrant) and `POST /search` (embedding similarity over product text); configure Qdrant with `QDRANT_HOST` / `QDRANT_PORT` (defaults `localhost` / `6333`).
- **Robust JSON pipeline** — Parse, normalize, validate product names; optional strict-JSON retry when rankings are empty.
- **Trust metrics** — Multi-provider runs: overlap, stability, rank variance, catalog alignment accuracy, aggregate trust and confidence. The analyze response may also include **ground-truth** `accuracy` / `trust_score` when labeled data exists for the query (single- or multi-provider).
- **Persistence + analytics** — SQLite under `TRUST_LENS_DATA_DIR` (default `./data`): stores LLM responses and analyze-run snapshots. Includes lightweight query history and dashboard rollups.
- **Drift tracking** — Tracks rank agreement over time for repeated queries; exposes a drift score and series.
- **Optional DEV mock** — Set `ENV=DEV` to skip real LLM calls and return a fixed mock payload (UI and pipeline testing).

## Requirements

- **Python** 3.11+ recommended (project includes a 3.12-friendly dependency set).
- **PyTorch** and **Transformers** (for explanation-insights NLP pipelines and optional `TrustScoreMLP` training code in `app/ml`).
- **Sentence Transformers** and **Qdrant** (for `/index`, `/search`, and optional local vector workflows).
- For **Ollama**: a running Ollama server and a suitable chat model (defaults to `phi3`).
- For **OpenAI** / **OpenRouter**: API keys in the environment.

## Quick start

### 1. Clone and create a virtual environment

```powershell
cd f:\CursorWorkspace\Projects\trust-lens-ai
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env.example` to `.env` and edit values.

Important variables:

| Variable | Purpose |
|----------|---------|
| `APP_NAME` | API title in OpenAPI |
| `ENVIRONMENT` | `development` \| `staging` \| `production` (production hides `/docs` and `/redoc`) |
| `ENV` | Set to `DEV` to use the built-in mock instead of real LLMs |
| `HOST`, `PORT` | Uvicorn bind address |
| `TRUST_LENS_DATA_DIR` | SQLite and analytics directory (default `data`) |
| `OLLAMA_BASE_URL` | Default `http://127.0.0.1:11434` |
| `OPENAI_API_KEY` | Required for OpenAI provider |
| `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL` | Required for OpenRouter |
| `TRUST_LENS_API_BASE` | Streamlit: base URL of the API (default `http://127.0.0.1:8000`) |
| `RAG_SERVICE_BASE_URL` | Optional: separate **rag-service** base URL; analyze calls `POST …/search` for retrieval-augmented catalog (default `http://localhost:8002`) |
| `RAG_SEARCH_TOP_K`, `RAG_SERVICE_TIMEOUT_SECONDS` | RAG hit count and HTTP timeout for that call |
| `QDRANT_HOST`, `QDRANT_PORT` | Qdrant for in-process `/index` and `/search` on the main API (defaults `localhost`, `6333`) |

Additional LLM tuning (see `app/core/config.py`): `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, `LLM_TIMEOUT_SECONDS`, `LLM_MAX_RETRIES`, `LLM_TEMPERATURE`.

### 3. Run the API

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) (not served when `ENVIRONMENT=production`).

### 4. Run the Streamlit UI (optional)

With the API running:

```powershell
streamlit run streamlit_app.py
```

The UI posts to `{TRUST_LENS_API_BASE}/v1/analyze` with a single selected provider.

## API overview

### Under `/v1`

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/health` | Liveness and API version |
| `POST` | `/v1/analyze` | Rank products + trust analysis; body: `{ "query": "...", "provider": "ollama" \| "openai" \| "openrouter" \| "all" }` |
| `GET` | `/v1/history` | Recent query history (from local SQLite), query param `limit` |
| `GET` | `/v1/drift` | Drift score and history for a given query (query param `query`) |
| `POST` | `/v1/geo` | Geo analysis over an analyze-like payload (returns a geo score/breakdown) |
| `POST` | `/v1/comparison/competitors` | Competitor comparison for a company given a query |
| `POST` | `/v1/financial/query` | Direct financial template query (prompt registry) |
| `POST` | `/v1/financial/recommendation-bias` | Heuristic bias check vs ground-truth names |
| `POST` | `/v1/insights/explanation` | Sentiment + factor tags on free-text explanation (Hugging Face pipelines) |

### Root (same FastAPI app)

These routes are mounted **without** the `/v1` prefix (see `app/main.py`).

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/index` | Embed `data/insurance_products.json` and `data/loan_providers.json` and upsert into Qdrant collection `financial_products` |
| `POST` | `/search` | Body: `{ "query": "...", "top_k": … }` — embedding search over that collection |
| `GET` | `/metrics` | Dashboard rollups (avg trust/geo, visibility, series) |

The optional **`rag-service/`** project is a separate service with its own `/index` and `/search`; point `RAG_SERVICE_BASE_URL` at it when you want `/v1/analyze` to pull catalog context from that service instead of static files only.

Full request and response models are defined in `app/models/` and exposed in OpenAPI.

## Project layout (high level)

- `app/` — FastAPI app (`main.py`), `/v1` routes, services, prompts, ML utilities, Qdrant + embedding helpers.
- `app/services/` — LLM client implementations, trust helpers, parsing and validation, drift/history/metrics utilities.
- `data/` — Product catalogs used in prompts and `/index`.
- `rag-service/` — Optional standalone FastAPI + Qdrant RAG API (`POST /search` consumed by the main app’s analyze flow).
- `streamlit_app.py` — Demo dashboard (root of repo).
- `frontend-ui/` — Additional Streamlit copy / layout variant (if present in your checkout).

For architecture, data flow, trust formulas, and extension points, see **[PROJECTOVERVIEW.md](PROJECTOVERVIEW.md)**.

## Version

API version string: `1.0.0` (`app/utils/version.py`).

## Author

Jayendran Subramanian  
Full Stack Data Engineer | AI Engineer

Passionate about building AI-driven systems for real-world financial intelligence and decisioning.

## Disclaimer

This project is a proof-of-concept AI system built for educational and research purposes only.

- It does NOT provide financial advice, recommendations, or guarantees.
- The outputs generated by LLMs may contain inaccuracies or hallucinations.
- Users should not rely on this system for real-world financial decision-making.
- Always consult certified financial advisors before making investment or insurance decisions.

The author is not responsible for any financial losses or decisions made based on this system.