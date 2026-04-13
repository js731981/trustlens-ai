![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![FastAPI](https://img.shields.io/badge/backend-FastAPI-teal)
![LLM](https://img.shields.io/badge/LLM-Ollama%20%7C%20OpenAI-orange)
![Status](https://img.shields.io/badge/status-MVP-brightgreen)


# Trust Lens AI

Trust Lens is a **FastAPI** service that asks large language models (LLMs) to rank financial products from a fixed catalog, then **parses, validates, and scores** those answers. You can run a single provider (Ollama, OpenAI, or OpenRouter) or compare all three in parallel to get **overlap**, **stability**, and an aggregate **trust score** with a short explanation.

A **Streamlit** demo calls the analyze API and visualizes rankings and cross-model metrics.

## Features

- **Multi-provider LLM ranking** — Ollama (local), OpenAI, OpenRouter; optional parallel `"all"` comparison.
- **Catalog-grounded prompts** — Insurance vs loan intent picks `data/insurance_products.json` or `data/loan_providers.json`.
- **Robust JSON pipeline** — Parse, normalize, validate product names; optional strict-JSON retry when rankings are empty.
- **Trust metrics** — When comparing providers: ranking overlap, stability, derived trust score and confidence band.
- **Persistence** — SQLite under `TRUST_LENS_DATA_DIR` (default `./data`): `llm_responses` table logs each LLM call.
- **Optional DEV mock** — Set `ENV=DEV` to skip real LLM calls and return a fixed mock payload (UI and pipeline testing).

## Requirements

- **Python** 3.11+ recommended (project includes a 3.12-friendly dependency set).
- **PyTorch** and **Transformers** (for explanation-insights NLP pipelines and optional `TrustScoreMLP` training code in `app/ml`).
- For **Ollama**: a running Ollama server and a suitable chat model (defaults try `phi3`, with `mistral` as fallback in code).
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

## API overview (prefix `/v1`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness and API version |
| `POST` | `/analyze` | Rank products + trust analysis; body: `{ "query": "...", "provider": "ollama" \| "openai" \| "openrouter" \| "all" }` |
| `GET` | `/history` | List stored analyze runs from SQLite (see project overview for schema notes) |
| `POST` | `/financial/query` | Direct financial template query (prompt registry) |
| `POST` | `/financial/recommendation-bias` | Heuristic bias check vs ground-truth names |
| `POST` | `/insights/explanation` | Sentiment + factor tags on free-text explanation (Hugging Face pipelines) |

Full request and response models are defined in `app/models/` and exposed in OpenAPI.

## Project layout (high level)

- `app/` — FastAPI app (`main.py`), routes, services, prompts, ML utilities.
- `services/` — LLM client implementations (`ollama`, `openai`, `openrouter`), trust helpers, parsing and validation.
- `data/` — Product catalogs used in prompts.
- `streamlit_app.py` — Demo dashboard.

For architecture, data flow, trust formulas, and extension points, see **[PROJECTOVERVIEW.md](PROJECTOVERVIEW.md)**.

## Version

API version string: `1.0.0` (`app/utils/version.py`).

Update README.md to include an Author section at the bottom.

Include:

## Author

Jayendran Subramanian  
Full Stack Data Engineer | AI Engineer  

Short line:
"Passionate about building AI-driven systems for real-world financial intelligence and decisioning."

Add a Disclaimer section to README.md.

Include:

## Disclaimer

This project is a proof-of-concept AI system built for educational and research purposes only.

- It does NOT provide financial advice, recommendations, or guarantees.
- The outputs generated by LLMs may contain inaccuracies or hallucinations.
- Users should not rely on this system for real-world financial decision-making.
- Always consult certified financial advisors before making investment or insurance decisions.

The author is not responsible for any financial losses or decisions made based on this system.