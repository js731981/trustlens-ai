---
title: TrustLens AI
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: gradio
short_description: Simulated multi-agent trust & GEO ranking demo
---

# TrustLens AI — Hugging Face Space

This folder is a **standalone Gradio Space** that showcases a **simulated multi-agent orchestration** flow (CrewAI-style): retrieval → ranking → trust → GEO → explanation. It mirrors the *feel* of a real decision-intelligence panel but is **fully in-process**—no external APIs, databases, or calls to the main TrustLens FastAPI app.

Core simulation lives in **`utils.py`** (`simulate_agents`); the UI, streaming trace, and session charts are in **`app.py`**.

## What you get

- **Simulated agent pipeline** — deterministic (query-seeded) products, scores, and structured explanation  
- **Live execution trace** — step-by-step agent messages; **Analyze** streams updates with short delays between steps  
- **Trust Score** and **GEO Score** (0–100 bars) plus **explainable breakdowns** (brand / relevance / data confidence; region / availability)  
- **Confidence score** (0–100) with **High / Medium / Low** tiers; reduced when fallback paths, simulated failures, or sparse results apply  
- **Ranked products (3–5)** with short per-item notes (insurance / loan heuristics from query text)  
- **Session trend charts** — last four trust and GEO values (random baselines on first run, then your runs)  
- **Session query history** — last five queries with trust and GEO percentages (Gradio state only)  
- **“Simulate Agent Failure”** toggle — randomly degrades one agent step, shows fallback messaging, still returns partial results  
- **Demo mode badge** — visible label that execution is simulated (tooltip on the badge)

## GEO in this demo

**GEO** is used as shorthand for how *specific* and *entity-grounded* the suggested products look (named plans and brands versus generic labels). The numeric GEO score comes from heuristics over the query and ranked strings, not from a live search engine or LLM.

## Stack

| Piece | Role |
|--------|------|
| `app.py` | Gradio layout, CSS theme, matplotlib trends, history HTML, trace streaming, loading UX |
| `utils.py` | `simulate_agents(query, simulate_failure=...)` — candidates, ranking, scores, trace, confidence, debug payloads; `run_trustlens` wraps it for older call shapes |

Dependencies: **`gradio`** and **`matplotlib`** only (see `requirements.txt`). No `torch` / `transformers` in this Space.

## Run locally

```bash
cd hf_space
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
python app.py
```

Use **Python 3.10+** (3.11 or 3.12 recommended). The app calls `demo.launch()`; open the printed local URL in your browser.

## Publish on Hugging Face

1. Create a **Gradio** Space and point it at this directory (or push this repo and set the Space’s path to `hf_space` if your repo layout matches).  
2. Ensure **`app.py`** is the entry file and **`requirements.txt`** is present (HF installs it automatically).  
3. Optional: edit the YAML front matter above (`title`, `emoji`, colors) for your Space card.

## Relation to the main project

The full **Trust Lens AI** product (FastAPI analyze API, optional RAG, Streamlit UI, persistence) lives in the repository root. This Space stays **lightweight** for sharing a high-quality **product demo** of multi-agent trust scoring and decision intelligence—without API keys or GPU stacks.

## Full project repository

For the complete implementation (backend, RAG pipeline, services, and advanced features):

https://github.com/js731981/trustlens-ai
