---
title: TrustLens AI
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: gradio
short_description: LLM trust & GEO ranking demo
---

# TrustLens AI — Hugging Face Space

This folder is a **standalone Gradio demo** that mirrors the *look and feel* of a “trust and GEO” analytics panel. It is **not** wired to the main Trust Lens AI FastAPI service: everything runs in-process using a small deterministic simulator in `utils.py`.

## What you get

- **Trust Score** and **GEO Score** (0–100) with tier labels and animated bars  
- **Top-3 ranking** with example product names (insurance / loan heuristics from the query text)  
- **Short explanation** in an LLM-style voice (still rule-based)  
- **Trust / GEO trend plots** (four points ending at the current score)  
- **Session query history** (last five rows, stored in Gradio state only)

## GEO in this demo

**GEO** here means **Generative Engine Optimization**, used as a shorthand for how *specific* and *entity-grounded* the suggested products look (named plans and brands versus generic labels). The numeric GEO score is derived from heuristics over the ranked strings, not from a live search engine or LLM.

## Stack

| Piece | Role |
|--------|------|
| `app.py` | Gradio layout, CSS theme, charts (`matplotlib`), history table HTML |
| `utils.py` | `run_trustlens(query)` — deterministic-ish scores and rankings from the query |

No database, no HTTP client, and no `torch` / `transformers` in this Space.

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

1. Create a **Gradio** Space and point it at this directory (or push this repo and set the Space’s file path to `hf_space` if your Space repo layout matches).  
2. Ensure **`app.py`** is the entry file and **`requirements.txt`** is present (HF installs it automatically).  
3. Optional: adjust the YAML front matter above (`title`, `emoji`, colors) for your Space card.

## Relation to the main project

The full **Trust Lens AI** product (FastAPI analyze API, optional RAG, Streamlit UI, persistence) lives in the repository root. This Space is intentionally **lightweight** for sharing a UI prototype without API keys or GPU stacks.
