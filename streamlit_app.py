"""Trust Lens Streamlit UI — run with: streamlit run streamlit_app.py"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
import httpx
import plotly.graph_objects as go
import streamlit as st

DEFAULT_API_BASE = "http://127.0.0.1:8000"

load_dotenv()


def _api_base() -> str:
    return os.environ.get("TRUST_LENS_API_BASE", DEFAULT_API_BASE).rstrip("/")


def _first_ok_provider_block(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Pick the first successful provider entry from the unified analyze envelope."""
    results = payload.get("results")
    if not isinstance(results, dict):
        return None
    for key in ("ollama", "openai", "openrouter"):
        block = results.get(key)
        if isinstance(block, dict) and not block.get("error"):
            return block
    for block in results.values():
        if isinstance(block, dict) and not block.get("error"):
            return block
    return None


def _trust_gauge(trust_0_1: float) -> go.Figure:
    value_pct = trust_0_1 * 100.0
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value_pct,
            number={"suffix": "%", "valueformat": ".0f"},
            title={"text": "Trust score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1f77b4"},
                "steps": [
                    {"range": [0, 40], "color": "#f2dede"},
                    {"range": [40, 70], "color": "#fcf8e3"},
                    {"range": [70, 100], "color": "#dff0d8"},
                ],
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=24, r=24, t=48, b=24))
    return fig


def _render_insights(insights: dict[str, Any]) -> None:
    st.subheader("Explanation insights")
    features = insights.get("features") or []
    sentiment = str(insights.get("sentiment", "neutral"))
    confidence = float(insights.get("confidence") or 0.0)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Sentiment", sentiment.replace("_", " ").title())
        if sentiment == "positive":
            st.success("Tone leans constructive / favorable.")
        elif sentiment == "negative":
            st.warning("Tone leans critical or cautious — sanity-check claims.")
        else:
            st.info("Tone is largely balanced or informational.")
    with c2:
        st.metric("Insight confidence", f"{confidence:.0%}")
    with c3:
        st.markdown("**Factors highlighted in the rationale**")
        if features:
            for f in features:
                st.markdown(f"- **{f}** — called out in the explanation text")
        else:
            st.caption("No strong price / trust / coverage cues detected in the rationale.")


def _render_bias(bias: dict[str, Any]) -> None:
    st.subheader("Bias alerts")
    detected = bool(bias.get("bias_detected"))
    btype = bias.get("bias_type")
    if detected:
        label = (btype or "unspecified").replace("_", " ").title()
        st.warning(f"**Bias flagged:** {label}. Review rankings and rationale before relying on them.")
    else:
        st.success("No strong bias signals detected by the heuristic checks.")


def _render_rankings(rankings: list[dict[str, Any]]) -> None:
    st.subheader("Ranked products")
    if not rankings:
        st.info("No products returned.")
        return
    rows: list[dict[str, Any]] = []
    for p in sorted(rankings, key=lambda x: int(x.get("rank", 0))):
        rows.append(
            {
                "Rank": p.get("rank"),
                "Product": p.get("name"),
                "Notes": p.get("notes") or "",
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Trust Lens", layout="wide", initial_sidebar_state="collapsed")
    st.title("Trust Lens")
    st.caption("Connects to the Trust Lens API `/v1/analyze` endpoint (multi-LLM).")

    base = _api_base()
    with st.sidebar:
        st.text_input("API base URL", value=base, key="api_base_input", help="Override default via TRUST_LENS_API_BASE env var.")
        st.caption("Start the API: `uvicorn app.main:app --reload`")

    api_base = st.session_state.get("api_base_input", base).rstrip("/")

    query = st.text_area(
        "Query",
        height=120,
        placeholder="Describe what you want ranked and any constraints…",
        label_visibility="collapsed",
    )
    provider = st.selectbox(
        "Provider",
        options=["ollama", "openai", "openrouter"],
        index=0,
        help="Select which LLM provider the API should use.",
    )
    col_go, _ = st.columns([1, 5])
    with col_go:
        run = st.button("Analyze", type="primary", use_container_width=True)

    if run:
        q = (query or "").strip()
        if not q:
            st.warning("Enter a query first.")
            return
        try:
            with st.spinner("Running ranking + trust analysis (may take a minute)…"):
                resp = httpx.post(
                    f"{api_base}/v1/analyze",
                    json={"query": q, "provider": provider},
                    timeout=httpx.Timeout(180.0, connect=10.0),
                )
                resp.raise_for_status()
                payload: dict[str, Any] = resp.json()
        except httpx.HTTPStatusError as exc:
            st.error(f"API error **{exc.response.status_code}**: `{exc.response.text[:500]}`")
            return
        except httpx.RequestError as exc:
            st.error(f"Could not reach **{api_base}** ({exc.__class__.__name__}: {exc}). Is the API running?")
            return

        st.session_state["last_payload"] = payload
        st.session_state["last_query"] = q

    payload = st.session_state.get("last_payload")
    if not payload:
        st.info("Submit a query to see ranked products, trust score, insights, and bias alerts.")
        return

    if st.session_state.get("last_query"):
        st.caption(f"Showing results for: **{st.session_state['last_query'][:200]}{'…' if len(st.session_state['last_query']) > 200 else ''}**")

    primary = _first_ok_provider_block(payload)
    if not primary:
        st.error("No successful provider result in the API response.")
        return

    used_list = payload.get("providers_used") or []
    if isinstance(used_list, list) and len(used_list) > 1:
        st.caption(f"Providers invoked: **{', '.join(str(p) for p in used_list)}** (showing first successful ranking below)")
    else:
        used_provider = primary.get("provider_used") or primary.get("provider") or "ollama"
        fb = primary.get("fallback_used")
        fb_note = " (fallback from failed remote provider)" if fb else ""
        st.caption(f"Provider used: **{used_provider}**{fb_note}")

    metrics = payload.get("metrics")
    trust_block = payload.get("trust")
    if isinstance(metrics, dict) and isinstance(trust_block, dict):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Overlap score", f"{float(metrics.get('overlap_score', 0)):.0%}")
        with c2:
            st.metric("Stability score", f"{float(metrics.get('stability_score', 0)):.0%}")
        with c3:
            st.metric("Trust score", f"{float(trust_block.get('score', 0)):.2f}")
            st.caption(f"Confidence: **{trust_block.get('confidence', '—')}**")

    api_expl = payload.get("explanation")
    if isinstance(api_expl, dict):
        st.subheader("Cross-model analysis")
        st.markdown(api_expl.get("summary") or "_No summary._")
        for line in api_expl.get("insights") or []:
            st.markdown(f"- {line}")

    parsed = primary.get("parsed_output") or {}
    rankings = parsed.get("ranked_products") or []
    _render_rankings(rankings)

    explanation = parsed.get("explanation") or ""
    with st.expander("Model response (raw text)", expanded=False):
        st.code(primary.get("raw_output") or primary.get("raw_response") or "", language="text")

    with st.expander("Explanation (parsed)", expanded=False):
        st.markdown(explanation or "_No explanation text._")


if __name__ == "__main__":
    main()
