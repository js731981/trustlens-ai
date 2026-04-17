"""Trust Lens AI — Streamlit analytics dashboard.

Run:
  streamlit run frontend-ui/app.py
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Literal

from dotenv import load_dotenv
import altair as alt
import httpx
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="TrustLens AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()

BASE_URL = "http://127.0.0.1:8001"
DEFAULT_API_BASE = BASE_URL

Provider = Literal["ollama", "openai", "openrouter", "all"]

_AGENT_ORDER: tuple[str, ...] = ("retrieval", "ranking", "trust", "analytics", "explanation")
_AGENT_META: dict[str, dict[str, str]] = {
    "retrieval": {"label": "Retrieval", "icon": "🔍"},
    "ranking": {"label": "Ranking", "icon": "🧠"},
    "trust": {"label": "Trust", "icon": "⚖️"},
    "analytics": {"label": "Analytics", "icon": "📊"},
    "explanation": {"label": "Explanation", "icon": "🗣️"},
}


def _api_base() -> str:
    return os.environ.get("TRUST_LENS_API_BASE", DEFAULT_API_BASE).rstrip("/")


def _safe_json(resp: httpx.Response) -> dict[str, Any] | None:
    try:
        data = resp.json()
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _safe_list(resp: httpx.Response) -> list[dict[str, Any]]:
    try:
        data = resp.json()
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        return []
    except Exception:
        return []


def _fmt_pct(value: Any) -> str:
    try:
        f = float(value)
    except Exception:
        return "—"
    if f <= 1.0:
        f *= 100.0
    return f"{f:.0f}%"

def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _confidence_band(score01: float) -> str:
    if score01 >= 0.8:
        return "High"
    if score01 >= 0.5:
        return "Medium"
    return "Low"


def _is_errorish(agent_block: Any) -> bool:
    if not isinstance(agent_block, dict):
        return False
    if agent_block.get("error"):
        return True
    out = agent_block.get("output")
    return isinstance(out, dict) and bool(out.get("error"))


def _agent_duration_s(agent_block: Any) -> float | None:
    if not isinstance(agent_block, dict):
        return None
    for k in ("duration_s", "duration_sec", "elapsed_s", "time_s"):
        v = _safe_float(agent_block.get(k))
        if v is not None:
            return v
    ms = _safe_float(agent_block.get("duration_ms") or agent_block.get("elapsed_ms") or agent_block.get("time_ms"))
    if ms is not None:
        return ms / 1000.0
    return None


def _extract_agent_outputs(payload: dict[str, Any]) -> dict[str, Any] | None:
    agent_outputs = payload.get("agent_outputs")
    return agent_outputs if isinstance(agent_outputs, dict) else None


def _extract_final_output(payload: dict[str, Any]) -> dict[str, Any]:
    # New multi-agent envelope: {"final_output": {...}}
    final_output = payload.get("final_output")
    if isinstance(final_output, dict):
        return final_output

    # Legacy API envelope: choose first ok provider parsed_output
    primary = _first_ok_provider_block(payload) or {}
    parsed = primary.get("parsed_output") or {}
    return parsed if isinstance(parsed, dict) else {}


def _extract_trust_geo(payload: dict[str, Any]) -> tuple[float | None, float | None]:
    # New envelope: {"metrics": {"trust_score": float, "geo_score": float}}
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        t = _safe_float(metrics.get("trust_score"))
        g = _safe_float(metrics.get("geo_score"))
        if (t is not None) or (g is not None):
            return t, g

    # Legacy: trust_score at top-level or nested; geo at payload.geo.score
    trust_score = payload.get("trust_score")
    if trust_score is None:
        trust = payload.get("trust")
        if isinstance(trust, dict):
            trust_score = trust.get("score")
    geo_score = None
    geo = payload.get("geo")
    if isinstance(geo, dict):
        geo_score = geo.get("score")
    return _safe_float(trust_score), _safe_float(geo_score)


def _render_agent_pipeline(agent_outputs: dict[str, Any]) -> None:
    st.subheader("Agent Pipeline")
    cols = st.columns(5)
    for idx, agent in enumerate(_AGENT_ORDER):
        meta = _AGENT_META.get(agent, {"label": agent.title(), "icon": "🤖"})
        block = agent_outputs.get(agent)

        status = "loading"
        if block is None:
            status = "skipped"
        elif _is_errorish(block):
            status = "failed"
        else:
            status = "done"

        dur_s = _agent_duration_s(block)
        with cols[idx]:
            with st.container(border=True):
                st.markdown(f"**{meta['icon']} {meta['label']}**")
                if status == "done":
                    st.success("Done")
                elif status == "failed":
                    st.error("Failed")
                elif status == "skipped":
                    st.warning("Missing")
                else:
                    st.info("Loading…")
                if dur_s is not None:
                    st.caption(f"Time: **{dur_s:.2f}s**")


def _agent_preview(agent: str, block: Any) -> str:
    if block is None:
        return "—"
    if isinstance(block, dict):
        out = block.get("output", block)
        if isinstance(out, dict):
            if agent == "retrieval":
                ctx = out.get("context") or out.get("contexts") or out.get("documents") or out.get("retrieved")
                if isinstance(ctx, list) and ctx:
                    first = ctx[0]
                    if isinstance(first, dict):
                        return str(first.get("text") or first.get("content") or first.get("snippet") or "")[:160] or "Retrieved context (list)"
                    return str(first)[:160]
                if isinstance(ctx, str) and ctx.strip():
                    return ctx.strip()[:160]
            if agent == "ranking":
                ranked = out.get("ranked_products") or out.get("ranking") or out.get("rankings")
                if isinstance(ranked, list) and ranked:
                    top = ranked[0]
                    if isinstance(top, dict):
                        return f"Top: {top.get('name') or top.get('product') or top.get('title') or '—'}"
                    return f"Top: {str(top)[:120]}"
            if agent == "trust":
                t = _safe_float(out.get("trust_score") or out.get("score"))
                g = _safe_float(out.get("geo_score"))
                if t is not None and g is not None:
                    return f"trust={t:.3f}, geo={g:.3f}"
                if t is not None:
                    return f"trust={t:.3f}"
            if agent == "explanation":
                txt = out.get("explanation") or out.get("text") or out.get("reasoning") or out.get("summary")
                if isinstance(txt, str) and txt.strip():
                    return txt.strip()[:160]
            if agent == "analytics":
                for k in ("drift", "anomalies", "insights", "signals"):
                    v = out.get(k)
                    if v is not None:
                        return f"{k}: {str(v)[:140]}"
        if isinstance(out, str) and out.strip():
            return out.strip()[:160]
    return str(block)[:160]


def _render_agent_details_panel(agent_outputs: dict[str, Any]) -> None:
    with st.expander("▶ Agent Details", expanded=False):
        for agent in _AGENT_ORDER:
            meta = _AGENT_META.get(agent, {"label": agent.title(), "icon": "🤖"})
            block = agent_outputs.get(agent)
            preview = _agent_preview(agent, block)
            status = "partial" if (block is None or _is_errorish(block)) else "ok"
            with st.container(border=True):
                left, right = st.columns([3, 1])
                with left:
                    st.markdown(f"**{meta['icon']} {meta['label']}**")
                    st.caption(preview or "—")
                with right:
                    if status == "ok":
                        st.success("OK")
                    else:
                        st.warning("Partial")


def _render_agent_debug(agent_outputs: dict[str, Any]) -> None:
    st.subheader("Agent outputs (debug)")
    for agent in _AGENT_ORDER:
        meta = _AGENT_META.get(agent, {"label": agent.title(), "icon": "🤖"})
        block = agent_outputs.get(agent)
        label = f"{meta['icon']} {meta['label']} Output"
        with st.expander(label, expanded=False):
            if block is None:
                st.warning("No output for this agent (partial result).")
                continue
            if isinstance(block, dict) and block.get("error"):
                st.error(str(block.get("error")))
            out = block.get("output") if isinstance(block, dict) else block
            if agent == "explanation":
                if isinstance(out, dict):
                    txt = out.get("explanation") or out.get("text") or out.get("reasoning") or out.get("summary")
                    if isinstance(txt, str) and txt.strip():
                        st.markdown(txt)
                    else:
                        st.json(out)
                elif isinstance(out, str):
                    st.markdown(out)
                else:
                    st.json(out)
            else:
                st.json(out if out is not None else block)


def _api_error(msg: str) -> None:
    st.error("Failed to fetch data")
    st.caption(msg)


def _get_json_try(
    api_base: str,
    paths: list[str],
    *,
    params: dict[str, Any] | None = None,
    timeout_s: float = 30.0,
) -> dict[str, Any] | None:
    last_err: str | None = None
    for path in paths:
        try:
            resp = httpx.get(
                f"{api_base}{path}",
                params=params,
                timeout=httpx.Timeout(timeout_s, connect=10.0),
            )
            if resp.status_code == 404:
                last_err = f"`GET {path}` returned 404"
                continue
            resp.raise_for_status()
            return _safe_json(resp)
        except httpx.HTTPStatusError as exc:
            last_err = f"`GET {path}` failed ({exc.response.status_code}): {exc.response.text[:500]}"
        except httpx.RequestError as exc:
            last_err = f"`GET {path}` request failed: {exc}"
        except Exception as exc:
            last_err = f"`GET {path}` failed: {exc}"
    if last_err:
        _api_error(last_err)
    return None


def _get_list_try(
    api_base: str,
    paths: list[str],
    *,
    params: dict[str, Any] | None = None,
    timeout_s: float = 30.0,
) -> list[dict[str, Any]]:
    last_err: str | None = None
    for path in paths:
        try:
            resp = httpx.get(
                f"{api_base}{path}",
                params=params,
                timeout=httpx.Timeout(timeout_s, connect=10.0),
            )
            if resp.status_code == 404:
                last_err = f"`GET {path}` returned 404"
                continue
            resp.raise_for_status()
            return _safe_list(resp)
        except httpx.HTTPStatusError as exc:
            last_err = f"`GET {path}` failed ({exc.response.status_code}): {exc.response.text[:500]}"
        except httpx.RequestError as exc:
            last_err = f"`GET {path}` request failed: {exc}"
        except Exception as exc:
            last_err = f"`GET {path}` failed: {exc}"
    if last_err:
        _api_error(last_err)
    return []


def _post_json_try(
    api_base: str,
    paths: list[str],
    *,
    body: dict[str, Any],
    timeout_s: float = 180.0,
) -> dict[str, Any] | None:
    last_err: str | None = None
    for path in paths:
        try:
            resp = httpx.post(
                f"{api_base}{path}",
                json=body,
                timeout=httpx.Timeout(timeout_s, connect=10.0),
            )
            if resp.status_code == 404:
                last_err = f"`POST {path}` returned 404"
                continue
            resp.raise_for_status()
            return _safe_json(resp)
        except httpx.HTTPStatusError as exc:
            last_err = f"`POST {path}` failed ({exc.response.status_code}): {exc.response.text[:500]}"
        except httpx.RequestError as exc:
            last_err = f"`POST {path}` request failed: {exc}"
        except Exception as exc:
            last_err = f"`POST {path}` failed: {exc}"
    if last_err:
        _api_error(last_err)
    return None


def _post_json(api_base: str, path: str, *, body: dict[str, Any], timeout_s: float = 180.0) -> dict[str, Any] | None:
    try:
        resp = httpx.post(
            f"{api_base}{path}",
            json=body,
            timeout=httpx.Timeout(timeout_s, connect=10.0),
        )
        resp.raise_for_status()
        return _safe_json(resp)
    except httpx.HTTPStatusError as exc:
        st.error(f"API error **{exc.response.status_code}**: `{exc.response.text[:500]}`")
        return None
    except httpx.RequestError as exc:
        st.error(f"Could not reach **{api_base}** ({exc.__class__.__name__}: {exc}). Is the API running?")
        return None


def _get_json(api_base: str, path: str, *, params: dict[str, Any] | None = None, timeout_s: float = 30.0) -> dict[str, Any] | None:
    try:
        resp = httpx.get(
            f"{api_base}{path}",
            params=params,
            timeout=httpx.Timeout(timeout_s, connect=10.0),
        )
        resp.raise_for_status()
        return _safe_json(resp)
    except Exception:
        return None


def _get_list(api_base: str, path: str, *, params: dict[str, Any] | None = None, timeout_s: float = 30.0) -> list[dict[str, Any]]:
    try:
        resp = httpx.get(
            f"{api_base}{path}",
            params=params,
            timeout=httpx.Timeout(timeout_s, connect=10.0),
        )
        resp.raise_for_status()
        return _safe_list(resp)
    except Exception:
        return []


def _first_ok_provider_block(payload: dict[str, Any]) -> dict[str, Any] | None:
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


def _provider_block(payload: dict[str, Any], provider: str) -> dict[str, Any] | None:
    results = payload.get("results")
    if not isinstance(results, dict):
        return None
    block = results.get(provider)
    return block if isinstance(block, dict) else None


def _rankings_from_provider_block(provider_block: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(provider_block, dict):
        return []
    parsed = provider_block.get("parsed_output") or {}
    ranked = parsed.get("ranked_products") or []
    if isinstance(ranked, list):
        return [x for x in ranked if isinstance(x, dict)]
    return []


def _render_rankings(rankings: list[dict[str, Any]]) -> None:
    if not rankings:
        st.info("No ranked results returned.")
        return
    rows: list[dict[str, Any]] = []
    for item in sorted(rankings, key=lambda x: int(x.get("rank", 10**9) or 10**9)):
        rows.append(
            {
                "Rank": item.get("rank"),
                "Product": item.get("name") or item.get("product") or item.get("title"),
                "Notes": item.get("notes") or item.get("reason") or item.get("rationale") or "",
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _download_report_button(api_base: str, payload: dict[str, Any]) -> None:
    primary = _first_ok_provider_block(payload) or {}
    parsed = primary.get("parsed_output") or {}
    ranked_products = parsed.get("ranked_products") or []

    trust = payload.get("trust") or {}
    trust_score = trust.get("score")
    if trust_score is None:
        trust_score = payload.get("trust_score")

    body = {
        "query": payload.get("query") or st.session_state.get("last_query") or "",
        "ranked_products": ranked_products if isinstance(ranked_products, list) else [],
        "trust_score": trust_score,
        "explanation": payload.get("explanation") or parsed.get("explanation") or "",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    col1, col2 = st.columns([1, 3])
    with col1:
        clicked = st.button("📄 Download Report", use_container_width=True)
    with col2:
        st.caption("Generates a PDF via the API `/download-report` endpoint (if enabled).")

    if not clicked:
        return

    paths_to_try = ["/v1/download-report", "/download-report"]
    pdf_bytes: bytes | None = None
    last_err: str | None = None

    for path in paths_to_try:
        try:
            resp = httpx.post(
                f"{api_base}{path}",
                json=body,
                timeout=httpx.Timeout(60.0, connect=10.0),
            )
            if resp.status_code == 404:
                last_err = f"{path} returned 404"
                continue
            resp.raise_for_status()
            pdf_bytes = resp.content
            break
        except Exception as exc:
            last_err = f"{path} failed: {exc}"

    if not pdf_bytes:
        st.warning("Report download is not available from the backend yet.")
        if last_err:
            st.caption(f"Debug: {last_err}")
        return

    st.download_button(
        label="⬇️ Save PDF",
        data=pdf_bytes,
        file_name="trustlens_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


def _render_export_section(api_base: str, payload: dict[str, Any]) -> None:
    st.subheader("📄 Export")
    if st.button("Download PDF Report"):
        paths_to_try = ["/v1/report", "/report", "/v1/download-report", "/download-report"]
        pdf_bytes: bytes | None = None
        last_err: str | None = None
        for path in paths_to_try:
            try:
                resp = httpx.post(
                    f"{api_base}{path}",
                    json=payload,
                    timeout=httpx.Timeout(60.0, connect=10.0),
                )
                if resp.status_code == 404:
                    last_err = f"{path} returned 404"
                    continue
                resp.raise_for_status()
                pdf_bytes = resp.content
                break
            except Exception as exc:
                last_err = f"{path} failed: {exc}"

        if not pdf_bytes:
            st.error("Report generation failed")
            if last_err:
                st.caption(f"Debug: {last_err}")
            return

        try:
            with open("report.pdf", "wb") as f:
                f.write(pdf_bytes)
            st.success("Downloaded report.pdf")
        except Exception:
            # Writing might fail on some deployments; still provide in-app download.
            st.success("Report generated (use button below to save).")

        st.download_button(
            "⬇️ Save report.pdf",
            data=pdf_bytes,
            file_name="report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


def _render_dashboard_tab(api_base: str) -> None:
    st.title("Dashboard")
    data = _get_json_try(api_base, ["/metrics", "/v1/metrics"], timeout_s=10.0) or {}

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Trust", f"{float(data.get('avg_trust') or 0.0) * 100:.1f}%")
    col2.metric("Visibility", f"{float(data.get('visibility') or 0.0) * 100:.1f}%")
    col3.metric("Queries", int(data.get("queries") or 0))
    col4.metric("GEO Score", f"{float(data.get('avg_geo') or 0.0) * 100:.1f}%")

    st.markdown("---")

    st.subheader("📈 Trust Trend")
    trust_series = data.get("trust_series", [])
    if isinstance(trust_series, list) and trust_series:
        st.line_chart(pd.DataFrame(trust_series))
    else:
        st.info("No trend data available")

    st.subheader("🚀 GEO Score Trend")
    geo_series = data.get("geo_series", [])
    if isinstance(geo_series, list) and geo_series:
        st.line_chart(pd.DataFrame(geo_series))
    else:
        st.info("No GEO trend data available")

    st.subheader("🤖 Model Comparison")
    model_scores = data.get("model_scores", {})
    if isinstance(model_scores, dict) and model_scores:
        st.bar_chart(pd.DataFrame(model_scores, index=[0]))
    else:
        st.info("No model comparison data")

    # Enhance: top queries + trust distribution (from history).
    history = _get_list_try(api_base, ["/history", "/v1/history"], timeout_s=10.0)
    if history:
        df = pd.DataFrame(history)

        st.markdown("---")
        st.subheader("🔥 Top Queries")
        if "query" in df.columns:
            counts = df["query"].astype(str).value_counts().head(5)
            st.bar_chart(counts)
        else:
            st.info("History data missing `query` field.")

        st.markdown("---")
        st.subheader("📊 Trust Distribution")
        trust_col = None
        for candidate in ("trust_score", "trust", "score"):
            if candidate in df.columns:
                trust_col = candidate
                break
        if trust_col is None:
            st.info("History data missing `trust_score` field.")
        else:
            trust_vals = pd.to_numeric(df[trust_col], errors="coerce").dropna()
            if trust_vals.empty:
                st.info("No trust values available to chart.")
            else:
                # Assume 0..1 if values look like proportions.
                series = trust_vals.copy()
                if (series.max() <= 1.0) and (series.min() >= 0.0):
                    series = series * 100.0
                chart_df = pd.DataFrame({"trust": series})
                chart = (
                    alt.Chart(chart_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("trust:Q", bin=alt.Bin(maxbins=20), title="Trust (%)"),
                        y=alt.Y("count():Q", title="Count"),
                    )
                    .properties(height=240)
                )
                st.altair_chart(chart, use_container_width=True)
    else:
        st.caption("No history yet (run a few analyses to populate Dashboard extras).")


def _render_comparison_tab(payload: dict[str, Any] | None) -> None:
    st.markdown("### 🤖 Side-by-side model output")
    if not payload:
        st.info("Run an analysis with provider **all** in the Analyze tab to populate this view.")
        return

    cols = st.columns(3)
    mapping = [
        ("openai", "GPT"),
        ("openrouter", "Claude"),
        ("ollama", "Llama"),
    ]
    for (provider, label), col in zip(mapping, cols, strict=False):
        with col:
            with st.container(border=True):
                st.markdown(f"**{label}** (`{provider}`)")
                block = _provider_block(payload, provider) or {}
                if block.get("error"):
                    st.error(str(block.get("error")))
                    continue
                rankings = _rankings_from_provider_block(block)
                if rankings:
                    top = rankings[0]
                    st.caption(f"Top pick: **{top.get('name', '—')}**")
                else:
                    st.caption("No ranking returned.")
                trust = payload.get("trust") or {}
                score = trust.get("score")
                if score is None:
                    score = payload.get("trust_score")
                if score is not None:
                    st.metric("Trust score", f"{float(score):.2f}")
                _render_rankings(rankings[:10])


def _render_drift_tab(api_base: str) -> None:
    st.markdown("### 📉 Ranking drift over time")
    history = _get_list_try(api_base, ["/history", "/v1/history"])
    query_options = [str(h.get("query")) for h in history if h.get("query")]
    query_options = list(dict.fromkeys(query_options))  # de-dupe, preserve order

    if not query_options:
        st.info("No stored queries yet. Run analyses first.")
        return

    selected = st.selectbox("Select query", options=query_options, index=0)
    drift = _get_json_try(api_base, ["/drift", "/v1/drift"], params={"query": selected})
    if not drift or not isinstance(drift.get("history"), list):
        st.info("No drift data for this query yet.")
        return

    st.metric("Drift score", f"{float(drift.get('drift_score') or 0.0):.2f}")

    points = [x for x in drift["history"] if isinstance(x, dict)]
    points.sort(key=lambda x: str(x.get("timestamp") or ""))
    if not points:
        st.info("Not enough drift points to chart.")
        return

    timestamps = [str(p.get("timestamp") or "") for p in points]
    products = sorted({str(p.get("product") or "") for p in points if p.get("product")})
    if not products:
        st.info("No products found in drift history.")
        return

    series: dict[str, list[float | None]] = {prod: [None] * len(timestamps) for prod in products}
    for i, p in enumerate(points):
        prod = str(p.get("product") or "")
        if prod in series:
            try:
                series[prod][i] = float(p.get("rank"))
            except Exception:
                series[prod][i] = None

    st.caption("Line chart shows rank by run order (timestamps in table below). Lower is better (rank 1).")
    with st.container(border=True):
        st.line_chart(series)

    with st.expander("🧾 Drift points", expanded=False):
        st.dataframe(points, use_container_width=True, hide_index=True)


def _render_history_tab(api_base: str) -> None:
    st.title("History")
    history = _get_list_try(api_base, ["/history", "/v1/history"], timeout_s=10.0)
    if not history:
        st.info("No history available")
        return

    df = pd.DataFrame(history)

    # Filters
    provider_filter = st.selectbox("Provider", ["all", "ollama", "openai", "openrouter"], index=0)
    if provider_filter != "all" and "provider" in df.columns:
        df = df[df["provider"].astype(str) == provider_filter]

    search = st.text_input("Search query")
    if search and "query" in df.columns:
        df = df[df["query"].astype(str).str.contains(search, case=False, na=False)]

    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("🔁 Re-run")
    rerun_query = st.selectbox(
        "Pick a query to re-run",
        options=sorted({str(q) for q in df.get("query", pd.Series([], dtype=str)).dropna().astype(str).tolist()}),
    )
    rerun_provider: Provider = st.selectbox(
        "Provider for re-run",
        options=["ollama", "openai", "openrouter", "all"],
        index=0,
    )
    if st.button("Re-run in Analyze", use_container_width=True):
        st.session_state["prefill_query"] = rerun_query
        st.session_state["prefill_provider"] = rerun_provider
        st.session_state["auto_run_analyze"] = True
        st.rerun()


def _render_geo_insights_tab(api_base: str, query: str) -> None:
    st.subheader("🚀 GEO Insights")
    if not query:
        st.info("Run an analysis first to generate GEO insights.")
        return

    geo = _get_json_try(api_base, ["/geo", "/v1/geo"], params={"query": query})
    if not geo:
        return

    issues = geo.get("issues")
    recommendations = geo.get("recommendations")
    if not isinstance(issues, list) or not isinstance(recommendations, list):
        _api_error("`/geo` response missing `issues` and/or `recommendations` arrays.")
        return

    st.warning("Issues")
    for issue in issues:
        st.write(f"- {issue}")

    st.success("Recommendations")
    for rec in recommendations:
        st.write(f"- {rec}")


def main() -> None:
    st.title("Trust Lens AI")
    st.caption("AI analytics dashboard for ranking quality, trust, drift, and history.")

    base = _api_base()
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        api_base = st.text_input("API base URL", value=base, help="Override via TRUST_LENS_API_BASE env var. Example: http://127.0.0.1:8001")
        st.caption("API expected: FastAPI (run `uvicorn app.main:app --reload`).")
        st.divider()
        show_debug = st.toggle("🐞 Show debug", value=False)

    api_base = (api_base or base).rstrip("/")

    tabs = st.tabs(
        [
            "🔍 Analyze",
            "📊 Dashboard",
            "🤖 Comparison",
            "📉 Drift",
            "🕘 History",
            "🚀 GEO Insights",
        ]
    )
    tab_analyze, tab_dash, tab_comp, tab_drift, tab_hist, tab_geo = tabs

    if "last_payload" not in st.session_state:
        st.session_state["last_payload"] = None
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None
    if "last_query" not in st.session_state:
        st.session_state["last_query"] = ""
    if "auto_run_analyze" not in st.session_state:
        st.session_state["auto_run_analyze"] = False

    with tab_analyze:
        st.subheader("🔍 Analyze")
        with st.container(border=True):
            prefill_q = str(st.session_state.get("query") or st.session_state.get("prefill_query") or "")
            query = st.text_area(
                "Query",
                height=120,
                placeholder="Describe what you want ranked and any constraints…",
                value=prefill_q,
            )
            st.session_state["prefill_query"] = ""
            st.session_state["query"] = ""

            provider_default = str(st.session_state.get("prefill_provider") or "ollama")
            provider_idx = ["ollama", "openai", "openrouter", "all"].index(provider_default) if provider_default in ["ollama", "openai", "openrouter", "all"] else 0
            provider: Provider = st.selectbox(
                "Provider",
                options=["ollama", "openai", "openrouter", "all"],
                index=provider_idx,
                help='Choose "all" to compare models and compute cross-provider trust metrics.',
            )
            st.session_state["prefill_provider"] = ""

            col_go, col_sp = st.columns([1, 4])
            with col_go:
                run = st.button("Analyze", type="primary", use_container_width=True)
            with col_sp:
                st.caption("Shows ranked results, trust score, explanation, plus report download.")

        should_run = bool(run) or bool(st.session_state.get("auto_run_analyze"))
        if should_run:
            st.session_state["auto_run_analyze"] = False
            q = (query or "").strip()
            if not q:
                st.warning("Enter a query first.")
            else:
                with st.spinner("Running analysis…"):
                    payload = _post_json_try(api_base, ["/v1/analyze", "/analyze"], body={"query": q, "provider": provider})
                if payload:
                    st.session_state["last_payload"] = payload
                    st.session_state["last_result"] = payload
                    st.session_state["last_query"] = q
                    # FORCE REFRESH so other tabs pull latest history/metrics
                    st.rerun()

        payload = st.session_state.get("last_result")
        if not payload:
            st.info("Submit a query to see ranked results, trust score, and explanation.")
        else:
            st.caption(f"Query: **{str(payload.get('query') or st.session_state.get('last_query') or '')[:200]}**")

            agent_outputs = _extract_agent_outputs(payload)
            if agent_outputs:
                _render_agent_pipeline(agent_outputs)
                _render_agent_details_panel(agent_outputs)
                if any((_is_errorish(agent_outputs.get(a)) or agent_outputs.get(a) is None) for a in _AGENT_ORDER):
                    st.warning("Partial result: one or more agents did not return output. Final results below may be incomplete.")
                st.markdown("---")

            primary = _first_ok_provider_block(payload)
            final_output = _extract_final_output(payload)
            if primary or final_output:
                st.subheader("📊 Ranked Results")
                # Prefer new final_output ranking; fall back to legacy provider block.
                rankings: list[dict[str, Any]] = []
                ranked = final_output.get("ranked_products")
                if isinstance(ranked, list):
                    rankings = [x for x in ranked if isinstance(x, dict)]
                if not rankings and primary:
                    rankings = _rankings_from_provider_block(primary)
                _render_rankings(rankings)
            else:
                st.error("No successful result in the API response.")

                st.markdown("---")
                st.subheader("🔢 Metrics")
                trust_score, geo_score = _extract_trust_geo(payload)
                accuracy = payload.get("accuracy") or (payload.get("metrics") or {}).get("accuracy")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if trust_score is not None:
                        st.metric("Trust Score", f"{round(float(trust_score) * 100, 1)}%")
                    else:
                        st.metric("Trust Score", "N/A")
                    if show_debug:
                        st.write("DEBUG trust:", trust_score)
                with col2:
                    st.metric("Accuracy Score", _fmt_pct(accuracy))
                with col3:
                    if geo_score is not None:
                        st.metric("GEO Score", f"{round(float(geo_score) * 100, 1)}%")
                    else:
                        st.metric("GEO Score", "N/A")

                confidence_score = _safe_float(payload.get("confidence_score"))
                if confidence_score is not None:
                    st.markdown(
                        f"**Confidence Score:** {_fmt_pct(confidence_score)} "
                        f"({_confidence_band(float(confidence_score))})"
                    )
                used_fallback = payload.get("used_fallback")
                llm_valid = payload.get("llm_valid")
                used_fallback_bool = bool(used_fallback) if used_fallback is not None else (llm_valid is False)
                if used_fallback_bool:
                    st.warning("⚠️ Fallback used due to invalid LLM output. Confidence reduced.")

                st.markdown("---")
                st.subheader("🧾 Explanation")
                parsed = (primary.get("parsed_output") if isinstance(primary, dict) else None) or {}
                explanation = payload.get("explanation") or final_output.get("explanation") or parsed.get("explanation") or ""
                if explanation:
                    st.write(explanation)
                else:
                    st.info("No explanation returned.")

                st.markdown("---")
                st.subheader("🤖 Model Comparison")
                try:
                    comp = _post_json_try(
                        api_base,
                        ["/v1/comparison", "/comparison"],
                        body={"query": st.session_state.get("last_query") or q},
                        timeout_s=60.0,
                    )
                    if comp and isinstance(comp, dict):
                        # Spec shape: {model: {trust, ranking}} OR nested keys.
                        for model, data in comp.items():
                            if not isinstance(data, dict):
                                continue
                            st.markdown(f"### {str(model).upper()}")
                            st.write(f"Trust: {float(data.get('trust') or 0.0) * 100:.1f}%")
                            st.write(data.get("ranking", []))
                    else:
                        st.info("Comparison not available")
                except Exception:
                    st.info("Comparison not available")

                st.markdown("---")

                geo = _post_json_try(api_base, ["/geo", "/v1/geo"], body=payload, timeout_s=60.0)
                if not geo:
                    geo = _get_json_try(api_base, ["/geo", "/v1/geo"], params={"query": st.session_state.get("last_query")}, timeout_s=30.0)

                try:
                    geo_inline = payload.get("geo")
                    if isinstance(geo_inline, dict) and geo_inline.get("score") is not None:
                        st.metric("GEO Score", f"{float(geo_inline.get('score') or 0.0) * 100:.1f}%")
                except Exception:
                    pass

                st.subheader("🚀 GEO Insights")
                try:
                    if geo and isinstance(geo.get("issues"), list) and isinstance(geo.get("recommendations"), list):
                        st.warning("Issues")
                        for i in geo.get("issues", []):
                            st.write(f"- {i}")

                        st.success("Recommendations")
                        for r in geo.get("recommendations", []):
                            st.write(f"- {r}")
                    else:
                        st.info("GEO not available")
                except Exception:
                    st.info("GEO not available")

                st.markdown("---")
                st.subheader("📉 Ranking Drift")
                try:
                    drift = _get_json_try(api_base, ["/drift", "/v1/drift"], params={"query": st.session_state.get("last_query") or q})
                    if drift and isinstance(drift.get("history"), list):
                        st.line_chart(pd.DataFrame(drift.get("history", [])))
                    else:
                        st.info("Drift not available")
                except Exception:
                    st.info("Drift not available")

                st.markdown("---")
                _render_export_section(api_base, payload)

            if show_debug:
                if agent_outputs:
                    _render_agent_debug(agent_outputs)
                with st.expander("🐞 Debug payload", expanded=False):
                    st.json(payload)

    with tab_dash:
        _render_dashboard_tab(api_base)

    with tab_comp:
        payload = st.session_state.get("last_result")
        _render_comparison_tab(payload)

    with tab_drift:
        _render_drift_tab(api_base)

    with tab_hist:
        _render_history_tab(api_base)

    with tab_geo:
        _render_geo_insights_tab(api_base, str(st.session_state.get("last_query") or ""))


if __name__ == "__main__":
    main()

