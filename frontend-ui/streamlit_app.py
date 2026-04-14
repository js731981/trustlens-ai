"""TrustLens AI — Streamlit UI. Run: streamlit run streamlit_app.py"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import httpx
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

DEFAULT_API_BASE = "http://localhost:8001"


def _api_base() -> str:
    return os.environ.get("TRUST_LENS_API_BASE", DEFAULT_API_BASE).rstrip("/")


def _first_ok_result(results: dict[str, Any] | None) -> dict[str, Any] | None:
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


def _ranked_rows(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    ranked = parsed.get("ranked_products") or []
    if not isinstance(ranked, list):
        return []
    rows: list[dict[str, Any]] = []
    for p in sorted(ranked, key=lambda x: int(x.get("rank", 0) or 0)):
        rows.append(
            {
                "Rank": p.get("rank"),
                "Product": p.get("name"),
                "Notes": (p.get("notes") or "") if isinstance(p, dict) else "",
            }
        )
    return rows


def _format_metric(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{float(value):.1%}"


@st.cache_data(ttl=10, show_spinner=False)
def _fetch_metrics(api_base: str) -> dict[str, Any] | None:
    try:
        resp = httpx.get(
            f"{api_base}/metrics",
            timeout=httpx.Timeout(20.0, connect=10.0),
            headers={"Cache-Control": "no-cache"},
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, dict) else None
    except Exception:
        return None


@st.cache_data(ttl=5, show_spinner=False)
def _fetch_history(api_base: str) -> list[dict[str, Any]]:
    try:
        resp = httpx.get(
            f"{api_base}/history",
            timeout=httpx.Timeout(20.0, connect=10.0),
            headers={"Cache-Control": "no-cache"},
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        return []
    except Exception:
        return []


def _coerce_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        # Accept both ISO and "YYYY-MM-DD HH:MM:SS" from sqlite.
        s = str(value).replace("Z", "+00:00")
        dt = pd.to_datetime(s, errors="coerce", utc=True)
        if pd.isna(dt):
            return None
        return dt.to_pydatetime()
    except Exception:
        return None


def _render_dashboard(api_base: str) -> None:
    st.title("Dashboard")
    st.caption(f"GET `{api_base}/metrics` and `{api_base}/history`")

    metrics = _fetch_metrics(api_base) or {}
    avg_trust = metrics.get("avg_trust")
    visibility = metrics.get("visibility")
    queries = metrics.get("queries")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Avg trust", f"{float(avg_trust or 0.0):.2f}")
    with c2:
        st.metric("Visibility", _format_metric(float(visibility) if visibility is not None else None))
    with c3:
        st.metric("Queries", f"{int(queries or 0)}")

    history = _fetch_history(api_base)
    if not history:
        st.info("No history yet. Run a few analyses to populate charts.")
        return

    df = pd.DataFrame(history)
    # normalize columns
    if "timestamp" not in df.columns and "created_at" in df.columns:
        df["timestamp"] = df["created_at"]
    if "timestamp" in df.columns:
        df["timestamp_dt"] = df["timestamp"].apply(_coerce_ts)
        df = df[df["timestamp_dt"].notna()].copy()
        df["timestamp_dt"] = pd.to_datetime(df["timestamp_dt"], utc=True)

    if "trust_score" in df.columns:
        df["trust_score"] = pd.to_numeric(df["trust_score"], errors="coerce")

    st.markdown("### Trust trend")
    trend = (
        df.dropna(subset=["timestamp_dt"])
        .sort_values("timestamp_dt")
        .loc[:, ["timestamp_dt", "trust_score"]]
        .dropna(subset=["trust_score"])
    )
    if trend.empty:
        st.caption("Not enough data to chart trust trend yet.")
    else:
        trend = trend.rename(columns={"timestamp_dt": "timestamp"})
        st.line_chart(trend.set_index("timestamp")["trust_score"])

    st.markdown("### Recent queries")
    cols = [c for c in ["id", "query", "provider", "trust_score", "timestamp", "created_at"] if c in df.columns]
    st.dataframe(df.sort_values("timestamp_dt", ascending=False)[cols].head(25), use_container_width=True, hide_index=True)


def _render_history(api_base: str) -> None:
    st.title("History")
    st.caption(f"GET `{api_base}/history`")

    history = _fetch_history(api_base)
    if not history:
        st.info("No saved queries yet.")
        return

    df = pd.DataFrame(history)
    if "timestamp" not in df.columns and "created_at" in df.columns:
        df["timestamp"] = df["created_at"]
    if "trust_score" in df.columns:
        df["trust_score"] = pd.to_numeric(df["trust_score"], errors="coerce")

    # Quick filters
    with st.sidebar:
        st.markdown("### History filters")
        provider = st.selectbox(
            "Provider",
            options=["all"] + sorted([str(x) for x in df.get("provider", pd.Series(dtype=str)).dropna().unique().tolist()]),
            index=0,
        )
        q_contains = st.text_input("Query contains", value="")

    view = df.copy()
    if provider != "all" and "provider" in view.columns:
        view = view[view["provider"].astype(str) == provider]
    if q_contains and "query" in view.columns:
        view = view[view["query"].astype(str).str.contains(q_contains, case=False, na=False)]

    cols = [c for c in ["id", "query", "provider", "trust_score", "timestamp", "created_at"] if c in view.columns]
    st.dataframe(view[cols], use_container_width=True, hide_index=True)

    st.markdown("### Re-run a past query")
    # Make a compact chooser instead of dozens of buttons.
    q_options = (
        view["query"].dropna().astype(str).head(100).tolist()
        if "query" in view.columns
        else []
    )
    q_pick = st.selectbox("Pick query", options=q_options, index=0 if q_options else None)
    if st.button("Use this query in Analyze", type="secondary", disabled=not bool(q_pick)):
        st.session_state["analyze_query_prefill"] = q_pick
        st.session_state["nav_page"] = "Analyze"
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="TrustLens AI", layout="wide")
    st.sidebar.title("TrustLens AI")
    if "nav_page" not in st.session_state:
        st.session_state["nav_page"] = "Analyze"

    page = st.sidebar.radio(
        "Menu",
        options=["Analyze", "Dashboard", "History"],
        index=["Analyze", "Dashboard", "History"].index(st.session_state["nav_page"]),
    )
    st.session_state["nav_page"] = page

    if page == "Dashboard":
        _render_dashboard(_api_base())
        return

    if page == "History":
        _render_history(_api_base())
        return

    st.title("Analyze")
    st.caption(f"POST `{_api_base()}/v1/analyze`")

    prefill = str(st.session_state.pop("analyze_query_prefill", "") or "")
    query = st.text_input(
        "Query",
        placeholder="What should we rank and analyze?",
        label_visibility="visible",
        value=prefill,
    )
    provider = st.selectbox(
        "Provider",
        options=["ollama", "openai", "all"],
        help="Backend runs one provider or compares all configured models when set to all.",
    )
    show_debug = st.checkbox("Show debug info", value=False)

    if st.button("Analyze", type="primary"):
        q = (query or "").strip()
        if not q:
            st.warning("Enter a query first.")
        else:
            try:
                with st.spinner("Calling analyze API…"):
                    r = httpx.post(
                        f"{_api_base()}/v1/analyze",
                        json={"query": q, "provider": provider},
                        timeout=httpx.Timeout(180.0, connect=15.0),
                    )
                    r.raise_for_status()
                    st.session_state["analyze_payload"] = r.json()
                    st.session_state["analyze_query"] = q
            except httpx.HTTPStatusError as e:
                st.error(f"API returned {e.response.status_code}: {e.response.text[:800]}")
                st.session_state.pop("analyze_payload", None)
            except httpx.RequestError as e:
                base = _api_base()
                st.error(
                    f"Could not reach API at `{base}` ({e.__class__.__name__}: {e}). "
                    f"Start the API (e.g. `uvicorn app.main:app --port …`) and ensure `TRUST_LENS_API_BASE` matches that port."
                )
                st.session_state.pop("analyze_payload", None)

    payload = st.session_state.get("analyze_payload")
    if not payload:
        st.info("Enter a query and click **Analyze** to see rankings and scores.")
        return

    if st.session_state.get("analyze_query"):
        st.caption(f"Query: **{st.session_state['analyze_query'][:300]}{'…' if len(st.session_state['analyze_query']) > 300 else ''}**")

    primary = _first_ok_result(payload.get("results"))
    if not primary:
        st.error("No successful provider result in the response.")
        errs = payload.get("results") or {}
        if isinstance(errs, dict):
            for k, v in errs.items():
                if isinstance(v, dict) and v.get("error"):
                    st.text(f"{k}: {v.get('error')}")
        return

    parsed = primary.get("parsed_output") or {}
    if not isinstance(parsed, dict):
        parsed = {}

    rows = _ranked_rows(parsed)
    st.subheader("Ranked products")
    if rows:
        st.table(pd.DataFrame(rows))
    else:
        st.info("No ranked products in the parsed response.")

    metrics_block = payload.get("metrics")
    trust_block = payload.get("trust")
    top_trust = payload.get("trust_score")
    top_accuracy = payload.get("accuracy")

    if top_trust is None and isinstance(trust_block, dict):
        top_trust = trust_block.get("score")
    if top_accuracy is None and isinstance(metrics_block, dict):
        top_accuracy = metrics_block.get("accuracy_score")
    if top_accuracy is None and isinstance(trust_block, dict):
        top_accuracy = trust_block.get("accuracy_score")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Trust score", _format_metric(float(top_trust) if top_trust is not None else None))
    with c2:
        st.metric("Accuracy score", _format_metric(float(top_accuracy) if top_accuracy is not None else None))

    st.subheader("Explanation")
    expl = payload.get("explanation")
    if isinstance(expl, dict):
        st.markdown(expl.get("summary") or "_No summary._")
        for line in expl.get("insights") or []:
            st.markdown(f"- {line}")
    else:
        st.markdown(str(parsed.get("explanation") or "_No explanation._"))

    if show_debug:
        st.subheader("Debug")
        dbg = payload.get("debug")
        st.code(json.dumps(dbg, indent=2, default=str) if dbg is not None else "null", language="json")

        with st.expander("Raw primary provider JSON"):
            st.code(json.dumps(primary, indent=2, default=str), language="json")


if __name__ == "__main__":
    main()
