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

_AGENT_ORDER: tuple[str, ...] = ("retrieval", "ranking", "trust", "geo", "explanation")
_AGENT_META: dict[str, dict[str, str]] = {
    "retrieval": {"label": "Retrieval Agent", "icon": "🔍"},
    "ranking": {"label": "Ranking Agent", "icon": "🧠"},
    "trust": {"label": "Trust Agent", "icon": "⚖️"},
    "geo": {"label": "GEO Agent", "icon": "🗺️"},
    "explanation": {"label": "Explanation Agent", "icon": "🗣️"},
}


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


def _extract_agent_outputs(payload: dict[str, Any]) -> dict[str, Any] | None:
    agent_outputs = payload.get("agent_outputs")
    return agent_outputs if isinstance(agent_outputs, dict) else None


def _extract_final_output(payload: dict[str, Any]) -> dict[str, Any]:
    final_output = payload.get("final_output")
    if isinstance(final_output, dict):
        return final_output
    primary = _first_ok_result(payload.get("results")) or {}
    parsed = primary.get("parsed_output") or {}
    return parsed if isinstance(parsed, dict) else {}


def _extract_trust_geo(payload: dict[str, Any]) -> tuple[float | None, float | None]:
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        t = _safe_float(metrics.get("trust_score"))
        g = _safe_float(metrics.get("geo_score"))
        if (t is not None) or (g is not None):
            return t, g
    trust_score = payload.get("trust_score")
    trust = payload.get("trust")
    if trust_score is None and isinstance(trust, dict):
        trust_score = trust.get("score")
    geo_score = None
    geo = payload.get("geo")
    if isinstance(geo, dict):
        geo_score = geo.get("score")
    return _safe_float(trust_score), _safe_float(geo_score)


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


def _debug_status_for_block(value: Any) -> tuple[str, str]:
    """
    Returns (emoji, label) for a debug panel section.

    - 🟢 Success: dict with no obvious error marker
    - 🔴 Failed: dict contains an error marker
    - 🟡 Partial: missing/empty/unknown shape
    """
    if value is None:
        return "🟡", "Partial"
    if isinstance(value, dict):
        if value.get("error"):
            return "🔴", "Failed"
        out = value.get("output")
        if isinstance(out, dict) and out.get("error"):
            return "🔴", "Failed"
        if len(value) == 0:
            return "🟡", "Partial"
        return "🟢", "Success"
    if isinstance(value, (list, str, int, float, bool)):
        return "🟢", "Success"
    return "🟡", "Partial"


def _debug_json_view(value: Any) -> None:
    if value is None:
        st.caption("—")
        return
    if isinstance(value, str):
        st.code(value.strip() or "—")
        return
    st.json(value)


def _debug_section_payload(
    *,
    agent_key: str,
    payload: dict[str, Any],
    agent_outputs: dict[str, Any] | None,
) -> Any:
    """
    Best-effort: provide the most useful JSON payload for each panel.
    """
    if agent_key == "geo":
        # GEO is computed server-side; it may also appear in payload["debug"]["geo"].
        dbg = payload.get("debug")
        if isinstance(dbg, dict) and isinstance(dbg.get("geo"), (dict, list, str)):
            return dbg.get("geo")
        return payload.get("geo")

    if not isinstance(agent_outputs, dict):
        return None
    block = agent_outputs.get(agent_key)
    if not isinstance(block, dict):
        return block
    return block.get("output") if "output" in block else block


def _render_debug_panel(payload: dict[str, Any], agent_outputs: dict[str, Any] | None) -> None:
    st.subheader("Debug Panel")
    st.caption("Accordion view of per-agent intermediate outputs.")

    for agent in _AGENT_ORDER:
        meta = _AGENT_META.get(agent, {"label": agent.title(), "icon": "🤖"})
        section_payload = _debug_section_payload(agent_key=agent, payload=payload, agent_outputs=agent_outputs)
        dot, status_label = _debug_status_for_block(section_payload)
        title = f"{dot} {meta['icon']} {meta['label']}"
        with st.expander(title, expanded=False):
            st.caption(f"Status: **{status_label}**")
            _debug_json_view(section_payload)

    with st.expander("Raw response (advanced)", expanded=False):
        primary = _first_ok_result(payload.get("results"))
        st.markdown("**Top-level debug**")
        dbg = payload.get("debug")
        st.code(json.dumps(dbg, indent=2, default=str) if dbg is not None else "null", language="json")
        st.markdown("**Raw primary provider JSON**")
        st.code(json.dumps(primary, indent=2, default=str) if primary is not None else "null", language="json")


def _render_agent_pipeline(agent_outputs: dict[str, Any]) -> None:
    st.subheader("Agent Pipeline")
    cols = st.columns(5)
    for idx, agent in enumerate(_AGENT_ORDER):
        meta = _AGENT_META.get(agent, {"label": agent.title(), "icon": "🤖"})
        block = agent_outputs.get(agent) if agent != "geo" else None
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


def _render_agent_details(agent_outputs: dict[str, Any]) -> None:
    with st.expander("▶ Agent Details", expanded=False):
        for agent in _AGENT_ORDER:
            meta = _AGENT_META.get(agent, {"label": agent.title(), "icon": "🤖"})
            block = agent_outputs.get(agent) if agent != "geo" else None
            ok = (block is not None) and (not _is_errorish(block))
            with st.container(border=True):
                st.markdown(f"**{meta['icon']} {meta['label']}**")
                if ok:
                    st.success("OK")
                else:
                    st.warning("Partial")
                out = block.get("output") if isinstance(block, dict) else block
                if agent == "explanation" and isinstance(out, dict):
                    txt = out.get("explanation") or out.get("text") or out.get("reasoning") or out.get("summary")
                    if isinstance(txt, str) and txt.strip():
                        st.caption(txt.strip()[:240])
                        continue
                st.caption(str(out)[:240] if out is not None else "—")


def _titlecase_agent(agent_key: str) -> str:
    a = (agent_key or "").strip()
    if not a:
        return "Agent"
    if a.lower() == "geo":
        return "GEO"
    return a[:1].upper() + a[1:].lower()


def _render_execution_timeline(trace: Any) -> None:
    if not isinstance(trace, list) or not trace:
        return
    st.subheader("Execution timeline")
    st.caption("Step-by-step agent execution with per-step latency.")
    for item in trace:
        if not isinstance(item, dict):
            continue
        agent = _titlecase_agent(str(item.get("agent") or ""))
        status = str(item.get("status") or "success").strip().lower()
        latency_ms = item.get("latency_ms")
        try:
            ms = int(latency_ms) if latency_ms is not None else 0
        except Exception:
            ms = 0

        label = f"{agent} ({ms}ms)"
        if status == "failed":
            st.error(f"✖ {label}")
        elif status == "fallback":
            st.warning(f"⚠ {label}")
        else:
            st.success(f"✔ {label}")


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
    show_debug = st.checkbox("Show Debug Info", value=False)
    simulate_failure: dict[str, bool] | None = None
    if show_debug:
        st.markdown("### Simulate failures")
        st.caption("Demo-only: force specific agent step failures to show graceful degradation.")
        c1, c2, c3 = st.columns(3)
        with c1:
            fail_retrieval = st.checkbox("Fail Retrieval", value=False)
        with c2:
            fail_ranking = st.checkbox("Fail Ranking", value=False)
        with c3:
            fail_trust = st.checkbox("Fail Trust", value=False)
        if fail_retrieval or fail_ranking or fail_trust:
            simulate_failure = {
                "retrieval": bool(fail_retrieval),
                "ranking": bool(fail_ranking),
                "trust": bool(fail_trust),
            }

    if st.button("Analyze", type="primary"):
        q = (query or "").strip()
        if not q:
            st.warning("Enter a query first.")
        else:
            try:
                with st.spinner("Calling analyze API…"):
                    req: dict[str, Any] = {"query": q, "provider": provider, "show_debug": bool(show_debug)}
                    if simulate_failure is not None:
                        req["simulate_failure"] = simulate_failure
                    r = httpx.post(
                        f"{_api_base()}/v1/analyze",
                        json=req,
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

    warnings = payload.get("warnings")
    if isinstance(warnings, list) and warnings:
        for w in warnings[:8]:
            if str(w).strip():
                st.warning(str(w))

    primary = _first_ok_result(payload.get("results"))
    final_output = _extract_final_output(payload)
    agent_outputs = _extract_agent_outputs(payload)
    _render_execution_timeline(payload.get("trace"))
    if agent_outputs:
        _render_agent_pipeline(agent_outputs)
        _render_agent_details(agent_outputs)
        if any((_is_errorish(agent_outputs.get(a)) or agent_outputs.get(a) is None) for a in _AGENT_ORDER):
            st.warning("Partial result: one or more agents did not return output.")

    if not primary and not final_output:
        st.error("No successful provider result in the response.")
        errs = payload.get("results") or {}
        if isinstance(errs, dict):
            for k, v in errs.items():
                if isinstance(v, dict) and v.get("error"):
                    st.text(f"{k}: {v.get('error')}")
        return

    parsed = primary.get("parsed_output") if isinstance(primary, dict) else None
    if not isinstance(parsed, dict):
        parsed = {}

    rows = _ranked_rows(parsed)
    if not rows:
        # Prefer final_output ranking if present
        ranked = final_output.get("ranked_products")
        if isinstance(ranked, list):
            rows = _ranked_rows({"ranked_products": ranked})
    st.subheader("Ranked products")
    if rows:
        st.table(pd.DataFrame(rows))
    else:
        st.info("No ranked products in the parsed response.")

    metrics_block = payload.get("metrics")
    trust_block = payload.get("trust")
    top_trust, top_geo = _extract_trust_geo(payload)
    top_accuracy = payload.get("accuracy")

    if top_accuracy is None and isinstance(metrics_block, dict):
        top_accuracy = metrics_block.get("accuracy_score")
    if top_accuracy is None and isinstance(trust_block, dict):
        top_accuracy = trust_block.get("accuracy_score")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Trust score", _format_metric(float(top_trust) if top_trust is not None else None))
    with c2:
        st.metric("Accuracy score", _format_metric(float(top_accuracy) if top_accuracy is not None else None))
    with c3:
        st.metric("GEO score", _format_metric(float(top_geo) if top_geo is not None else None))

    # Reliability / calibration
    confidence_score = _safe_float(payload.get("confidence_score"))
    quality_score = _safe_float(payload.get("quality_score"))
    brand_detected = payload.get("brand_detected")
    brand_detected_bool = bool(brand_detected) if brand_detected is not None else None
    llm_valid = payload.get("llm_valid")
    llm_valid_bool = bool(llm_valid) if llm_valid is not None else None
    used_fallback = payload.get("used_fallback")
    used_fallback_bool = bool(used_fallback) if used_fallback is not None else (llm_valid_bool is False)
    if confidence_score is not None:
        st.markdown(
            f"**Confidence Score:** {round(float(confidence_score) * 100, 1)}% "
            f"({_confidence_band(float(confidence_score))})"
        )
    if quality_score is not None:
        st.caption(f"Quality score: **{round(float(quality_score) * 100, 1)}%**")
    if used_fallback_bool:
        st.warning("⚠️ Fallback used due to invalid LLM output. Confidence reduced.")
    # Low-quality / unverified provider warning
    low_quality = False
    if quality_score is not None and float(quality_score) < 0.4:
        low_quality = True
    if brand_detected_bool is False:
        low_quality = True
    if confidence_score is not None and float(confidence_score) < 0.5:
        low_quality = True
    if low_quality:
        st.warning("⚠️ Results may include low-confidence or unverified providers.")

    st.subheader("Explanation")
    expl = payload.get("explanation")
    if isinstance(expl, dict):
        st.markdown(expl.get("summary") or "_No summary._")
        for line in expl.get("insights") or []:
            st.markdown(f"- {line}")
    else:
        st.markdown(str(final_output.get("explanation") or parsed.get("explanation") or "_No explanation._"))

    if show_debug:
        _render_debug_panel(payload, agent_outputs)


if __name__ == "__main__":
    main()
