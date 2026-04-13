"""TrustLens AI — Streamlit UI. Run: streamlit run streamlit_app.py"""

from __future__ import annotations

import json
import os
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


def main() -> None:
    st.set_page_config(page_title="TrustLens AI", layout="wide")
    st.title("TrustLens AI")
    st.caption(f"POST `{_api_base()}/v1/analyze`")

    query = st.text_input(
        "Query",
        placeholder="What should we rank and analyze?",
        label_visibility="visible",
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
