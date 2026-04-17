from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
import json
import os
from typing import Any, cast

from crewai import Crew, LLM, Process, Task

from app.agents.analytics_agent import build_analytics_agent
from app.agents.explanation_agent import build_explanation_agent
from app.agents.ranking_agent import build_ranking_agent
from app.agents.retrieval_agent import build_retrieval_agent
from app.agents.trust_agent import build_trust_agent
from app.core.config import get_settings
from app.core.logging import get_logger
from app.crew.tasks import build_tasks
from app.models.analyze import AnalyzeRequest
from app.tools.scoring_tool import analytics_tool, trust_tool
from app.services.utils.parser import parse_llm_json

logger = get_logger(__name__)

class _StepFailed(RuntimeError):
    def __init__(self, agent: str, entry: dict[str, Any], message: str):
        super().__init__(message)
        self.agent = agent
        self.entry = entry


def safe_agent_run(fn, default=None, agent_name: str = "unknown"):
    try:
        return fn()
    except Exception as e:
        logger.warning(f"{agent_name} failed: {str(e)}")
        return default


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _summarize_agent_output(agent: str, output: Any) -> str:
    try:
        if output is None:
            return "no output"
        if isinstance(output, dict):
            if agent == "retrieval":
                results = output.get("results")
                n = len(results) if isinstance(results, list) else 0
                src = str(output.get("catalog_source") or "").strip()
                intent = str(output.get("intent") or "").strip()
                parts = [f"results={n}"]
                if intent:
                    parts.append(f"intent={intent}")
                if src:
                    parts.append(f"catalog_source={src}")
                return ", ".join(parts)
            if agent == "ranking":
                parsed = output.get("parsed_output") if isinstance(output.get("parsed_output"), dict) else {}
                ranked = (parsed or {}).get("ranked_products")
                n = len(ranked) if isinstance(ranked, list) else 0
                prov = str(output.get("provider_used") or "").strip()
                fb = bool(output.get("fallback_used") or False)
                parts = [f"ranked_products={n}"]
                if prov:
                    parts.append(f"provider_used={prov}")
                if fb:
                    parts.append("fallback_used=true")
                return ", ".join(parts)
            if agent == "trust":
                geo = output.get("geo")
                geo_score = None
                if isinstance(geo, dict):
                    geo_score = geo.get("score")
                trust_score = output.get("trust_score")
                acc = output.get("accuracy")
                parts: list[str] = []
                if trust_score is not None:
                    parts.append(f"trust_score={trust_score}")
                else:
                    parts.append("trust_score=null")
                if acc is not None:
                    parts.append(f"accuracy={acc}")
                if geo_score is not None:
                    parts.append(f"geo_score={geo_score}")
                return ", ".join(parts) if parts else "ok"
            if agent == "analytics":
                return f"saved={bool(output.get('saved'))}" if "saved" in output else f"keys={sorted(output.keys())}"
            if agent == "explanation":
                summary = output.get("summary")
                if isinstance(summary, str) and summary.strip():
                    s = summary.strip().replace("\n", " ")
                    return s[:180] + ("…" if len(s) > 180 else "")
                return "no summary"
            keys = sorted(str(k) for k in output.keys())
            return f"keys={keys[:12]}" + ("…" if len(keys) > 12 else "")
        if isinstance(output, list):
            return f"items={len(output)}"
        text = str(output).strip()
        if not text:
            return "empty string"
        text = text.replace("\n", " ")
        return text[:180] + ("…" if len(text) > 180 else "")
    except Exception:
        return "summary unavailable"


def _run_traced_step(agent: str, fn) -> tuple[Any, dict[str, Any]]:
    start_iso = _utc_now_iso()
    t0 = time.perf_counter()
    try:
        out = fn()
        ok = True
        err = None
    except Exception as e:
        out = None
        ok = False
        err = str(e)
    t1 = time.perf_counter()
    end_iso = _utc_now_iso()
    entry: dict[str, Any] = {
        "agent": agent,
        "start_time": start_iso,
        "end_time": end_iso,
        "duration_ms": int(round((t1 - t0) * 1000.0)),
        "success": ok,
        "error": err,
        "output_summary": _summarize_agent_output(agent, out),
    }
    return out, entry


def _crew_llm(provider: str) -> LLM:
    """
    Configure the CrewAI planner LLM from existing settings.

    We keep this lightweight: agents primarily call internal tools; this LLM steers tool use.
    """
    settings = get_settings()

    normalized = (provider or "ollama").strip().lower()
    api_key = (settings.llm_api_key or "").strip() or None

    # If the request is for Ollama (or OpenAI is not configured), keep the planner local.
    # This prevents CrewAI from hard-failing before our internal tools can run.
    if normalized == "ollama" or (normalized == "openai" and api_key is None):
        ollama_base_url = os.getenv("OLLAMA_BASE_URL") or settings.ollama_base_url or "http://localhost:11434"
        # Default to a smaller local model to keep CrewAI planner latency low.
        ollama_model = os.getenv("OLLAMA_MODEL") or settings.ollama_model or "phi3"
        return LLM(
            model=f"ollama/{ollama_model}",
            api_key=None,
            base_url=ollama_base_url,
            temperature=settings.llm_temperature,
            timeout=float(settings.llm_timeout_seconds),
            max_retries=1,
        )

    # Otherwise, default to the configured OpenAI(-compatible) setup.
    model = settings.llm_model
    if "/" not in model:
        model = f"openai/{model}"

    return LLM(
        model=model,
        api_key=api_key,
        base_url=settings.llm_base_url,
        temperature=settings.llm_temperature,
        timeout=float(settings.llm_timeout_seconds),
        max_retries=1,
    )


def _parse_task_json(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if raw is None:
        return {}
    text = str(raw).strip()
    if not text:
        return {}
    try:
        # Crew tasks often wrap JSON with helper text; reuse our hardened LLM JSON extraction.
        env = parse_llm_json(text)
        if isinstance(env, dict) and isinstance(env.get("data"), dict):
            return cast(dict[str, Any], env["data"])
        return {}
    except Exception:
        logger.debug("crewai_task_output_not_json", extra={"raw_preview": text[:3000]})
        return {}


def _task_output_as_dict(task: Any) -> dict[str, Any]:
    """
    Best-effort extraction of a CrewAI Task output across versions.
    """
    for attr in ("output", "result", "raw"):
        if hasattr(task, attr):
            v = getattr(task, attr)
            if v is None:
                continue
            # TaskOutput objects commonly store raw text under `.raw`
            if hasattr(v, "raw"):
                return _parse_task_json(getattr(v, "raw"))
            return _parse_task_json(v)
    return {}


def _sanitize_ranked_products(value: Any) -> list[dict[str, Any]]:
    """
    Enforce strict shape for downstream agents:
      { "ranked_products": [ { "name": str, "rank": int }, ... ] }
    """
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        try:
            rank = int(item.get("rank"))
        except Exception:
            continue
        if rank <= 0:
            continue
        out.append({"name": name, "rank": rank})
    out.sort(key=lambda x: int(x["rank"]))
    # de-dupe by name (keep best/lowest rank)
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in out:
        k = item["name"].strip().lower()
        if k in seen:
            continue
        seen.add(k)
        deduped.append(item)
    return deduped


def run_trustlens_agents(
    query: str,
    *,
    provider: str = "ollama",
    simulate_failure: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """
    Run TrustLens as a sequential multi-agent CrewAI pipeline.

    Returns a unified response dict:
    {
      "ranking": [...],
      "trust": 0.82 | null,
      "geo": 0.74 | null,
      "explanation": "...",
      "raw_output": "...",
      "provider_used": "ollama" | "openai" | "openrouter",
      "fallback_used": bool,
      "accuracy": 0.55 | null,
      "agents_trace": [...]   # optional debug
    }
    """
    llm = _crew_llm(provider)

    retrieval_agent = build_retrieval_agent()
    ranking_agent = build_ranking_agent()
    # Planner LLM can still be used for tool calling / orchestration.
    ranking_agent.llm = llm
    trust_agent = build_trust_agent()
    analytics_agent = build_analytics_agent()
    explanation_agent = build_explanation_agent()
    explanation_agent.llm = llm

    tasks = build_tasks(
        retrieval_agent=retrieval_agent,
        ranking_agent=ranking_agent,
        trust_agent=trust_agent,
        analytics_agent=analytics_agent,
        explanation_agent=explanation_agent,
    )

    if provider == "all":
        raise ValueError('CrewAI sequential pipeline is currently for single-provider mode; use provider != "all".')

    # We run the exact same underlying tools in order, but with explicit timing + status tracking.
    # This makes execution visible in UI/debug mode without depending on CrewAI internal callbacks.
    from app.tools.rag_tool import retrieve_context
    from app.tools.llm_tool import ranking_tool, explanation_tool
    from app.tools.scoring_tool import trust_tool as trust_tool_fn, analytics_tool as analytics_tool_fn

    agent_trace: list[dict[str, Any]] = []

    simulate_failure = simulate_failure or {}

    retrieval_output, tr = _run_traced_step(
        "retrieval",
        lambda: (_ for _ in ()).throw(Exception("Simulated retrieval failure"))
        if bool(simulate_failure.get("retrieval"))
        else retrieve_context(query),
    )
    agent_trace.append(tr)
    if not isinstance(retrieval_output, dict):
        retrieval_output = {}

    ranking_output, tr = _run_traced_step(
        "ranking",
        lambda: (_ for _ in ()).throw(Exception("Simulated ranking failure"))
        if bool(simulate_failure.get("ranking"))
        else ranking_tool(query=query, provider=str(provider or "ollama"), retrieval=cast(dict[str, Any], retrieval_output)),
    )
    agent_trace.append(tr)
    if not isinstance(ranking_output, dict):
        ranking_output = {}

    parsed_output = ranking_output.get("parsed_output") if isinstance(ranking_output.get("parsed_output"), dict) else {}
    ranked_products = _sanitize_ranked_products((parsed_output or {}).get("ranked_products"))
    llm_valid = bool(ranking_output.get("llm_valid")) if "llm_valid" in ranking_output else True
    used_fallback = bool(ranking_output.get("used_fallback")) if "used_fallback" in ranking_output else (not llm_valid)
    parsing_success = llm_valid
    if not ranked_products:
        ranked_products = [
            {"rank": 1, "name": "HDFC Bank Personal Loan", "notes": "Fallback"},
            {"rank": 2, "name": "ICICI Bank Personal Loan", "notes": "Fallback"},
        ]
    elif not llm_valid:
        # Optional: mark ranked rows as fallback when LLM output was invalid.
        for item in ranked_products:
            if isinstance(item, dict) and not item.get("notes"):
                item["notes"] = "Fallback"

    rag_has_context = False
    try:
        retrieved = retrieval_output.get("retrieved_documents")
        if isinstance(retrieved, list) and len(retrieved) > 0:
            rag_has_context = True
    except Exception:
        rag_has_context = False

    ranking_result = {
        "provider_used": str(ranking_output.get("provider_used") or provider),
        "ranked_products": ranked_products,
        "explanation": str((parsed_output or {}).get("explanation") or ""),
        "rag_has_context": rag_has_context,
        "llm_valid": llm_valid,
    }

    def _traced(agent: str, fn) -> tuple[Any, dict[str, Any]]:
        start_iso = _utc_now_iso()
        t0 = time.perf_counter()
        try:
            out = fn()
            ok = True
            err = None
        except Exception as e:
            out = None
            ok = False
            err = str(e)
        t1 = time.perf_counter()
        end_iso = _utc_now_iso()
        entry: dict[str, Any] = {
            "agent": agent,
            "start_time": start_iso,
            "end_time": end_iso,
            "duration_ms": int(round((t1 - t0) * 1000.0)),
            "success": ok,
            "error": err,
            "output_summary": _summarize_agent_output(agent, out),
        }
        if not ok:
            raise _StepFailed(agent, entry, err or f"{agent} failed")
        return out, entry

    async def _run_parallel_post_ranking() -> tuple[Any, Any, Any]:
        trust_task = asyncio.to_thread(
            lambda: _traced(
                "trust",
                lambda: (_ for _ in ()).throw(Exception("Simulated trust failure"))
                if bool(simulate_failure.get("trust"))
                else trust_tool_fn(query=query, ranking=ranking_result),
            )
        )
        analytics_task = asyncio.to_thread(
            lambda: _traced(
                "analytics",
                lambda: analytics_tool_fn(
                    query=query,
                    provider=str(ranking_output.get("provider_used") or provider),
                    # Analytics should not block on trust; best-effort persistence.
                    trust_score=None,
                    geo=None,
                ),
            )
        )
        explanation_task = asyncio.to_thread(
            lambda: _traced(
                "explanation",
                lambda: explanation_tool(
                    parsed_output={
                        "ranked_products": ranked_products,
                        "explanation": str((parsed_output or {}).get("explanation") or ""),
                    }
                ),
            )
        )
        return await asyncio.gather(trust_task, analytics_task, explanation_task, return_exceptions=True)

    trust_pair, analytics_pair, explanation_pair = asyncio.run(_run_parallel_post_ranking())

    trust_output: dict[str, Any] = {}
    analytics_output: dict[str, Any] = {}
    explanation_output: dict[str, Any] = {}

    for agent_name, result in (
        ("trust", trust_pair),
        ("analytics", analytics_pair),
        ("explanation", explanation_pair),
    ):
        if isinstance(result, _StepFailed):
            logger.exception("%s_step_failed", agent_name, extra={"agent": agent_name, "error": str(result)})
            agent_trace.append(result.entry)
            continue
        if isinstance(result, Exception):
            logger.exception("%s_step_failed_unexpected", agent_name, extra={"agent": agent_name, "error": str(result)})
            agent_trace.append(
                {
                    "agent": agent_name,
                    "start_time": _utc_now_iso(),
                    "end_time": _utc_now_iso(),
                    "duration_ms": 0,
                    "success": False,
                    "error": str(result),
                    "output_summary": "no output",
                }
            )
            continue
        if isinstance(result, tuple) and len(result) == 2:
            out, tr = result
            agent_trace.append(cast(dict[str, Any], tr))
            if agent_name == "trust" and isinstance(out, dict):
                trust_output = out
            elif agent_name == "analytics" and isinstance(out, dict):
                analytics_output = out
            elif agent_name == "explanation" and isinstance(out, dict):
                explanation_output = out
            continue

        logger.warning("agent_parallel_unexpected_result", extra={"agent": agent_name, "type": str(type(result))})
        agent_trace.append(
            {
                "agent": agent_name,
                "start_time": _utc_now_iso(),
                "end_time": _utc_now_iso(),
                "duration_ms": 0,
                "success": False,
                "error": "unexpected result shape",
                "output_summary": "no output",
            }
        )

    def _split_steps(text: str) -> list[str]:
        t = (text or "").strip()
        if not t:
            return []
        if " | " in t:
            parts = [p.strip() for p in t.split(" | ")]
            return [p for p in parts if p]
        if "; " in t:
            parts = [p.strip() for p in t.split("; ")]
            return [p for p in parts if p]
        return [t]

    # trust tool returns `geo` payload; allow a `geo_score` fallback if present.
    geo_payload = trust_output.get("geo") if isinstance(trust_output.get("geo"), dict) else None
    geo_score = None
    if isinstance(geo_payload, dict):
        try:
            geo_score = float(geo_payload.get("score"))  # type: ignore[arg-type]
        except Exception:
            geo_score = None
    if geo_score is None:
        try:
            geo_score = float(trust_output.get("geo_score"))  # type: ignore[arg-type]
        except Exception:
            geo_score = None

    trust_score = trust_output.get("trust_score")
    try:
        trust_score_f = float(trust_score) if trust_score is not None else 0.0
    except Exception:
        trust_score_f = 0.0

    accuracy = trust_output.get("accuracy")
    try:
        accuracy_f = float(accuracy) if accuracy is not None else None
    except Exception:
        accuracy_f = None

    explanation_text = str(explanation_output.get("summary") or "").strip()
    trust_reason = str(trust_output.get("trust_reason") or "").strip()
    geo_reason = str(trust_output.get("geo_reason") or (geo_payload or {}).get("geo_reason") or "").strip()

    debug_panel = {
        "retrieval": cast(dict[str, Any], retrieval_output),
        "ranking": {
            "provider_used": ranking_output.get("provider_used"),
            "fallback_used": ranking_output.get("fallback_used"),
            "prompt_used": ranking_output.get("prompt_used"),
            "raw_output": ranking_output.get("raw_output"),
            "raw_llm_json": ranking_output.get("raw_llm_json"),
            "parsed_output": ranking_output.get("parsed_output"),
            "llm_valid": ranking_output.get("llm_valid"),
            "used_fallback": ranking_output.get("used_fallback"),
            "parsing_success": ranking_output.get("parsing_success"),
        },
        "trust": {
            "trust_score": trust_output.get("trust_score"),
            "accuracy": trust_output.get("accuracy"),
            "trust_reason": trust_reason,
            "trust_calculation_steps": _split_steps(trust_reason),
            "llm_valid": trust_output.get("llm_valid"),
        },
        "geo": {
            "geo": geo_payload,
            "geo_score": (geo_payload or {}).get("score") if isinstance(geo_payload, dict) else None,
            "geo_reason": geo_reason,
            "geo_calculation_steps": _split_steps(geo_reason),
            "issues": (geo_payload or {}).get("issues") if isinstance(geo_payload, dict) else None,
            "recommendations": (geo_payload or {}).get("recommendations") if isinstance(geo_payload, dict) else None,
        },
        "explanation": {
            "explanation_prompt": "Summarize the model-provided `explanation` field for the ranked products.",
            "output": explanation_output,
        },
    }

    final_result = {
        "ranked_products": ranked_products,
        "trust_score": trust_score_f,
        "geo_score": float(geo_score) if geo_score is not None else 0.0,
        "explanation": explanation_text,
        "trust_reason": trust_output.get("trust_reason"),
        "geo_reason": trust_output.get("geo_reason"),
        "llm_valid": llm_valid,
        "used_fallback": used_fallback,
        "parsing_success": parsing_success,
        "agent_debug": {
            "retrieval_status": "success" if agent_trace[0].get("success") else "fallback",
            "ranking_status": "success" if agent_trace[1].get("success") else "failed",
            "trust_status": "computed",
            "geo_status": "computed",
        },
        "health": {
            "retrieval_ok": bool(retrieval_output),
            "ranking_ok": bool(ranking_output),
            "trust_ok": bool(trust_output),
            "analytics_ok": bool(analytics_output),
            "explanation_ok": bool(explanation_output),
        },
    }

    logger.info("Partial pipeline executed successfully")

    warnings: list[str] = []
    trace_by_agent: dict[str, dict[str, Any]] = {}
    for item in agent_trace:
        if isinstance(item, dict) and isinstance(item.get("agent"), str):
            trace_by_agent[str(item.get("agent"))] = item

    if isinstance(trace_by_agent.get("retrieval"), dict) and not bool(trace_by_agent["retrieval"].get("success", True)):
        warnings.append("⚠ Retrieval failed → fallback used")
    # Ranking step can fail hard but still produce fallback ranked_products downstream.
    if isinstance(trace_by_agent.get("ranking"), dict) and not bool(trace_by_agent["ranking"].get("success", True)):
        warnings.append("⚠ Ranking failed → fallback used")
    if isinstance(trace_by_agent.get("trust"), dict) and not bool(trace_by_agent["trust"].get("success", True)):
        warnings.append("⚠ Trust failed → partial scoring used")

    return {
        "ranking": final_result["ranked_products"],
        "trust": final_result["trust_score"],
        "geo": geo_payload if isinstance(geo_payload, dict) else {"score": final_result["geo_score"]},
        "explanation": final_result["explanation"],
        # Ranking tool output might include raw_output/provider_used; best-effort passthrough.
        "raw_output": str(ranking_output.get("raw_output") or ranking_output or ""),
        "provider_used": str(ranking_output.get("provider_used") or provider),
        "fallback_used": bool(ranking_output.get("fallback_used") or False),
        "accuracy": accuracy_f,
        "final_output": final_result,
        "debug": debug_panel,
        "llm_valid": llm_valid,
        "used_fallback": used_fallback,
        "parsing_success": parsing_success,
        # New structured logs requested by UI/debug mode.
        "agent_trace": agent_trace,
        # Backward-compatible agent outputs payload used by existing debug wiring.
        "agents_trace": [
            {"agent": "retrieval", "output": retrieval_output},
            {"agent": "ranking", "output": ranking_output},
            {"agent": "trust", "output": trust_output},
            {"agent": "analytics", "output": analytics_output},
            {"agent": "explanation", "output": explanation_output},
        ],
        "warnings": warnings,
    }


def run_trustlens_agents_from_request(body: AnalyzeRequest) -> dict[str, Any]:
    return run_trustlens_agents(
        body.query,
        provider=str(body.provider),
        simulate_failure=body.simulate_failure,
    )

