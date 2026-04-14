from __future__ import annotations

import hashlib
import os
import sys
import time
from typing import Dict, List, Tuple

import gradio as gr

from utils import run_multi_llm, run_trustlens

# Allow importing the backend `app/` package from the project root
_ROOT = os.path.dirname(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from app.services.geo.recommender import generate_geo_recommendations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


APP_TITLE = "TrustLens AI"
TAGLINE = "Understand how AI ranks financial products — and how much you can trust it."

CSS = """
:root {
  --tl-bg: #0b1220;
  --tl-surface: rgba(255,255,255,0.06);
  --tl-border: rgba(255,255,255,0.10);
  --tl-text: rgba(255,255,255,0.92);
  --tl-muted: rgba(255,255,255,0.70);
  --tl-primary: #2563eb;
  --tl-primary-2: #14b8a6;
  --tl-green: #22c55e;
  --tl-yellow: #eab308;
  --tl-red: #ef4444;
}

.gradio-container,
.gradio-container textarea,
.gradio-container input,
.gradio-container button {
  font-family: "Inter", "Segoe UI", Roboto, system-ui, -apple-system, sans-serif !important;
}

.gradio-container {
  font-size: 16px;
  line-height: 1.5;
  color: var(--tl-text);
}

.gradio-container h1 {
  font-size: 30px !important;
  font-weight: 700 !important;
  letter-spacing: -0.02em;
}

.gradio-container h3 {
  font-size: 18px !important;
  font-weight: 600 !important;
  letter-spacing: -0.01em;
}

.gradio-container {
  background: radial-gradient(900px 500px at 20% -10%, rgba(99, 102, 241, 0.25), transparent 60%),
              radial-gradient(700px 400px at 100% 0%, rgba(34, 197, 94, 0.16), transparent 55%),
              var(--tl-bg);
}

.tl-wrap {
  max-width: 1040px;
  margin: 0 auto;
}

.tl-hero h1, .tl-hero h3, .tl-hero p { color: var(--tl-text); }
.tl-hero p { color: var(--tl-muted); font-size: 1.02rem; margin-top: 0.35rem; }

/* Brand header */
.header {
  background: linear-gradient(90deg, #1e3a8a, #2563eb, #06b6d4);
  padding: 16px 24px;
  border-radius: 12px;
  margin-bottom: 20px;
  position: relative;
  overflow: hidden;
}

.header::after {
  content: "";
  position: absolute;
  inset: 0;
  background: radial-gradient(900px 200px at 10% 0%, rgba(255,255,255,0.18), transparent 55%);
  pointer-events: none;
}

.header h1 {
  color: white;
  margin: 0;
}

.header p {
  color: #dbeafe;
  margin: 4px 0 0 0;
}

/* Clean section dividers */
hr {
  border: none;
  border-top: 1px solid rgba(255,255,255,0.08);
  margin: 20px 0;
}

.tl-divider {
  border-top: 1px solid rgba(255,255,255,0.08);
  margin: 14px 0 18px;
}

.block {
  margin-bottom: 20px !important;
}

.gr-group {
  padding: 16px !important;
  border-radius: 12px !important;
  background-color: rgba(255, 255, 255, 0.03) !important;
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
  transition: all 0.2s ease;
}

.gr-group:hover {
  border-color: #2563eb !important;
  transition: all 0.2s ease;
}

.tl-card {
  background: var(--tl-surface);
  border: 1px solid var(--tl-border);
  border-radius: 16px;
  padding: 16px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}

.tl-card h3 { margin-top: 0; }

.tl-scorebox {
  display: grid;
  place-items: center;
  text-align: center;
  padding: 14px 12px 10px;
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,0.10);
  background:
    radial-gradient(600px 240px at 30% 0%, rgba(37,99,235,0.25), transparent 60%),
    rgba(255,255,255,0.03);
}

.tl-scorebox .tl-score-pct,
.trust-score {
  font-size: 3.0rem;
  font-weight: 800;
  letter-spacing: -0.03em;
  line-height: 1.05;
  margin: 2px 0 0;
}

.tl-scorebox .tl-score-label,
.trust-label {
  font-size: 1.05rem;
  font-weight: 600;
  color: var(--tl-muted);
  margin-top: 6px;
}

.tl-scorebox.tl-high .tl-score-pct { color: var(--tl-green); }
.tl-scorebox.tl-med .tl-score-pct { color: var(--tl-yellow); }
.tl-scorebox.tl-low .tl-score-pct { color: var(--tl-red); }

.tl-analyze-row { gap: 12px; align-items: flex-end; }
.tl-analyze-row button { min-height: 44px; }

.tl-foot { color: var(--tl-muted); font-size: 0.92rem; }

.gradio-container ul {
  margin-top: 10px;
  padding-left: 20px;
}

.gradio-container li {
  margin-bottom: 6px;
  color: #cbd5f5;
}

@media (max-width: 720px) {
  .tl-card { padding: 14px; }
}
"""

custom_css = """
.gradio-container textarea,
.gradio-container input {
  font-size: 16px !important;
  line-height: 1.5 !important;
  color: #ffffff !important;
  background-color: #1f2937 !important;
  border-radius: 10px !important;
  padding: 14px !important;
  border: 1px solid #374151 !important;
}

.gradio-container textarea {
  width: 100% !important;
}

.gradio-container textarea {
  resize: vertical;
}

.gradio-container textarea::placeholder {
  color: #9ca3af !important;
}

.gradio-container textarea::selection {
  background: #2563eb !important;
  color: #ffffff !important;
}

.gradio-container textarea::-moz-selection {
  background: #2563eb !important;
  color: #ffffff !important;
}

.gradio-container button {
  font-size: 16px !important;
  font-weight: 600 !important;
  border-radius: 10px !important;
  padding: 10px 16px !important;
  background: linear-gradient(90deg, #2563eb, #3b82f6) !important;
  border: none !important;
  color: white !important;
  transition: all 0.2s ease !important;
  will-change: transform, box-shadow;
}

.gradio-container button:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}

.gradio-container button:active {
  transform: translateY(0px);
  box-shadow: 0 2px 8px rgba(0,0,0,0.18);
}

.trust-box {
  text-align: center;
  padding: 16px;
}
"""


def _split_title_and_reason(item: str) -> tuple[str, str]:
    text = (item or "").strip()
    if not text:
        return "", ""

    for sep in [" — ", " – ", " - ", ": "]:
        if sep in text:
            left, right = text.split(sep, 1)
            title = left.strip()
            reason = right.strip()
            return title, reason

    return text, ""


def _clamp_0_1(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _stable_seed_int(text: str) -> int:
    digest = hashlib.md5((text or "").strip().lower().encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def build_chart_data(score_0_to_1: float) -> Dict[str, int]:
    """
    Simulate a simple trust distribution (High/Medium/Low) that:
    - sums to 100
    - shifts towards "High" as score increases
    """
    s = _clamp_0_1(score_0_to_1)
    high = int(round(50 + 40 * s))  # 50..90
    medium = int(round(10 + 20 * (1 - s)))  # 30..10-ish
    high = max(0, min(100, high))
    medium = max(0, min(100 - high, medium))
    low = max(0, 100 - high - medium)
    return {"High": high, "Medium": medium, "Low": low}


def generate_confidence_scores(ranking: List[str], query: str) -> Dict[str, float]:
    """
    Simulated per-item confidence scores (0..1) that are deterministic per query.
    Keeps the list sorted (best-ranked gets highest confidence).
    """
    items = [x.strip() for x in (ranking or []) if (x or "").strip()]
    if not items:
        return {}

    base_seed = _stable_seed_int(query)
    raw: List[Tuple[str, float]] = []
    for name in items:
        seed = base_seed ^ _stable_seed_int(name)
        # Map to 0.72..0.96
        val = 0.72 + ((seed % 1000) / 1000.0) * 0.24
        raw.append((name, float(val)))

    # Enforce monotonic decrease by rank while keeping deterministic variation.
    scores: Dict[str, float] = {}
    prev = 0.99
    for i, (name, v) in enumerate(raw):
        target = min(prev - 0.03, v) if i > 0 else max(v, 0.86)
        target = max(0.60, min(0.97, target))
        scores[name] = target
        prev = target
    return scores


def _plot_trust_distribution(dist: Dict[str, int]):
    labels = ["High", "Medium", "Low"]
    values = [int(dist.get(k, 0)) for k in labels]
    colors = ["#22c55e", "#eab308", "#ef4444"]

    fig, ax = plt.subplots(figsize=(4.2, 1.8), dpi=140)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    y = list(range(len(labels)))
    ax.barh(y, values, color=colors, height=0.52)
    ax.set_yticks(y, labels=labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)

    for yi, val in zip(y, values):
        ax.text(min(val + 2, 98), yi, f"{val}%", va="center", ha="left", fontsize=9, color="#e5e7eb")

    ax.tick_params(axis="x", colors="#94a3b8", labelsize=8)
    ax.tick_params(axis="y", colors="#e5e7eb", labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.18)
    ax.set_title("Trust distribution", fontsize=10, color="#e5e7eb", pad=6)
    fig.tight_layout(pad=0.8)
    return fig


def _plot_confidence_bars(conf: Dict[str, float]):
    items = list(conf.items())
    labels = [k.replace(" Health Insurance", "").replace(" Personal Loan", "") for k, _ in items]
    values = [float(v) * 100 for _, v in items]

    fig, ax = plt.subplots(figsize=(4.8, 2.2), dpi=140)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    y = list(range(len(labels)))
    ax.barh(y, values, color="#60a5fa", height=0.55)
    ax.set_yticks(y, labels=labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)

    for yi, val in zip(y, values):
        ax.text(min(val + 2, 98), yi, f"{int(round(val))}%", va="center", ha="left", fontsize=9, color="#e5e7eb")

    ax.tick_params(axis="x", colors="#94a3b8", labelsize=8)
    ax.tick_params(axis="y", colors="#e5e7eb", labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.18)
    ax.set_title("Ranking confidence", fontsize=10, color="#e5e7eb", pad=6)
    fig.tight_layout(pad=0.8)
    return fig


def generate_trust_chart(data: Dict[str, float | int]):
    """
    Bar chart: Models vs Trust Score (0..100).
    """
    items = list((data or {}).items())
    labels = [str(k) for k, _ in items]
    values = [float(v) for _, v in items]

    fig, ax = plt.subplots(figsize=(4.6, 2.6), dpi=140)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    colors = ["#60a5fa", "#34d399", "#93c5fd", "#6ee7b7"]  # soft blue/green
    bars = ax.bar(labels, values, color=colors[: max(1, len(values))])

    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", colors="#e5e7eb", labelsize=9)
    ax.tick_params(axis="y", colors="#94a3b8", labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.18)
    ax.set_title("Trust score comparison", fontsize=10, color="#e5e7eb", pad=6)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(val + 2, 98),
            f"{int(round(val))}%",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#e5e7eb",
        )

    fig.tight_layout(pad=0.8)
    return fig


def generate_confidence_chart(data: Dict[str, float | int]):
    """
    Horizontal bar chart: Products vs Confidence (0..100).
    """
    items = list((data or {}).items())
    labels = [str(k) for k, _ in items]
    values = [float(v) for _, v in items]

    fig, ax = plt.subplots(figsize=(4.8, 2.6), dpi=140)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    y = list(range(len(labels)))
    ax.barh(y, values, color="#34d399", height=0.55)
    ax.set_yticks(y, labels=labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)

    for yi, val in zip(y, values):
        ax.text(min(val + 2, 98), yi, f"{int(round(val))}%", va="center", ha="left", fontsize=9, color="#e5e7eb")

    ax.tick_params(axis="x", colors="#94a3b8", labelsize=8)
    ax.tick_params(axis="y", colors="#e5e7eb", labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.18)
    ax.set_title("Confidence distribution", fontsize=10, color="#e5e7eb", pad=6)
    fig.tight_layout(pad=0.8)
    return fig


def _format_agreement_md(agreement_score_0_to_1: float) -> str:
    s = _clamp_0_1(agreement_score_0_to_1)
    pct = int(round(s * 100))
    return f"### 🤝 Model Agreement\n\n**{pct}%**"


def format_score(score_0_to_1: float) -> str:
    score = _clamp_0_1(score_0_to_1)
    pct = int(round(score * 100))

    if score >= 0.85:
        emoji = "🟢"
        label = "High Trust"
        cls = "tl-high"
    elif score >= 0.70:
        emoji = "🟡"
        label = "Medium Trust"
        cls = "tl-med"
    else:
        emoji = "🔴"
        label = "Low Trust"
        cls = "tl-low"

    return "\n".join(
        [
            f'<div class="tl-scorebox trust-box {cls}">',
            f'  <div class="tl-score-pct trust-score">{emoji} {pct}%</div>',
            f'  <div class="tl-score-label trust-label">{label}</div>',
            "</div>",
        ]
    )


def _trim_text(text: str, max_chars: int = 900) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1].rstrip() + "…"


def _smart_paragraphs(text: str, max_paras: int = 3) -> str:
    t = " ".join((text or "").strip().split())
    if not t:
        return ""

    # Light-touch readability: split into short paragraphs (2–3) without heavy NLP.
    parts = [p.strip() for p in t.replace("\n", " ").split(". ") if p.strip()]
    if len(parts) <= 2:
        return "\n\n".join([p if p.endswith(".") else p + "." for p in parts]).strip()

    # Group sentences into up to max_paras paragraphs.
    total = len(parts)
    para_size = max(1, (total + max_paras - 1) // max_paras)
    paras: list[str] = []
    for i in range(0, total, para_size):
        chunk = parts[i : i + para_size]
        para = ". ".join(chunk)
        if not para.endswith("."):
            para += "."
        paras.append(para)
        if len(paras) >= max_paras:
            break
    return "\n\n".join(paras).strip()


def format_explanation(explanation: str) -> str:
    body = _trim_text(explanation, max_chars=900)
    body = _smart_paragraphs(body, max_paras=3)
    return body or "_No explanation yet._"


def format_ranking(ranking: list[str]) -> str:
    items = [x.strip() for x in (ranking or []) if (x or "").strip()]
    lines: list[str] = []
    if not items:
        lines.append("_No results yet._")
        return "\n".join(lines)

    for i, raw in enumerate(items[:10], start=1):
        title, reason = _split_title_and_reason(raw)
        if reason:
            lines.append(f"{i}. **{title}**\n   *{reason}*")
        else:
            lines.append(f"{i}. **{title}**")
        lines.append("")
    return "\n".join(lines).rstrip()


def _company_from_item(name: str) -> str:
    s = " ".join((name or "").strip().split())
    if not s:
        return ""
    first = s.split(" ", 1)[0].strip()
    return first


def format_competitor_comparison(company: str, ranking: list[str]) -> str:
    """
    Renders:
      - Your company highlight
      - Competitor rankings
      - Optional top performer highlight
    """
    your = (company or "").strip()
    if not your:
        your = "Your Company"

    items = [x for x in (ranking or []) if (x or "").strip()]
    if not items:
        return "_No comparison yet._"

    companies: list[str] = []
    for item in items:
        c = _company_from_item(item)
        if c and c not in companies:
            companies.append(c)

    if not companies:
        return "_No comparison yet._"

    your_norm = your.strip().lower()
    your_rank = 0
    for idx, c in enumerate(companies, start=1):
        if c.lower() == your_norm:
            your_rank = idx
            break

    top = companies[0]
    lines: list[str] = []
    lines.append(f"**Your company:** **{your}**" + (f" (Rank **{your_rank}**)" if your_rank else ""))
    lines.append("")
    if top.lower() != your_norm:
        lines.append(f"**Top performer:** 🥇 **{top}** (Rank **1**)")
        lines.append("")
    lines.append("**Competitor rankings:**")
    for idx, c in enumerate(companies, start=1):
        if c.lower() == your_norm:
            lines.append(f"- ⭐ **{c}** — Rank **{idx}**")
        else:
            lines.append(f"- {c} — Rank **{idx}**")
    return "\n".join(lines).strip()


def _format_multi_llm_card(model_name: str, ranking: List[str], score_0_to_1: float, base_ranking: List[str]) -> str:
    items = [x.strip() for x in (ranking or []) if (x or "").strip()]
    base_items = [x.strip() for x in (base_ranking or []) if (x or "").strip()]
    n = max(len(items), len(base_items))
    if n == 0:
        return f"### {model_name}\n\n_No results yet._\n\nTrust: 0%"

    def short_name(x: str) -> str:
        return (
            (x or "")
            .replace(" Health Insurance", "")
            .replace(" Personal Loan", "")
            .replace(" Insurance", "")
            .strip()
        )

    lines: List[str] = [f"### {model_name}", ""]
    for i in range(n):
        item = items[i] if i < len(items) else ""
        base = base_items[i] if i < len(base_items) else ""
        label = short_name(item) if item else "—"
        if base and item and (item != base):
            # Highlight per-rank differences vs the baseline ranking.
            label = f"**{label}**"
        lines.append(f"{i+1}. {label}")
    lines.append("")
    pct = int(round(max(0.0, min(1.0, float(score_0_to_1))) * 100))
    lines.append(f"Trust: {pct}%")
    return "\n".join(lines)


def analyze_ui(query: str, company: str):
    q = (query or "").strip()
    if not q:
        warning = "⚠️ Please enter a query to analyze"
        empty_dist = _plot_trust_distribution(build_chart_data(0.0))
        empty_conf = _plot_confidence_bars({})
        empty_multi = _format_multi_llm_card("GPT-4", [], 0.0, [])
        empty_trust_cmp = generate_trust_chart({"GPT-4": 90, "Claude": 85, "Llama": 80})
        empty_conf_dist = generate_confidence_chart({"HDFC": 92, "ICICI": 85, "Star": 78})
        empty_agreement = _format_agreement_md(0.75)
        empty_competitors = format_competitor_comparison(company, [])
        empty_geo_issues = "_No issues yet._"
        empty_geo_recs = "_No recommendations yet._"
        return (
            warning,
            format_ranking([]),
            format_score(0),
            0.0,
            empty_dist,
            empty_conf,
            format_explanation(""),
            empty_multi,
            _format_multi_llm_card("Claude", [], 0.0, []),
            _format_multi_llm_card("Llama", [], 0.0, []),
            empty_trust_cmp,
            empty_conf_dist,
            empty_agreement,
            empty_competitors,
            empty_geo_issues,
            empty_geo_recs,
        )

    time.sleep(0.8)
    result = run_trustlens(q)
    ranking_md = format_ranking(list(result.get("ranking") or []))
    competitor_md = format_competitor_comparison(company, list(result.get("ranking") or []))
    score_val = float(result.get("score", 0.0))
    score_md = format_score(score_val)
    dist = build_chart_data(score_val)
    dist_fig = _plot_trust_distribution(dist)
    conf = generate_confidence_scores(list(result.get("ranking") or []), q)
    conf_fig = _plot_confidence_bars(conf)
    explanation_md = format_explanation(str(result.get("explanation") or ""))

    multi = run_multi_llm(q)
    base_ranking = list(result.get("ranking") or [])
    gpt = multi.get("gpt4") or {}
    claude = multi.get("claude") or {}
    llama = multi.get("llama") or {}
    gpt_md = _format_multi_llm_card("GPT-4", list(gpt.get("ranking") or []), float(gpt.get("score") or 0.0), base_ranking)
    claude_md = _format_multi_llm_card(
        "Claude", list(claude.get("ranking") or []), float(claude.get("score") or 0.0), base_ranking
    )
    llama_md = _format_multi_llm_card("Llama", list(llama.get("ranking") or []), float(llama.get("score") or 0.0), base_ranking)

    trust_cmp = generate_trust_chart(
        {
            "GPT-4": float(gpt.get("score") or 0.0) * 100.0,
            "Claude": float(claude.get("score") or 0.0) * 100.0,
            "Llama": float(llama.get("score") or 0.0) * 100.0,
        }
    )
    conf_dist = generate_confidence_chart(
        {
            (k.replace(" Health Insurance", "").replace(" Personal Loan", "").replace(" Insurance", "").strip() or k): float(v) * 100.0
            for k, v in list(conf.items())[:3]
        }
    )
    agreement_md = _format_agreement_md(0.75)

    # GEO optimisation insights (simple demo wiring)
    q_lower = q.lower()
    if ("health" in q_lower) or ("insurance" in q_lower):
        ground_truth = [
            "HDFC Ergo Health Insurance",
            "ICICI Lombard Health Insurance",
            "Star Health Insurance",
        ]
    elif "loan" in q_lower:
        ground_truth = [
            "HDFC Bank Personal Loan",
            "SBI Personal Loan",
            "ICICI Personal Loan",
        ]
    else:
        ground_truth = list(dict.fromkeys(list(result.get("ranking") or [])))

    ranking_list = list(result.get("ranking") or [])
    missing_products = [p for p in ground_truth if p not in ranking_list]
    geo = generate_geo_recommendations(
        {
            "rankings": [{"query": q, "ranking": ranking_list}],
            "missing_products": [{"query": q, "items": missing_products}],
            "ground_truth": [{"query": q, "items": ground_truth}],
        }
    )
    geo_issues = geo.get("issues") or []
    geo_recs = geo.get("recommendations") or []
    geo_issues_md = "\n".join([f"- {x}" for x in geo_issues]) if geo_issues else "_No issues detected._"
    geo_recs_md = "\n".join([f"- {x}" for x in geo_recs]) if geo_recs else "_No recommendations yet._"

    return (
        "",
        ranking_md,
        score_md,
        _clamp_0_1(score_val),
        dist_fig,
        conf_fig,
        explanation_md,
        gpt_md,
        claude_md,
        llama_md,
        trust_cmp,
        conf_dist,
        agreement_md,
        competitor_md,
        geo_issues_md,
        geo_recs_md,
    )


def begin_run(query: str) -> tuple[gr.update, str]:
    q = (query or "").strip()
    if not q:
        return gr.update(interactive=True), "⚠️ Please enter a query to analyze"
    return gr.update(interactive=False), ""


def end_run() -> gr.update:
    return gr.update(interactive=True)


theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="teal",
    neutral_hue="slate",
).set(
    body_background_fill="#0b1220",
    body_background_fill_dark="#0b1220",
    body_text_color="#e5e7eb",
    body_text_color_dark="#e5e7eb",
    block_background_fill="rgba(255,255,255,0.06)",
    block_background_fill_dark="rgba(255,255,255,0.06)",
    block_border_color="rgba(255,255,255,0.10)",
    block_border_color_dark="rgba(255,255,255,0.10)",
    button_primary_background_fill="#2563eb",
    button_primary_background_fill_dark="#2563eb",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    input_background_fill="rgba(255,255,255,0.05)",
    input_background_fill_dark="rgba(255,255,255,0.05)",
)

with gr.Blocks(title=APP_TITLE, css=CSS + custom_css, theme=theme) as demo:
    with gr.Column(elem_classes=["tl-wrap"]):
        gr.Markdown(
            """
<div class="header">
  <h1>🚀 TrustLens AI</h1>
  <p><strong>LLM Trust &amp; Ranking Intelligence</strong></p>
  <p>Understand how AI ranks financial products — and how much you can trust it.</p>
</div>
""",
        )

        gr.HTML("<hr />")

        with gr.Group(elem_classes=["tl-card"]):
            gr.Markdown("### 🔎 Query")
            with gr.Row(equal_height=True, elem_classes=["tl-analyze-row"]):
                with gr.Column(scale=2, min_width=320):
                    query_in = gr.Textbox(
                        label="What financial product are you looking for?",
                        placeholder="e.g., Best health insurance for family",
                        lines=2,
                        max_lines=4,
                    )
                    company_in = gr.Textbox(
                        label="Your company (for competitor comparison)",
                        placeholder="e.g., HDFC",
                        value="HDFC",
                        lines=1,
                        max_lines=1,
                    )
                with gr.Column(scale=1, min_width=160):
                    analyze_btn = gr.Button("🔍 Analyze", variant="primary")
            gr.Markdown(
                "\n".join(
                    [
                        "**Try examples:**",
                        "- Best health insurance for family",
                        "- Best personal loan in India",
                        "- Affordable insurance plans",
                    ]
                )
            )
            status_md = gr.Markdown(value="")

        gr.HTML("<hr />")

        with gr.Row(equal_height=True):
            with gr.Column():
                with gr.Group(elem_classes=["tl-card"]):
                    gr.Markdown("### 📊 Ranked Results")
                    ranked_out = gr.Markdown(value=format_ranking([]))

            with gr.Column():
                with gr.Group(elem_classes=["tl-card"]):
                    gr.Markdown("### 🔢 Trust Score")
                    trust_md = gr.HTML(value=format_score(0))
                    trust_progress = gr.Slider(
                        label="Trust Level",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=0.0,
                        interactive=False,
                    )
                    gr.Markdown("### 📈 Trust Distribution")
                    trust_dist_plot = gr.Plot(value=_plot_trust_distribution(build_chart_data(0.0)))
                    gr.Markdown("### 🧠 Ranking Confidence")
                    confidence_plot = gr.Plot(value=_plot_confidence_bars({}))

            with gr.Column():
                with gr.Group(elem_classes=["tl-card"]):
                    gr.Markdown("### 💡 Why this ranking?")
                    explanation_out = gr.Markdown(value=format_explanation(""))

        gr.HTML("<div class='tl-divider'></div>")

        with gr.Group(elem_classes=["tl-card"]):
            gr.Markdown("### 🤖 Multi-LLM Comparison")
            gr.Markdown("Compare how different AI models rank the same query", elem_classes=["tl-foot"])
            with gr.Row():
                with gr.Column(min_width=220):
                    gpt4_card = gr.Markdown(value=_format_multi_llm_card("GPT-4", [], 0.0, []))
                with gr.Column(min_width=220):
                    claude_card = gr.Markdown(value=_format_multi_llm_card("Claude", [], 0.0, []))
                with gr.Column(min_width=220):
                    llama_card = gr.Markdown(value=_format_multi_llm_card("Llama", [], 0.0, []))

        gr.HTML("<div class='tl-divider'></div>")

        with gr.Group(elem_classes=["tl-card"]):
            gr.Markdown("### 📈 Analytics Dashboard")
            with gr.Row():
                with gr.Column(min_width=320):
                    trust_comparison_plot = gr.Plot(value=generate_trust_chart({"GPT-4": 90, "Claude": 85, "Llama": 80}))
                with gr.Column(min_width=320):
                    confidence_distribution_plot = gr.Plot(value=generate_confidence_chart({"HDFC": 92, "ICICI": 85, "Star": 78}))
            agreement_out = gr.Markdown(value=_format_agreement_md(0.75))

        gr.HTML("<div class='tl-divider'></div>")

        with gr.Group(elem_classes=["tl-card"]):
            gr.Markdown("### 🏆 Competitor Comparison")
            competitor_out = gr.Markdown(value=format_competitor_comparison("HDFC", []))

        gr.HTML("<div class='tl-divider'></div>")

        with gr.Group(elem_classes=["tl-card"]):
            gr.Markdown("### 🚀 GEO Optimisation Insights")
            with gr.Row():
                with gr.Column(min_width=320):
                    gr.Markdown("**Issues**")
                    geo_issues_out = gr.Markdown(value="_No issues yet._")
                with gr.Column(min_width=320):
                    gr.Markdown("**Actionable recommendations**")
                    geo_recs_out = gr.Markdown(value="_No recommendations yet._")

        gr.HTML("<hr />")
        gr.Markdown(
            "\n".join(
                [
                    "---",
                    "Built with ❤️ using TrustLens AI  ",
                    "AI-powered financial intelligence demo",
                    "--------------------------------------",
                ]
            ),
            elem_classes=["tl-foot"],
        )

    (
        analyze_btn.click(
            fn=begin_run,
            inputs=[query_in],
            outputs=[analyze_btn, status_md],
            queue=False,
        )
        .then(
            fn=analyze_ui,
            inputs=[query_in, company_in],
            outputs=[
                status_md,
                ranked_out,
                trust_md,
                trust_progress,
                trust_dist_plot,
                confidence_plot,
                explanation_out,
                gpt4_card,
                claude_card,
                llama_card,
                trust_comparison_plot,
                confidence_distribution_plot,
                agreement_out,
                competitor_out,
                geo_issues_out,
                geo_recs_out,
            ],
            show_progress="full",
        )
        .then(
            fn=end_run,
            outputs=[analyze_btn],
            queue=False,
        )
    )


if __name__ == "__main__":
    demo.queue()
    demo.launch()

