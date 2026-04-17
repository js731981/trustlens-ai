from __future__ import annotations

import html
import random
import time
from typing import Any

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt

from utils import simulate_agents

matplotlib.use("Agg")

APP_TITLE = "TrustLens AI"
TAGLINE = "Multi-Agent Trust & GEO Intelligence (Simulated CrewAI-style)"
SUBTEXT = "Agents collaborate to retrieve, rank, score trust/GEO, and explain decisions — fully simulated in-browser."
BADGES = ["CrewAI", "RAG (Qdrant)", "Multi-Agent", "Decision Intelligence"]

CSS = """
:root {
  --tl-bg: #0b1220;
  --tl-surface: rgba(255,255,255,0.06);
  --tl-border: rgba(255,255,255,0.10);
  --tl-text: rgba(255,255,255,0.92);
  --tl-muted: rgba(255,255,255,0.70);
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
  background: radial-gradient(900px 500px at 20% -10%, rgba(99, 102, 241, 0.25), transparent 60%),
              radial-gradient(700px 400px at 100% 0%, rgba(34, 197, 94, 0.16), transparent 55%),
              var(--tl-bg);
}

.tl-wrap { max-width: 1060px; margin: 0 auto; }

.header {
  background: linear-gradient(90deg, #1e3a8a, #2563eb, #06b6d4);
  padding: 16px 24px;
  border-radius: 14px;
  margin-bottom: 18px;
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
.header h1 { color: white; margin: 0; font-size: 30px; font-weight: 800; letter-spacing: -0.02em; }
.header p { color: #dbeafe; margin: 4px 0 0 0; }
.header .markdown-body > p,
.header .prose > p { color: #dbeafe; }
.header .markdown-body > p strong,
.header .prose > p strong { color: #ffffff; }
.header .markdown-body > p em,
.header .prose > p em { color: rgba(219, 234, 254, 0.92); }

/* Capability tags: native Gradio buttons (avoids HF stripping HTML badge spans inside Markdown). */
.tl-tag-row {
  margin-top: 10px;
  margin-bottom: 2px;
  display: flex !important;
  flex-wrap: wrap !important;
  align-items: center !important;
  gap: 8px !important;
  width: 100% !important;
}
/* Shrink each column/block to the button’s intrinsic width (no equal flex stretch). */
.tl-tag-row > .form,
.tl-tag-row > div {
  flex: 0 0 auto !important;
  width: auto !important;
  min-width: 0 !important;
  max-width: none !important;
}
.gradio-container .tl-tag-row .tag-btn,
.gradio-container .tl-tag-row .tag-btn.form {
  width: fit-content !important;
  min-width: 0 !important;
}
.gradio-container .tl-tag-row .tag-btn > button,
.gradio-container .tl-tag-row .tag-btn button,
.gradio-container .tl-tag-row button.tag-btn {
  border-radius: 20px !important;
  background: linear-gradient(135deg, rgba(255,255,255,0.22), rgba(255,255,255,0.08)) !important;
  border: 1px solid rgba(255,255,255,0.30) !important;
  padding: 4px 14px !important;
  font-size: 12px !important;
  font-weight: 800 !important;
  letter-spacing: 0.01em !important;
  color: rgba(255,255,255,0.95) !important;
  cursor: default !important;
  min-height: 30px !important;
  line-height: 1.25 !important;
  box-shadow: none !important;
  opacity: 1 !important;
  width: auto !important;
  min-width: 0 !important;
  max-width: none !important;
  white-space: normal !important;
  text-align: center !important;
}
.gradio-container .tl-tag-row .tag-btn > button:hover,
.gradio-container .tl-tag-row .tag-btn button:hover,
.gradio-container .tl-tag-row button.tag-btn:hover,
.gradio-container .tl-tag-row .tag-btn > button:active,
.gradio-container .tl-tag-row .tag-btn button:active,
.gradio-container .tl-tag-row button.tag-btn:active {
  transform: none !important;
  box-shadow: none !important;
}

hr { border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 18px 0; }
.tl-divider { border-top: 1px solid rgba(255,255,255,0.08); margin: 14px 0 18px; }
.tl-foot { color: var(--tl-muted); font-size: 0.92rem; }

.tl-card {
  background: var(--tl-surface);
  border: 1px solid var(--tl-border);
  border-radius: 16px;
  padding: 16px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
  transition: transform 160ms ease, border-color 160ms ease;
}
.tl-card:hover { transform: translateY(-1px); border-color: rgba(37,99,235,0.65); }

.tl-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.04);
  color: rgba(255,255,255,0.82);
  font-weight: 700;
  font-size: 0.95rem;
}

.tl-metric {
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 14px;
  padding: 12px 12px 10px;
  background: rgba(255,255,255,0.03);
}
.tl-metric-top {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 8px;
}
.tl-metric-name { font-weight: 800; color: rgba(255,255,255,0.92); }
.tl-metric-val { font-weight: 900; letter-spacing: -0.02em; color: rgba(255,255,255,0.92); }
.tl-bar {
  height: 10px;
  border-radius: 999px;
  background: rgba(255,255,255,0.10);
  overflow: hidden;
  border: 1px solid rgba(255,255,255,0.08);
}
.tl-bar > div {
  height: 100%;
  width: 0%;
  border-radius: 999px;
  transition: width 700ms cubic-bezier(0.2, 0.8, 0.2, 1);
}
.tl-bar.trust > div { background: linear-gradient(90deg, #22c55e, #14b8a6); }
.tl-bar.geo > div { background: linear-gradient(90deg, #3b82f6, #22c55e); }
.tl-bar.conf > div { background: linear-gradient(90deg, #a855f7, #6366f1); }
.tl-metric-sub { margin-top: 8px; color: rgba(255,255,255,0.70); font-weight: 700; font-size: 0.94rem; }

.gradio-container textarea,
.gradio-container input {
  font-size: 16px !important;
  line-height: 1.5 !important;
  color: #ffffff !important;
  background-color: #111827 !important;
  border-radius: 12px !important;
  padding: 14px !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
}
.gradio-container textarea::placeholder { color: #9ca3af !important; }
.gradio-container button {
  font-size: 16px !important;
  font-weight: 800 !important;
  border-radius: 12px !important;
  padding: 10px 16px !important;
  background: linear-gradient(90deg, #2563eb, #3b82f6) !important;
  border: none !important;
  color: white !important;
  transition: transform 160ms ease, box-shadow 160ms ease !important;
  will-change: transform, box-shadow;
}
.gradio-container button:hover { transform: translateY(-1px); box-shadow: 0 8px 22px rgba(0,0,0,0.22); }
.gradio-container button:active { transform: translateY(0px); box-shadow: 0 4px 14px rgba(0,0,0,0.18); }

/* Loading panel */
.tl-loading {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 12px;
  border-radius: 12px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.03);
  color: rgba(255,255,255,0.90);
  font-weight: 800;
}
.tl-spinner {
  width: 16px;
  height: 16px;
  border-radius: 999px;
  border: 2px solid rgba(255,255,255,0.22);
  border-top-color: rgba(255,255,255,0.92);
  animation: tlspin 0.9s linear infinite;
  flex: 0 0 auto;
}
@keyframes tlspin { to { transform: rotate(360deg); } }
.tl-loading-sub { color: rgba(255,255,255,0.70); font-weight: 750; }

/* System mode badge near Analyze button */
.tl-mode-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  height: 26px;
  padding: 0 10px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.18);
  background: rgba(255,255,255,0.06);
  color: rgba(255,255,255,0.90);
  font-weight: 900;
  font-size: 12px;
  letter-spacing: 0.01em;
  white-space: nowrap;
}
.tl-mode-badge.llm { border-color: rgba(59,130,246,0.55); background: rgba(59,130,246,0.14); }
.tl-mode-badge.hybrid { border-color: rgba(34,197,94,0.55); background: rgba(34,197,94,0.14); }
.tl-mode-badge.fallback { border-color: rgba(234,179,8,0.55); background: rgba(234,179,8,0.14); }

/* Dark history table (custom HTML) */
.tl-history {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 12px;
  overflow: hidden;
}
.tl-history-scroll {
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
}
.tl-history-table {
  width: 100%;
  border-collapse: collapse;
  color: rgba(255,255,255,0.90);
  min-width: 560px; /* enables horizontal scroll on mobile */
}
.tl-history-table thead th {
  background: rgba(255,255,255,0.06);
  color: rgba(255,255,255,0.95);
  text-align: left;
  font-weight: 900;
  padding: 12px 12px;
  border-bottom: 1px solid rgba(255,255,255,0.10);
  white-space: nowrap;
}
.tl-history-table tbody td {
  padding: 12px 12px;
  border-bottom: 1px solid rgba(255,255,255,0.06);
  vertical-align: top;
}
.tl-history-table tbody tr:nth-child(odd) td { background: rgba(255,255,255,0.02); }
.tl-history-table tbody tr:hover td { background: rgba(255,255,255,0.04); }
.tl-history-table tbody tr:last-child td { border-bottom: none; }
.tl-history-query { color: rgba(255,255,255,0.92); font-weight: 750; }
.tl-history-num { font-variant-numeric: tabular-nums; font-weight: 850; }
.tl-history-muted { color: rgba(255,255,255,0.70); }

/* Minimal execution trace timeline */
.tl-trace {
  margin-top: 6px;
  padding-left: 10px;
  border-left: 2px solid rgba(255,255,255,0.10);
}
.tl-trace-item {
  position: relative;
  display: flex;
  align-items: baseline;
  gap: 10px;
  padding: 8px 0 8px 14px;
}
.tl-trace-item::before {
  content: "";
  position: absolute;
  left: -7px;
  top: 16px;
  width: 10px;
  height: 10px;
  border-radius: 999px;
  background: rgba(255,255,255,0.18);
  border: 2px solid rgba(255,255,255,0.18);
}
.tl-trace-item.success::before { background: var(--tl-green); border-color: var(--tl-green); }
.tl-trace-item.fallback::before { background: var(--tl-yellow); border-color: var(--tl-yellow); }
.tl-trace-item.failed::before { background: var(--tl-red); border-color: var(--tl-red); }
.tl-trace-label { font-weight: 850; color: rgba(255,255,255,0.92); }
.tl-trace-ms { margin-left: auto; color: rgba(255,255,255,0.70); font-weight: 800; font-variant-numeric: tabular-nums; }
.tl-trace-icon { width: 18px; display: inline-flex; justify-content: center; }

/* Yellow warning banner + badges (fallback results) */
.tl-banner-warning {
  border: 1px solid rgba(234, 179, 8, 0.35);
  background: rgba(234, 179, 8, 0.12);
  color: rgba(255,255,255,0.92);
  border-radius: 14px;
  padding: 12px 14px;
  font-weight: 800;
}
.tl-banner-warning strong { color: #fef08a; }
.tl-fallback-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 2px 8px;
  margin-left: 8px;
  border-radius: 999px;
  border: 1px solid rgba(234, 179, 8, 0.45);
  background: rgba(234, 179, 8, 0.14);
  color: rgba(255,255,255,0.92);
  font-weight: 900;
  font-size: 12px;
  letter-spacing: 0.01em;
}

.tl-breakdown {
  margin-top: 10px;
  padding: 12px;
  border-radius: 12px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.03);
  color: rgba(255,255,255,0.86);
  font-weight: 650;
  font-size: 0.94rem;
  line-height: 1.45;
}
.tl-breakdown h4 { margin: 0 0 8px 0; font-size: 0.98rem; color: rgba(255,255,255,0.95); }
.tl-breakdown ul { margin: 0; padding-left: 18px; }
.tl-breakdown li { margin: 4px 0; }
"""


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _interpret_trust_pct(pct: int) -> tuple[str, str]:
    """
    Trust Score thresholds (inclusive):
    0–40 → Low Trust 🔴
    40–70 → Medium Trust 🟡
    70–100 → High Trust 🟢
    """
    p = max(0, min(100, int(pct)))
    if p >= 70:
        return "🟢", "High Trust"
    if p >= 40:
        return "🟡", "Medium Trust"
    return "🔴", "Low Trust"


def _interpret_geo_pct(pct: int) -> tuple[str, str]:
    """
    GEO Score thresholds (inclusive):
    0–40 → Low Coverage
    40–70 → Moderate Coverage
    70–100 → High Coverage
    """
    p = max(0, min(100, int(pct)))
    if p >= 70:
        return "🟢", "High Coverage"
    if p >= 40:
        return "🟡", "Moderate Coverage"
    return "🔴", "Low Coverage"


def _metric_interpretation_line(name: str, pct: int, kind: str) -> tuple[str, str]:
    """
    Returns (emoji_for_title, display_line).
    Example: "Trust Score: 80% (High Trust 🟢)"
    """
    if kind == "trust":
        emoji, label = _interpret_trust_pct(pct)
        return emoji, f"{name}: {pct}% ({label} {emoji})"
    if kind == "geo":
        emoji, label = _interpret_geo_pct(pct)
        return emoji, f"{name}: {pct}% ({label} {emoji})"
    return "•", f"{name}: {pct}%"


def _score_tier(score_0_to_1: float) -> tuple[str, str]:
    s = _clamp01(score_0_to_1)
    if s >= 0.85:
        return "🟢", "High"
    if s >= 0.70:
        return "🟡", "Medium"
    return "🔴", "Low"


def _metric_html(name: str, score_0_to_1: float, kind: str) -> str:
    s = _clamp01(score_0_to_1)
    pct = int(round(s * 100))
    emoji, line = _metric_interpretation_line(name, pct, kind)
    return "\n".join(
        [
            '<div class="tl-metric">',
            f'  <div class="tl-metric-top"><div class="tl-metric-name">{emoji} {name}</div><div class="tl-metric-val">{pct}%</div></div>',
            f'  <div class="tl-bar {kind}"><div style="width: {pct}%;"></div></div>',
            f'  <div class="tl-metric-sub">{line}</div>',
            "</div>",
        ]
    )


def _pair_label(trust: float, geo: float) -> str:
    t_pct = int(round(_clamp01(trust) * 100))
    g_pct = int(round(_clamp01(geo) * 100))
    _t_emoji, t_label = _interpret_trust_pct(t_pct)
    _g_emoji, g_label = _interpret_geo_pct(g_pct)
    return f"{t_label} / {g_label}"


def _format_ranking_top3(ranking: list[str], *, is_fallback: bool = False) -> str:
    items = [str(x).strip() for x in (ranking or []) if str(x or "").strip()]
    if not items:
        return "_No results yet._"
    medals = ["🥇", "🥈", "🥉"]
    lines: list[str] = []
    for i, name in enumerate(items[:3]):
        badge = " <span class='tl-fallback-badge'>⚠ Fallback</span>" if is_fallback else ""
        lines.append(f"{medals[i]} **{name}**{badge}")
    return "\n".join(lines).strip()


def _format_ranking_detailed(
    ranking: list[str],
    product_notes: dict[str, str] | None,
    *,
    is_fallback: bool = False,
) -> str:
    items = [str(x).strip() for x in (ranking or []) if str(x or "").strip()]
    if not items:
        return "_No results yet. Run **Analyze** to simulate retrieval + ranking._"
    notes = product_notes or {}
    lines: list[str] = []
    for i, name in enumerate(items):
        badge = " <span class='tl-fallback-badge'>⚠ Fallback</span>" if is_fallback and i == 0 else ""
        note = (notes.get(name) or "").strip()
        lead = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"**{i+1}.**"
        if note:
            safe = html.escape(note)
            lines.append(f"{lead} **{name}**{badge}  \n_{safe}_")
        else:
            lines.append(f"{lead} **{name}**{badge}")
    return "\n\n".join(lines).strip()


def _interpret_confidence_pct(pct: int) -> tuple[str, str]:
    p = max(0, min(100, int(pct)))
    if p >= 90:
        return "🟢", "High"
    if p >= 60:
        return "🟡", "Medium"
    return "🔴", "Low"


def _confidence_metric_html(score_0_to_100: float) -> str:
    pct = max(0, min(100, int(round(float(score_0_to_100)))))
    emoji, label = _interpret_confidence_pct(pct)
    frac = pct / 100.0
    line = f"Confidence: {pct}% ({label} {emoji})"
    return "\n".join(
        [
            '<div class="tl-metric">',
            f'  <div class="tl-metric-top"><div class="tl-metric-name">{emoji} Confidence</div>'
            f'<div class="tl-metric-val">{pct}%</div></div>',
            f'  <div class="tl-bar conf"><div style="width: {pct}%;"></div></div>',
            f'  <div class="tl-metric-sub">{html.escape(line)}</div>',
            "</div>",
        ]
    )


def _breakdown_html(trust_bd: dict[str, Any] | None, geo_bd: dict[str, Any] | None) -> str:
    def rows(d: dict[str, Any] | None, title: str) -> str:
        if not d:
            return ""
        parts = [f"<h4>{html.escape(title)}</h4>", "<ul>"]
        for k, v in d.items():
            try:
                pts = int(v)
            except Exception:
                pts = 0
            parts.append(f"<li><strong>{html.escape(str(k))}:</strong> +{pts}</li>")
        parts.append("</ul>")
        return "\n".join(parts)

    t = rows(trust_bd, "Trust score breakdown")
    g = rows(geo_bd, "GEO score breakdown")
    if not t and not g:
        return "<div class='tl-breakdown tl-foot'>—</div>"
    return f'<div class="tl-breakdown">{t}{g}</div>'


def _agent_failure_banner_html(agent_failed: bool, failed_key: str | None) -> str:
    if not agent_failed or not (failed_key or "").strip():
        return ""
    label = str(failed_key).strip().title()
    return (
        "<div class='tl-banner-warning'>"
        f"<strong>⚠</strong> Simulated agent failure: <strong>{html.escape(label)}</strong> — "
        "pipeline continued with fallback signals."
        "</div>"
    )


def _fallback_banner_html(is_fallback: bool) -> str:
    if not is_fallback:
        return ""
    return (
        "<div class='tl-banner-warning'>"
        "<strong>⚠</strong> Some results generated using fallback logic due to LLM/RAG limitations."
        "</div>"
    )


def _plot_trend(values: list[float], title: str, color: str):
    vals = [max(0.0, min(1.0, float(v))) for v in (values or [])]
    if len(vals) < 4:
        pad_rng = random.Random(17)
        while len(vals) < 4:
            anchor = vals[-1] if vals else pad_rng.uniform(0.52, 0.72)
            vals.append(max(0.08, min(0.95, anchor + pad_rng.uniform(-0.06, 0.06))))
    vals = vals[-4:]

    x = [1, 2, 3, 4]
    y = [v * 100.0 for v in vals]

    fig, ax = plt.subplots(figsize=(5.2, 2.6), dpi=140)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    ax.plot(x, y, color=color, linewidth=2.6, marker="o", markersize=5)
    ax.fill_between(x, y, [min(y)] * len(y), color=color, alpha=0.15)

    ax.set_ylim(0, 100)
    ax.set_xticks(x, labels=["T-3", "T-2", "T-1", "Now"])
    ax.tick_params(axis="x", colors="#e5e7eb", labelsize=9)
    ax.tick_params(axis="y", colors="#94a3b8", labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.18)
    ax.set_title(title, fontsize=10.5, color="#e5e7eb", pad=8)

    for xi, yi in zip(x, y):
        ax.text(xi, min(yi + 3, 98), f"{int(round(yi))}%", ha="center", va="bottom", fontsize=8.5, color="#e5e7eb")

    fig.tight_layout(pad=0.8)
    return fig


def render_history_table(history: list[dict[str, Any]]) -> str:
    items = list(history or [])[:5]

    if not items:
        return """
<div class="tl-history">
  <div class="tl-history-scroll">
    <div style="padding: 14px 12px;" class="tl-history-muted">
      No queries yet. Run an analysis to populate this table.
    </div>
  </div>
</div>
""".strip()

    rows: list[str] = []
    for item in items:
        q = html.escape(str(item.get("query") or "").strip())
        trust = int(round(max(0.0, min(1.0, float(item.get("trust") or 0.0))) * 100))
        geo = int(round(max(0.0, min(1.0, float(item.get("geo") or 0.0))) * 100))
        rows.append(
            "\n".join(
                [
                    "<tr>",
                    f'  <td class="tl-history-query">{q}</td>',
                    f'  <td class="tl-history-num">{trust}%</td>',
                    f'  <td class="tl-history-num">{geo}%</td>',
                    "</tr>",
                ]
            )
        )

    return "\n".join(
        [
            '<div class="tl-history">',
            '  <div class="tl-history-scroll">',
            '    <table class="tl-history-table">',
            "      <thead>",
            "        <tr>",
            "          <th>Query</th>",
            "          <th>Trust</th>",
            "          <th>GEO</th>",
            "        </tr>",
            "      </thead>",
            "      <tbody>",
            "\n".join(f"        {r}" for r in rows),
            "      </tbody>",
            "    </table>",
            "  </div>",
            "</div>",
        ]
    )


def _trace_html(trace: Any) -> str:
    """
    Render API response field `trace` as a minimal vertical timeline.
    Accepts a list of dicts (preferred) or a list of strings.
    """
    items: list[dict[str, Any]] = []
    if isinstance(trace, list):
        for t in trace:
            if isinstance(t, dict):
                items.append(t)
            elif isinstance(t, str):
                items.append({"name": t})

    def norm_status(x: Any) -> str:
        s = str(x or "").strip().lower()
        if s in {"success", "ok", "pass", "passed"}:
            return "success"
        if s in {"fallback", "warn", "warning", "degraded"}:
            return "fallback"
        if s in {"failed", "fail", "error", "err"}:
            return "failed"
        return "success"

    def icon_for(st: str) -> str:
        if st == "success":
            return "✔"
        if st == "fallback":
            return "⚠"
        return "✖"

    if not items:
        return "<div class='tl-trace tl-foot'>—</div>"

    rows: list[str] = ["<div class='tl-trace'>"]
    for it in items:
        name = html.escape(str(it.get("name") or it.get("agent") or it.get("step") or "Agent"))
        st = norm_status(it.get("status"))
        ms_raw = it.get("duration_ms", it.get("ms", it.get("latency_ms")))
        ms = ""
        try:
            if ms_raw is not None and ms_raw != "":
                ms = f"{int(float(ms_raw))} ms"
        except Exception:
            ms = ""
        rows.append(
            "\n".join(
                [
                    f"<div class='tl-trace-item {st}'>",
                    f"  <span class='tl-trace-icon'>{icon_for(st)}</span>",
                    f"  <span class='tl-trace-label'>{name}</span>",
                    f"  <span class='tl-trace-ms'>{html.escape(ms) if ms else ''}</span>",
                    "</div>",
                ]
            )
        )
    rows.append("</div>")
    return "\n".join(rows)


def begin_run(query: str) -> tuple[gr.update, str, gr.update]:
    q = (query or "").strip()
    if not q:
        return gr.update(interactive=True), "⚠️ Please enter a query to analyze.", gr.update(visible=False)
    return (
        gr.update(interactive=False),
        "Agents are working... orchestrating the multi-agent pipeline.",
        gr.update(visible=True),
    )


def end_run() -> tuple[gr.update, gr.update]:
    return gr.update(interactive=True), gr.update(visible=False)


def _mode_badge_html(mode: str) -> str:
    m = (mode or "").strip()
    cls = "llm"
    label = "LLM Mode"
    if m == "Hybrid Mode":
        cls = "hybrid"
        label = "Hybrid Mode (LLM + RAG)"
    elif m == "Fallback Mode":
        cls = "fallback"
        label = "Fallback Mode"
    return f"<span class='tl-mode-badge {cls}'>🧩 {html.escape(label)}</span>"


def _infer_mode_from_debug(debug: dict[str, Any]) -> str:
    """
    Mode precedence:
      - If fallback triggered -> Fallback Mode
      - Else if RAG used -> Hybrid Mode
      - Else -> LLM Mode
    """
    retrieval = debug.get("retrieval_output") if isinstance(debug, dict) else None
    if not isinstance(retrieval, dict):
        return "LLM Mode"

    notes = str(retrieval.get("notes") or "").strip()
    if notes == "Fallback":
        return "Fallback Mode"

    retrieved_docs = retrieval.get("retrieved_documents")
    rag_used = bool(retrieved_docs) and isinstance(retrieved_docs, list)
    return "Hybrid Mode" if rag_used else "LLM Mode"


def _pretty_json_payload(x: Any) -> Any:
    """
    Return a JSON-friendly payload for `gr.JSON`.
    If `x` is a plain string, wrap it so it still renders nicely as JSON.
    """
    if x is None:
        return {}
    if isinstance(x, str):
        return {"text": x}
    return x


def toggle_debug(show: bool):
    v = bool(show)
    return (gr.update(visible=v),) * 5


def _alerts_html(is_fallback: bool, agent_failed: bool, failed_key: str | None) -> str:
    return (_agent_failure_banner_html(agent_failed, failed_key) + _fallback_banner_html(is_fallback)).strip()


def analyze_ui(
    query: str,
    history_state: list[dict[str, Any]],
    trust_hist_state: list[float],
    geo_hist_state: list[float],
    simulate_failure: bool,
):
    q = (query or "").strip()
    th0 = list(trust_hist_state or [])
    gh0 = list(geo_hist_state or [])

    def _empty_pack(msg: str):
        return (
            msg,
            "",
            _format_ranking_detailed([], {}),
            _metric_html("Trust Score", 0.0, "trust"),
            _metric_html("GEO Score", 0.0, "geo"),
            _confidence_metric_html(0.0),
            _breakdown_html(None, None),
            "<span class='tl-pill'>—</span>",
            _trace_html([]),
            "_No explanation yet._",
            _plot_trend(th0, "Trust Trend", "#22c55e"),
            _plot_trend(gh0, "GEO Trend", "#3b82f6"),
            render_history_table(history_state),
            history_state,
            th0,
            gh0,
            _mode_badge_html("LLM Mode"),
            {},
            {},
            {},
            {},
            {},
        )

    if not q:
        return _empty_pack("⚠️ Please enter a query to analyze.")

    u = gr.update()
    result = simulate_agents(q, simulate_failure=bool(simulate_failure))

    trace_full = list(result.get("trace") or [])
    ranking = list(result.get("ranking") or [])
    notes_map = result.get("product_notes") if isinstance(result.get("product_notes"), dict) else {}
    trust_val = float(result.get("trust_score") or 0.0)
    geo_val = float(result.get("geo_score") or 0.0)
    conf_int = int(result.get("confidence_score") or 0)
    explanation = str(result.get("explanation") or "").strip() or "_No explanation yet._"
    debug = result.get("debug") or {}
    notes = ""
    try:
        notes = str((debug.get("retrieval_output") or {}).get("notes") or "").strip()
    except Exception:
        notes = ""
    is_fallback = notes == "Fallback" or bool(result.get("used_fallback"))
    agent_failed = bool(result.get("agent_failed"))
    failed_key = str(result.get("failed_agent_key") or "") or None
    mode = _infer_mode_from_debug(debug)

    th = list(th0)
    gh = list(gh0)
    if len(th) == 0:
        br = random.Random(2026)
        th = [br.uniform(0.55, 0.74) for _ in range(3)]
        gh = [br.uniform(0.48, 0.68) for _ in range(3)]
    th.append(trust_val)
    gh.append(geo_val)
    th, gh = th[-4:], gh[-4:]

    history = list(history_state or [])
    history.insert(0, {"query": q, "trust": trust_val, "geo": geo_val})
    history = history[:5]

    conf_emoji, conf_lbl = _interpret_confidence_pct(conf_int)
    pill = (
        f"<span class='tl-pill'>🏷️ {_pair_label(trust_val, geo_val)} · "
        f"Confidence: {conf_lbl} {conf_emoji}</span>"
    )

    final_pack = (
        "",
        _alerts_html(is_fallback, agent_failed, failed_key),
        _format_ranking_detailed(ranking, notes_map, is_fallback=is_fallback),
        _metric_html("Trust Score", trust_val, "trust"),
        _metric_html("GEO Score", geo_val, "geo"),
        _confidence_metric_html(float(conf_int)),
        _breakdown_html(result.get("trust_breakdown"), result.get("geo_breakdown")),
        pill,
        _trace_html(trace_full),
        explanation,
        _plot_trend(th, "Trust Trend", "#22c55e"),
        _plot_trend(gh, "GEO Trend", "#3b82f6"),
        render_history_table(history),
        history,
        th,
        gh,
        _mode_badge_html(mode),
        _pretty_json_payload(debug.get("retrieval_output")),
        _pretty_json_payload(debug.get("ranking_raw_llm_output")),
        _pretty_json_payload(debug.get("trust_calculation_steps")),
        _pretty_json_payload(debug.get("geo_calculation_steps")),
        _pretty_json_payload(debug.get("explanation_prompt_output")),
    )

    yield (
        "**Agents are working...** Simulated agents are executing step-by-step.",
        u,
        u,
        u,
        u,
        u,
        u,
        u,
        _trace_html([]),
        u,
        u,
        u,
        u,
        u,
        u,
        u,
        u,
        u,
        u,
        u,
        u,
        u,
    )

    for i in range(len(trace_full)):
        if i > 0:
            time.sleep(random.uniform(0.5, 1.0))
        partial = trace_full[: i + 1]
        done = [str((t or {}).get("name") or "") for t in partial]
        status = "**Agents are working...**\n\n" + "\n".join(f"- {html.escape(x)}" for x in done if x)
        yield (
            status,
            u,
            u,
            u,
            u,
            u,
            u,
            u,
            _trace_html(partial),
            u,
            u,
            u,
            u,
            u,
            u,
            u,
            u,
            u,
            u,
            u,
            u,
            u,
        )

    yield final_pack


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

with gr.Blocks(title=APP_TITLE, theme=theme, css=CSS) as demo:
    history_state = gr.State([])  # session-based history (last 5)
    trust_hist_state = gr.State([])
    geo_hist_state = gr.State([])

    with gr.Column(elem_classes=["tl-wrap"]):
        with gr.Column(elem_classes=["header"]):
            gr.Markdown(
                f"# 🚀 {APP_TITLE}\n\n"
                f"**{TAGLINE}**\n\n"
                f"{SUBTEXT}\n\n"
                "**🧪 Demo Mode:** Simulated Multi-Agent Execution  \n"
                "_No external APIs. This simulates CrewAI-style orchestration._"
            )
            with gr.Row(equal_height=True, elem_classes=["tl-tag-row"]):
                for badge in BADGES:
                    with gr.Column(scale=0, min_width=0):
                        gr.Button(badge, interactive=False, elem_classes=["tag-btn"])
            gr.Markdown("Mini analytics platform UI (simulated). No external API calls.")

        with gr.Group(elem_classes=["tl-card"]):
            gr.Markdown("### 🔎 Input")
            simulate_failure = gr.Checkbox(
                label="⚠ Simulate Agent Failure",
                value=False,
            )
            with gr.Row(equal_height=True):
                with gr.Column(scale=3, min_width=360):
                    query_in = gr.Textbox(
                        label="Enter your query",
                        placeholder="e.g., Best health insurance in India",
                        lines=2,
                        max_lines=4,
                    )
                with gr.Column(scale=1, min_width=170):
                    with gr.Row(equal_height=True):
                        analyze_btn = gr.Button("Analyze", variant="primary")
                        mode_badge = gr.HTML(value=_mode_badge_html("LLM Mode"))

            gr.Markdown("**Examples**")
            gr.Examples(
                examples=[
                    ["Best health insurance in India"],
                    ["Best personal loan provider"],
                    ["Affordable insurance plans"],
                ],
                inputs=[query_in],
                label="",
            )
            loading_panel = gr.HTML(
                value=(
                    "<div class='tl-loading'>"
                    "<div class='tl-spinner'></div>"
                    "<div>"
                    "<div>Agents are working...</div>"
                    "<div class='tl-loading-sub'>Simulated orchestration with staged trace updates.</div>"
                    "</div>"
                    "</div>"
                ),
                visible=False,
            )
            status_md = gr.Markdown(value="")
            show_debug = gr.Checkbox(label="Show debug info", value=False)

        with gr.Row(equal_height=True):
            with gr.Column():
                with gr.Group(elem_classes=["tl-card"]):
                    gr.Markdown("### 📊 Ranking")
                    fallback_banner = gr.HTML(value="")
                    ranked_out = gr.Markdown(value=_format_ranking_detailed([], {}))

            with gr.Column():
                with gr.Group(elem_classes=["tl-card"]):
                    gr.Markdown("### 📈 Scores")
                    with gr.Row():
                        trust_html = gr.HTML(value=_metric_html("Trust Score", 0.0, "trust"))
                        geo_html = gr.HTML(value=_metric_html("GEO Score", 0.0, "geo"))
                        confidence_html = gr.HTML(value=_confidence_metric_html(0.0))
                    breakdown_html = gr.HTML(value=_breakdown_html(None, None))
                    label_pill = gr.HTML(value="<span class='tl-pill'>—</span>")
                with gr.Group(elem_classes=["tl-card"]):
                    gr.Markdown("### 🧠 Agent Execution Trace")
                    trace_html = gr.HTML(value=_trace_html([]))

            with gr.Column():
                with gr.Group(elem_classes=["tl-card"]):
                    gr.Markdown("### 💡 Explanation")
                    explanation_out = gr.Markdown(value="_No explanation yet._")

        with gr.Group(elem_classes=["tl-card"]):
            gr.Markdown("### 🧩 Debug info")
            gr.Markdown("Hidden by default to avoid clutter. Enable **Show debug info** above to view.", elem_classes=["tl-foot"])

            with gr.Accordion("1. Retrieval Output", open=False, visible=False) as acc_retrieval:
                dbg_retrieval = gr.JSON(value={})
            with gr.Accordion("2. Ranking Raw LLM Output", open=False, visible=False) as acc_ranking_raw:
                dbg_ranking_raw = gr.JSON(value={})
            with gr.Accordion("3. Trust Calculation Steps", open=False, visible=False) as acc_trust_steps:
                dbg_trust_steps = gr.JSON(value={})
            with gr.Accordion("4. GEO Calculation Steps", open=False, visible=False) as acc_geo_steps:
                dbg_geo_steps = gr.JSON(value={})
            with gr.Accordion("5. Explanation Prompt + Output", open=False, visible=False) as acc_expl:
                dbg_expl = gr.JSON(value={})

        gr.HTML("<div class='tl-divider'></div>")

        with gr.Group(elem_classes=["tl-card"]):
            gr.Markdown("### 🔥 Trends")
            with gr.Row():
                trust_trend_plot = gr.Plot(value=_plot_trend([], "Trust Trend", "#22c55e"))
                geo_trend_plot = gr.Plot(value=_plot_trend([], "GEO Trend", "#3b82f6"))

        gr.HTML("<div class='tl-divider'></div>")

        with gr.Group(elem_classes=["tl-card"]):
            gr.Markdown("### 🧾 Query History (Session)")
            gr.Markdown("Last 5 queries for this session.", elem_classes=["tl-foot"])
            history_table = gr.HTML(value=render_history_table([]))

        gr.HTML("<hr />")
        gr.Markdown(
            "**Powered by Multi-Agent AI (CrewAI + RAG)**  \nDesigned for scalable AI decision intelligence systems",
            elem_classes=["tl-foot"],
        )

    (
        analyze_btn.click(
            fn=begin_run,
            inputs=[query_in],
            outputs=[analyze_btn, status_md, loading_panel],
            queue=False,
        )
        .then(
            fn=analyze_ui,
            inputs=[query_in, history_state, trust_hist_state, geo_hist_state, simulate_failure],
            outputs=[
                status_md,
                fallback_banner,
                ranked_out,
                trust_html,
                geo_html,
                confidence_html,
                breakdown_html,
                label_pill,
                trace_html,
                explanation_out,
                trust_trend_plot,
                geo_trend_plot,
                history_table,
                history_state,
                trust_hist_state,
                geo_hist_state,
                mode_badge,
                dbg_retrieval,
                dbg_ranking_raw,
                dbg_trust_steps,
                dbg_geo_steps,
                dbg_expl,
            ],
            show_progress="full",
        )
        .then(fn=end_run, outputs=[analyze_btn, loading_panel], queue=False)
    )

    show_debug.change(
        fn=toggle_debug,
        inputs=[show_debug],
        outputs=[acc_retrieval, acc_ranking_raw, acc_trust_steps, acc_geo_steps, acc_expl],
        queue=False,
    )


if __name__ == "__main__":
    demo.queue()
    demo.launch()

