from __future__ import annotations

import html
import time
from typing import Any

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt

from utils import run_trustlens

matplotlib.use("Agg")

APP_TITLE = "TrustLens AI"
TAGLINE = "LLM Trust & GEO Intelligence"

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
"""


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


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
    emoji, tier = _score_tier(s)
    return "\n".join(
        [
            '<div class="tl-metric">',
            f'  <div class="tl-metric-top"><div class="tl-metric-name">{emoji} {name}</div><div class="tl-metric-val">{pct}%</div></div>',
            f'  <div class="tl-bar {kind}"><div style="width: {pct}%;"></div></div>',
            f'  <div class="tl-metric-sub">{tier} {name}</div>',
            "</div>",
        ]
    )


def _pair_label(trust: float, geo: float) -> str:
    _t_emoji, t_tier = _score_tier(trust)
    _g_emoji, g_tier = _score_tier(geo)
    return f"{t_tier} Trust / {g_tier} GEO"


def _format_ranking_top3(ranking: list[str]) -> str:
    items = [str(x).strip() for x in (ranking or []) if str(x or "").strip()]
    if not items:
        return "_No results yet._"
    medals = ["🥇", "🥈", "🥉"]
    lines: list[str] = []
    for i, name in enumerate(items[:3]):
        lines.append(f"{medals[i]} **{name}**")
    return "\n".join(lines).strip()


def _plot_trend(values: list[float], title: str, color: str):
    vals = [float(v) for v in (values or [])][:4]
    if len(vals) < 4:
        vals = (vals + [0.0, 0.0, 0.0, 0.0])[:4]

    x = [1, 2, 3, 4]
    y = [max(0.0, min(1.0, v)) * 100.0 for v in vals]

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


def begin_run(query: str) -> tuple[gr.update, str]:
    q = (query or "").strip()
    if not q:
        return gr.update(interactive=True), "⚠️ Please enter a query to analyze."
    return gr.update(interactive=False), ""


def end_run() -> gr.update:
    return gr.update(interactive=True)


def analyze_ui(query: str, history_state: list[dict[str, Any]]):
    q = (query or "").strip()
    if not q:
        return (
            "⚠️ Please enter a query to analyze.",
            _format_ranking_top3([]),
            _metric_html("Trust Score", 0.0, "trust"),
            _metric_html("GEO Score", 0.0, "geo"),
            "<span class='tl-pill'>—</span>",
            "_No explanation yet._",
            _plot_trend([0.0, 0.0, 0.0, 0.0], "Trust Trend", "#22c55e"),
            _plot_trend([0.0, 0.0, 0.0, 0.0], "GEO Trend", "#3b82f6"),
            render_history_table(history_state),
            history_state,
        )

    time.sleep(0.25)  # demo realism, still < 1s
    result = run_trustlens(q)
    ranking = list(result.get("ranking") or [])
    trust_val = float(result.get("trust_score") or 0.0)
    geo_val = float(result.get("geo_score") or 0.0)
    explanation = str(result.get("explanation") or "").strip() or "_No explanation yet._"

    trend = result.get("trend") or {}
    trust_trend = list(trend.get("trust") or [])
    geo_trend = list(trend.get("geo") or [])

    history = list(history_state or [])
    history.insert(0, {"query": q, "trust": trust_val, "geo": geo_val})
    history = history[:5]

    return (
        "",
        _format_ranking_top3(ranking),
        _metric_html("Trust Score", trust_val, "trust"),
        _metric_html("GEO Score", geo_val, "geo"),
        f"<span class='tl-pill'>🏷️ {_pair_label(trust_val, geo_val)}</span>",
        explanation,
        _plot_trend(trust_trend, "Trust Trend", "#22c55e"),
        _plot_trend(geo_trend, "GEO Trend", "#3b82f6"),
        render_history_table(history),
        history,
    )


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

with gr.Blocks(title=APP_TITLE) as demo:
    history_state = gr.State([])  # session-based history (last 5)

    with gr.Column(elem_classes=["tl-wrap"]):
        gr.Markdown(
            f"""
<div class="header">
  <h1>🚀 {APP_TITLE}</h1>
  <p><strong>{TAGLINE}</strong></p>
  <p>Mini analytics platform UI (simulated). No external API calls.</p>
</div>
""".strip()
        )

        with gr.Group(elem_classes=["tl-card"]):
            gr.Markdown("### 🔎 Input")
            with gr.Row(equal_height=True):
                with gr.Column(scale=3, min_width=360):
                    query_in = gr.Textbox(
                        label="Enter your query",
                        placeholder="e.g., Best health insurance in India",
                        lines=2,
                        max_lines=4,
                    )
                with gr.Column(scale=1, min_width=170):
                    analyze_btn = gr.Button("Analyze", variant="primary")

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
            status_md = gr.Markdown(value="")

        with gr.Row(equal_height=True):
            with gr.Column():
                with gr.Group(elem_classes=["tl-card"]):
                    gr.Markdown("### 📊 Ranking")
                    ranked_out = gr.Markdown(value=_format_ranking_top3([]))

            with gr.Column():
                with gr.Group(elem_classes=["tl-card"]):
                    gr.Markdown("### 📈 Scores")
                    with gr.Row():
                        trust_html = gr.HTML(value=_metric_html("Trust Score", 0.0, "trust"))
                        geo_html = gr.HTML(value=_metric_html("GEO Score", 0.0, "geo"))
                    label_pill = gr.HTML(value="<span class='tl-pill'>—</span>")

            with gr.Column():
                with gr.Group(elem_classes=["tl-card"]):
                    gr.Markdown("### 💡 Explanation")
                    explanation_out = gr.Markdown(value="_No explanation yet._")

        gr.HTML("<div class='tl-divider'></div>")

        with gr.Group(elem_classes=["tl-card"]):
            gr.Markdown("### 🔥 Trends")
            with gr.Row():
                trust_trend_plot = gr.Plot(value=_plot_trend([0.0, 0.0, 0.0, 0.0], "Trust Trend", "#22c55e"))
                geo_trend_plot = gr.Plot(value=_plot_trend([0.0, 0.0, 0.0, 0.0], "GEO Trend", "#3b82f6"))

        gr.HTML("<div class='tl-divider'></div>")

        with gr.Group(elem_classes=["tl-card"]):
            gr.Markdown("### 🧾 Query History (Session)")
            gr.Markdown("Last 5 queries for this session.", elem_classes=["tl-foot"])
            history_table = gr.HTML(value=render_history_table([]))

        gr.HTML("<hr />")
        gr.Markdown("**Powered by AI (simulated)**  \nTrustLens AI demo layer — no external APIs, no heavy models.", elem_classes=["tl-foot"])

    (
        analyze_btn.click(fn=begin_run, inputs=[query_in], outputs=[analyze_btn, status_md], queue=False)
        .then(
            fn=analyze_ui,
            inputs=[query_in, history_state],
            outputs=[
                status_md,
                ranked_out,
                trust_html,
                geo_html,
                label_pill,
                explanation_out,
                trust_trend_plot,
                geo_trend_plot,
                history_table,
                history_state,
            ],
            show_progress="full",
        )
        .then(fn=end_run, outputs=[analyze_btn], queue=False)
    )


if __name__ == "__main__":
    demo.queue()
    demo.launch(theme=theme, css=CSS)

