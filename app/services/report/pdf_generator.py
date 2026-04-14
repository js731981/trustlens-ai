from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import get_settings

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _as_list(x: Any) -> list[Any]:
    if isinstance(x, list):
        return x
    if x is None:
        return []
    return [x]


def _safe_text(x: Any) -> str:
    return str(x or "").strip()


def generate_pdf_report(data: dict) -> str:
    """
    Generate a cleanly formatted PDF report and return the file path.

    Expected keys (best-effort):
      - query: str
      - ranked_products: list[dict|str]
      - trust_score: float|str
      - explanation: str (or dict with summary/insights)
      - timestamp: ISO8601 string (optional; auto-filled if missing)
    """
    settings = get_settings()
    out_dir = Path(settings.data_dir) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = _safe_text(data.get("timestamp"))
    if not ts:
        ts = datetime.now(tz=UTC).isoformat()

    filename = f"trustlens_report_{ts.replace(':', '').replace('.', '').replace('Z', '')}.pdf"
    file_path = str(out_dir / filename)

    query = _safe_text(data.get("query"))
    trust_score = data.get("trust_score")
    ranked_products = _as_list(data.get("ranked_products"))
    explanation = data.get("explanation")

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TLTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        spaceAfter=12,
    )
    h_style = ParagraphStyle(
        "TLHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12.5,
        leading=16,
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "TLBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=14,
        spaceAfter=6,
    )
    small_style = ParagraphStyle(
        "TLSmall",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor("#475569"),
        spaceAfter=6,
    )

    doc = SimpleDocTemplate(
        file_path,
        pagesize=LETTER,
        rightMargin=0.85 * inch,
        leftMargin=0.85 * inch,
        topMargin=0.8 * inch,
        bottomMargin=0.8 * inch,
        title="TrustLens AI Report",
        author="TrustLens AI",
    )

    story: list[Any] = []
    story.append(Paragraph("TrustLens AI — Analysis Report", title_style))
    story.append(Paragraph(f"<b>Generated:</b> {ts}", small_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Query", h_style))
    story.append(Paragraph(query or "—", body_style))

    story.append(Paragraph("Trust score", h_style))
    trust_txt = "—" if trust_score is None else _safe_text(trust_score)
    story.append(Paragraph(trust_txt, body_style))

    story.append(Paragraph("Ranked products", h_style))
    if not ranked_products:
        story.append(Paragraph("—", body_style))
    else:
        rows: list[list[str]] = [["Rank", "Product", "Notes"]]
        for i, item in enumerate(ranked_products[:20], start=1):
            if isinstance(item, dict):
                name = _safe_text(item.get("name") or item.get("product") or item.get("title"))
                notes = _safe_text(item.get("reason") or item.get("rationale") or item.get("notes"))
                if not notes and item.get("features") is not None:
                    notes = _safe_text(item.get("features"))
            else:
                name = _safe_text(item)
                notes = ""
            rows.append([str(i), name or "—", notes])

        table = Table(
            rows,
            colWidths=[0.55 * inch, 3.35 * inch, 2.55 * inch],
            hAlign="LEFT",
        )
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 10),
                    ("ALIGN", (0, 0), (0, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.append(table)

    story.append(Spacer(1, 10))
    story.append(Paragraph("Explanation", h_style))
    if isinstance(explanation, dict):
        summary = _safe_text(explanation.get("summary"))
        insights = _as_list(explanation.get("insights"))
        if summary:
            story.append(Paragraph(summary, body_style))
        if insights:
            bullets = ListFlowable(
                [
                    ListItem(Paragraph(_safe_text(x) or "—", body_style), leftIndent=14)
                    for x in insights
                    if _safe_text(x)
                ],
                bulletType="bullet",
                leftIndent=14,
                bulletFontName="Helvetica",
                bulletFontSize=10,
            )
            story.append(bullets)
        if not summary and not insights:
            story.append(Paragraph("—", body_style))
    else:
        exp_txt = _safe_text(explanation)
        story.append(Paragraph(exp_txt or "—", body_style))

    doc.build(story)
    return file_path

