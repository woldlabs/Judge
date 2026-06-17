"""
Professional technical report generation for Judge.

Produces high-quality PDF + Markdown + JSON artifacts.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

from judge.core.models import AnalysisResult, AnomalyEvent, Modality


def generate_report(
    result: AnalysisResult,
    output_path: str | Path,
    format: str = "pdf",
    include_plots: bool = True,
) -> Path:
    """
    Generate report artifacts.

    format: "pdf", "markdown", or "all"
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if format in ("pdf", "all"):
        pdf_path = out.with_suffix(".pdf")
        _generate_pdf(result, pdf_path, include_plots=include_plots)
        if format == "pdf":
            return pdf_path

    md_path = out.with_suffix(".md")
    _generate_markdown(result, md_path)

    json_path = out.with_suffix(".json")
    result.save_json(str(json_path))

    if format == "markdown":
        return md_path
    return out


def _generate_markdown(result: AnalysisResult, path: Path) -> None:
    lines = []
    lines.append(f"# Judge Analysis Report")
    lines.append(f"**Session:** `{result.session_id}`")
    lines.append(f"**Generated:** {result.timestamp}")
    lines.append(f"**Total duration analyzed:** {result.duration_seconds:.1f} s")
    lines.append("")
    lines.append("## Files Processed")
    for f in result.files_processed:
        lines.append(f"- `{f}`")
    lines.append("")

    lines.append("## Executive Summary")
    total_events = len(result.events)
    lines.append(f"- **Total candidate events:** {total_events}")
    for mod, stats in result.modality_stats.items():
        lines.append(f"- **{mod}**: {stats.get('event_count', 0)} events, max score {stats.get('max_score', 0):.2f}")
    lines.append("")

    lines.append("## Ranked Events")
    for i, ev in enumerate(sorted(result.events, key=lambda e: -e.score)[:50], 1):
        lines.append(f"### {i}. {ev.modality.value.upper()} — {ev.pretty_time()} (+{ev.duration*1000:.0f} ms)")
        lines.append(f"**Score:** {ev.score:.2f} (peak {ev.peak_score:.2f})")
        lines.append(f"**File:** `{ev.file_path}`")
        lines.append(f"**Description:** {ev.description}")
        if ev.tags:
            lines.append(f"**Tags:** {', '.join(ev.tags)}")
        if ev.features:
            feat_str = ", ".join(f"{k}={v:.3g}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in list(ev.features.items())[:6])
            lines.append(f"**Key features:** {feat_str}")
        lines.append("")

    lines.append("## Parameters")
    for k, v in result.parameters.items():
        lines.append(f"- {k}: {v}")

    path.write_text("\n".join(lines), encoding="utf-8")


def _generate_pdf(result: AnalysisResult, path: Path, include_plots: bool = True) -> None:
    doc = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        rightMargin=0.6*inch,
        leftMargin=0.6*inch,
        topMargin=0.6*inch,
        bottomMargin=0.6*inch,
        title=f"Judge Report {result.session_id}",
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="TitleMain",
        parent=styles["Title"],
        fontSize=18,
        spaceAfter=6,
        textColor=colors.HexColor("#1a1a2e"),
    ))
    styles.add(ParagraphStyle(
        name="SectionHeader",
        parent=styles["Heading2"],
        fontSize=13,
        spaceBefore=14,
        spaceAfter=6,
        textColor=colors.HexColor("#16213e"),
    ))
    styles.add(ParagraphStyle(
        name="EventTitle",
        parent=styles["Heading3"],
        fontSize=10,
        spaceBefore=8,
        spaceAfter=2,
        textColor=colors.HexColor("#0f3460"),
    ))
    styles.add(ParagraphStyle(
        name="BodyTextTight",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=11,
        spaceAfter=2,
    ))
    styles.add(ParagraphStyle(
        name="SmallMono",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=7.5,
    ))

    story = []

    # Header
    story.append(Paragraph("JUDGE — Joint Unconventional Data &amp; Geophysical Examination", styles["TitleMain"]))
    story.append(Paragraph(f"Technical Analysis Report — Session <font face='Courier'>{result.session_id}</font>", styles["Normal"]))
    story.append(Paragraph(f"Generated: {result.timestamp} &nbsp;&nbsp;|&nbsp;&nbsp; Duration analyzed: {result.duration_seconds:.1f}s", styles["Normal"]))
    story.append(Spacer(1, 12))

    # Summary table
    story.append(Paragraph("Executive Summary", styles["SectionHeader"]))
    summary_data = [
        ["Total Events", str(len(result.events))],
        ["Files Analyzed", str(len(result.files_processed))],
    ]
    for mod, st in result.modality_stats.items():
        summary_data.append([
            f"{mod.capitalize()} Events",
            f"{st.get('event_count',0)} (max score {st.get('max_score',0):.2f})"
        ])

    t = Table(summary_data, colWidths=[2.2*inch, 4.5*inch])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8e8e8")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    # Files
    story.append(Paragraph("Data Provenance", styles["SectionHeader"]))
    for fp in result.files_processed[:12]:
        story.append(Paragraph(f"<font face='Courier' size='8'>{fp}</font>", styles["BodyTextTight"]))
    if len(result.files_processed) > 12:
        story.append(Paragraph(f"... and {len(result.files_processed)-12} more files", styles["BodyTextTight"]))

    story.append(Spacer(1, 8))

    # Events
    story.append(Paragraph("Detected Events (Ranked by Composite Score)", styles["SectionHeader"]))

    sorted_events = sorted(result.events, key=lambda e: -e.score)[:60]

    for ev in sorted_events:
        header = f"{ev.modality.value.upper()}  |  {ev.pretty_time()}  +{ev.duration*1000:.0f} ms  |  score={ev.score:.2f}"
        story.append(Paragraph(f"<b>{header}</b>", styles["EventTitle"]))
        story.append(Paragraph(ev.description, styles["BodyTextTight"]))
        story.append(Paragraph(f"<font size='7'>File: {Path(ev.file_path).name}</font>", styles["BodyTextTight"]))
        if ev.features:
            feat = ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in list(ev.features.items())[:5])
            story.append(Paragraph(f"<font size='7' color='#444444'>features: {feat}</font>", styles["BodyTextTight"]))
        story.append(Spacer(1, 3))

    if include_plots and result.events:
        story.append(PageBreak())
        story.append(Paragraph("Distribution Overview", styles["SectionHeader"]))
        plot_path = _create_score_distribution_plot(result)
        if plot_path:
            story.append(Image(str(plot_path), width=6.5*inch, height=3.2*inch))
            story.append(Spacer(1, 8))

    # Parameters footer
    story.append(Spacer(1, 16))
    story.append(Paragraph("Analysis Parameters", styles["SectionHeader"]))
    param_text = " | ".join(f"{k}={v}" for k, v in result.parameters.items())
    story.append(Paragraph(f"<font size='8'>{param_text}</font>", styles["BodyTextTight"]))

    story.append(Spacer(1, 20))
    story.append(Paragraph(
        "<i>This report was produced by the Judge framework. All events are statistical candidates only. "
        "Interpret with domain expertise and additional corroborating data.</i>",
        ParagraphStyle("Disclaimer", parent=styles["Normal"], fontSize=7, textColor=colors.gray)
    ))

    doc.build(story)


def _create_score_distribution_plot(result: AnalysisResult) -> Optional[Path]:
    """Generate a small distribution plot for the PDF."""
    try:
        import tempfile
        tmpdir = Path(tempfile.mkdtemp())
        fig, ax = plt.subplots(figsize=(8, 3.8))

        if not result.events:
            return None

        scores = [e.score for e in result.events]
        mods = [e.modality.value for e in result.events]

        colors_map = {"video": "#e63946", "audio": "#457b9d", "sensor": "#2a9d8f"}
        c = [colors_map.get(m, "#333") for m in mods]

        ax.scatter(range(len(scores)), sorted(scores, reverse=True), c=c, s=18, alpha=0.75)
        ax.set_xlabel("Event rank")
        ax.set_ylabel("Composite anomaly score")
        ax.set_title("Event Score Distribution (color = modality)")
        ax.grid(True, alpha=0.3)

        # legend
        for m, col in colors_map.items():
            ax.scatter([], [], c=col, label=m, s=25)
        ax.legend(loc="upper right", fontsize=7)

        p = tmpdir / "score_dist.png"
        plt.tight_layout()
        plt.savefig(p, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return p
    except Exception:
        return None
