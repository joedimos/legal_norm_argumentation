"""Rebuild the manuscript PDF using ReportLab (optional publication dependency)."""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Flowable, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

INK = colors.HexColor("#152f43")
ACCENT = colors.HexColor("#166b78")
PALE = colors.HexColor("#f0f5f7")
GRAY = colors.HexColor("#526572")
PAGE_W, PAGE_H = A4
WIDTH = PAGE_W - 44 * mm


def fonts(root):
    for name, file in [
        ("Serif", "DejaVuSerif.ttf"),
        ("SerifBold", "DejaVuSerif-Bold.ttf"),
        ("Sans", "DejaVuSans.ttf"),
        ("SansBold", "DejaVuSans-Bold.ttf"),
        ("Mono", "DejaVuSansMono.ttf"),
    ]:
        pdfmetrics.registerFont(TTFont(name, str(root / file)))
    pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="SerifBold", italic="Serif", boldItalic="SerifBold")
    pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="SansBold", italic="Sans", boldItalic="SansBold")


def markup(text):
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r'<font name="Mono" size="8.0">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(
        r"(https://[^\s<>]+)",
        lambda m: (
            '<link href="'
            + m[0].rstrip(".")
            + '" color="#166b78">'
            + m[0].rstrip(".")
            + "</link>"
            + ("." if m[0].endswith(".") else "")
        ),
        text,
    )
    return text


class Diagram(Flowable):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.width = WIDTH
        self.height = 228 if kind == "architecture" else 210

    def draw(self):
        c = self.canv
        scale = self.width / 470
        c.saveState()
        c.scale(scale, scale)

        def node(x, y, w, h, lines, fill=PALE):
            c.setFillColor(fill)
            c.setStrokeColor(colors.HexColor("#bacbd4"))
            c.roundRect(x, y, w, h, 5, fill=1, stroke=1)
            c.setFillColor(INK)
            c.setFont("Sans", 8.0)
            start = y + h / 2 + (len(lines) - 1) * 5
            for line in lines:
                c.drawCentredString(x + w / 2, start - 3, line)
                start -= 10

        def arrow(x1, y1, x2, y2):
            from math import atan2, cos, pi, sin

            c.setStrokeColor(ACCENT)
            c.setLineWidth(0.8)
            c.line(x1, y1, x2, y2)
            a = atan2(y2 - y1, x2 - x1)
            p = c.beginPath()
            p.moveTo(x2, y2)
            p.lineTo(x2 - 4 * cos(a - pi / 7), y2 - 4 * sin(a - pi / 7))
            p.lineTo(x2 - 4 * cos(a + pi / 7), y2 - 4 * sin(a + pi / 7))
            p.close()
            c.setFillColor(ACCENT)
            c.drawPath(p, fill=1, stroke=0)

        if self.kind == "architecture":
            node(0, 176, 135, 35, ["Versioned legal sources", "Exact spans and hashes"])
            node(0, 112, 135, 40, ["Extraction, alternatives", "and source review"])
            node(0, 52, 135, 37, ["Typed norms, conflicts", "and argumentation"])
            node(168, 176, 134, 35, ["Independent facts", "Observation evidence"])
            node(335, 176, 135, 35, ["Candidate response", "Mapping alternatives"])
            node(335, 112, 135, 40, ["Response-mapping", "review"])
            node(168, 74, 134, 45, ["Finite product analysis", "Verification / questions"])
            node(335, 52, 135, 37, ["Bounded model", "critique and revision"])
            node(
                0,
                0,
                470,
                29,
                ["Authenticated reviews, signed annotations, runtime claims, and audit evidence"],
                colors.HexColor("#e5eff1"),
            )
            arrow(67, 176, 67, 152)
            arrow(67, 112, 67, 89)
            arrow(135, 70, 168, 87)
            arrow(235, 176, 235, 119)
            arrow(402, 176, 402, 152)
            arrow(335, 130, 300, 113)
            arrow(302, 86, 335, 70)
            arrow(402, 89, 402, 112)
            arrow(235, 74, 235, 29)
        else:
            node(0, 137, 100, 39, ["Queued", "Runtime bound"])
            node(123, 137, 100, 39, ["Claim committed", "No automatic retry"])
            node(246, 137, 100, 39, ["Provider dispatch", "One initiation"])
            node(369, 137, 100, 39, ["Draft stored", "Review required"])
            node(100, 64, 140, 36, ["Runtime/review failure", "Zero provider calls"])
            node(255, 64, 110, 36, ["Crash / unknown", "Investigate"])
            node(100, 3, 140, 36, ["Admin interruption", "Late results retained"])
            node(369, 3, 100, 36, ["Reviewed verdict", "Stop or next job"])
            arrow(100, 157, 123, 157)
            arrow(223, 157, 246, 157)
            arrow(346, 157, 369, 157)
            arrow(173, 137, 173, 100)
            arrow(296, 137, 310, 100)
            arrow(310, 64, 240, 24)
            arrow(419, 137, 419, 39)
            c.setFont("Sans", 7.3)
            c.setFillColor(GRAY)
            c.drawString(
                2,
                193,
                "A reviewed violation may create a separate revision job; a claim is never replayed automatically.",
            )
        c.restoreState()


def render(source, output, font_dir):
    fonts(font_dir)
    body = ParagraphStyle(
        "body",
        fontName="Serif",
        fontSize=9.45,
        leading=13.4,
        alignment=TA_JUSTIFY,
        spaceAfter=6.5,
        textColor=INK,
        splitLongWords=True,
    )
    styles = {
        "body": body,
        "title": ParagraphStyle(
            "title", parent=body, fontName="SansBold", fontSize=22.5, leading=27, alignment=TA_CENTER, spaceAfter=8
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=body, fontName="Serif", fontSize=12.0, leading=16.5, alignment=TA_CENTER, spaceAfter=12
        ),
        "author": ParagraphStyle(
            "author", parent=body, fontName="Sans", fontSize=9.2, leading=14, alignment=TA_CENTER, spaceAfter=6
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=body,
            fontName="SansBold",
            fontSize=12.2,
            leading=16,
            spaceBefore=13,
            spaceAfter=7,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=body,
            fontName="SansBold",
            fontSize=10.2,
            leading=14,
            spaceBefore=9,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "abstract": ParagraphStyle("abstract", parent=body, fontSize=9.2, leading=13, leftIndent=12, rightIndent=12),
        "caption": ParagraphStyle("caption", parent=body, fontName="Sans", fontSize=8.2, leading=11.5, spaceAfter=9),
        "equation": ParagraphStyle(
            "equation",
            parent=body,
            fontName="Sans",
            fontSize=9.1,
            leading=13.5,
            alignment=TA_CENTER,
            leftIndent=10,
            rightIndent=10,
            spaceBefore=3,
            spaceAfter=10,
        ),
        "table": ParagraphStyle(
            "table", parent=body, fontName="Sans", fontSize=7.7, leading=10.5, alignment=0, spaceAfter=0
        ),
        "tablehead": ParagraphStyle(
            "tablehead",
            parent=body,
            fontName="SansBold",
            fontSize=7.8,
            leading=10.8,
            alignment=0,
            textColor=colors.white,
            spaceAfter=0,
        ),
        "ref": ParagraphStyle(
            "ref", parent=body, fontSize=8.2, leading=10.7, alignment=0, spaceAfter=4, splitLongWords=True
        ),
    }
    text = source.read_text(encoding="utf-8")
    metadata = re.search(r"Research software manuscript \| ([^\n|]+) \| Version ([^\n]+)", text)
    if metadata is None:
        raise ValueError("Manuscript must identify its date and software version")
    footer = f"Dimos | Version {metadata[2].strip()} | {metadata[1].strip()}"
    if "@CODE_COMMIT@" in text:
        raise ValueError("Unresolved code commit")
    for symbol in ("\u2011", "\u2013", "\u2014"):
        text = text.replace(symbol, "-")
    blocks = re.split(r"\n\s*\n", text.strip())
    story = []
    abstract = False
    refs = False
    intro = False
    i = 0
    while i < len(blocks):
        block = blocks[i].strip()
        i += 1
        if not block:
            continue
        if block.startswith("# "):
            story.append(Paragraph(markup(block[2:]), styles["title"]))
            continue
        if block.startswith("## ") and not intro and "Abstract" not in block:
            story.append(Paragraph(markup(block[3:]), styles["subtitle"]))
            intro = True
            continue
        if block.startswith("## "):
            heading = block[3:]
            abstract = heading == "Abstract"
            refs = heading == "References"
            story.append(Paragraph(markup(heading), styles["h1"]))
            continue
        if block.startswith("### "):
            story.append(Paragraph(markup(block[4:]), styles["h2"]))
            continue
        if block.startswith("[[ARCHITECTURE]]") or block.startswith("[[STATE_MACHINE]]"):
            diagram = Diagram("architecture" if "ARCHITECTURE" in block else "state")
            group = [Spacer(1, 6), diagram]
            if i < len(blocks) and blocks[i].startswith("**Figure"):
                group.append(Paragraph(markup(blocks[i]), styles["caption"]))
                i += 1
            story.append(KeepTogether(group))
            continue
        if block.startswith("|"):
            lines = [line for line in block.splitlines() if line.strip().startswith("|")]
            rows = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines]
            rows = [row for row in rows if not all(re.fullmatch(r"[:\- ]+", cell) for cell in row)]
            n = len(rows[0])
            fractions = [0.20, 0.42, 0.38] if n == 3 else [0.25, 0.20, 0.21, 0.34]
            data = [
                [Paragraph(markup(cell), styles["tablehead" if j == 0 else "table"]) for cell in row]
                for j, row in enumerate(rows)
            ]
            table = Table(data, colWidths=[WIDTH * f for f in fractions], repeatRows=1, hAlign="LEFT",
                          rowSplitRange=(3, -1) if len(rows) > 4 else None)
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), INK),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 7),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                        ("TOPPADDING", (0, 0), (-1, -1), 7),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [PALE, colors.white]),
                        ("LINEBELOW", (0, 0), (-1, 0), 0.7, INK),
                        ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.HexColor("#bacbd4")),
                    ]
                )
            )
            story.extend([Spacer(1, 5), table, Spacer(1, 10)])
            continue
        if block.startswith("EQUATION:"):
            equation = markup(block[9:].strip())
            for tag in ("sub", "super"):
                equation = equation.replace("&lt;" + tag + "&gt;", "<" + tag + ">").replace(
                    "&lt;/" + tag + "&gt;", "</" + tag + ">"
                )
            story.append(Paragraph(equation, styles["equation"]))
            continue
        if block.startswith("**Equation "):
            story.append(Paragraph(markup(block), styles["caption"]))
            story[-1].keepWithNext = True
            continue
        chosen = (
            "ref"
            if refs
            else "abstract"
            if abstract
            else "author"
            if not any(isinstance(f, Paragraph) and f.style.name == "h1" for f in story)
            else "body"
        )
        story.append(Paragraph(markup(" ".join(block.splitlines())), styles[chosen]))

    def page(c, doc):
        c.saveState()
        c.setStrokeColor(colors.HexColor("#c7d4dc"))
        c.setLineWidth(0.4)
        c.line(22 * mm, PAGE_H - 16.5 * mm, PAGE_W - 22 * mm, PAGE_H - 16.5 * mm)
        c.setFont("Sans", 7.1)
        c.setFillColor(GRAY)
        c.drawString(22 * mm, PAGE_H - 13.3 * mm, "LEGAL NORM ARGUMENTATION | Research software manuscript")
        c.drawRightString(PAGE_W - 22 * mm, 13 * mm, str(doc.page))
        c.drawString(22 * mm, 13 * mm, footer)
        c.restoreState()

    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=22 * mm,
        leftMargin=22 * mm,
        topMargin=22 * mm,
        bottomMargin=22 * mm,
        title="Evidence-Bound Legal Norm Argumentation",
        author="Joseph Dimos",
        subject="Legal autoformalization and governed normative AI",
    )
    doc.build(story, onFirstPage=page, onLaterPages=page)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("manuscript.md"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font-dir", type=Path, default=Path("/usr/share/fonts/truetype/dejavu"))
    args = parser.parse_args()
    render(args.input, args.output, args.font_dir)
