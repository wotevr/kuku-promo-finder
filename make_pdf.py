"""
Turns solution.md into solution.pdf (max 2 pages) with reportlab.

    python make_pdf.py                 # solution.md -> solution.pdf
    python make_pdf.py notes.md out.pdf

Supports the simple Markdown used in solution.md: # / ## / ### headings, paragraphs, - bullets,
1. numbered items, | tables |, **bold**, *italic*, `code` and [links](url).
If the PDF is longer than 2 pages it shrinks the font and tries again.
"""

import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

MAX_PAGES = 2


def inline(text):
    """Markdown inline -> reportlab mini-HTML."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<link href="\2" color="#1a56b0"><u>\1</u></link>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", text)
    text = re.sub(r"`([^`]+)`", r'<font face="Courier">\1</font>', text)
    return text


def styles(size):
    base = ParagraphStyle("body", fontName="Helvetica", fontSize=size, leading=size * 1.28, alignment=TA_LEFT,
                          spaceAfter=size * 0.35)
    return {
        "body": base,
        "small": ParagraphStyle("small", parent=base, fontSize=size - 0.6, leading=(size - 0.6) * 1.22, spaceAfter=0),
        "h1": ParagraphStyle("h1", parent=base, fontName="Helvetica-Bold", fontSize=size + 5, leading=size + 7,
                             spaceAfter=size * 0.3),
        "h2": ParagraphStyle("h2", parent=base, fontName="Helvetica-Bold", fontSize=size + 1.6, leading=size + 3,
                             spaceBefore=size * 0.55, spaceAfter=size * 0.25, textColor=colors.HexColor("#1b3a6b")),
        "h3": ParagraphStyle("h3", parent=base, fontName="Helvetica-Bold", fontSize=size + 0.4,
                             spaceBefore=size * 0.3, spaceAfter=size * 0.15),
    }


def build(md_text, out_path, size):
    st = styles(size)
    story, lines, i = [], md_text.splitlines(), 0
    width = A4[0] - 3.0 * cm

    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
            continue
        if line.startswith("|"):  # table
            raw = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    raw.append(cells)
                i += 1
            # column width ~ length of its longest cell (capped), so short columns stay narrow
            n = max(len(r) for r in raw)
            need = [min(70, max(6, max(len(r[c]) if c < len(r) else 0 for r in raw))) for c in range(n)]
            col_w = [width * x / sum(need) for x in need]
            rows = [[Paragraph(inline(c), st["small"]) for c in r] + [""] * (n - len(r)) for r in raw]
            t = Table(rows, colWidths=col_w, hAlign="LEFT", repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8edf5")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c2d1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]))
            story += [t, Spacer(1, size * 0.4)]
            continue
        if line.startswith("### "):
            story.append(Paragraph(inline(line[4:]), st["h3"]))
        elif line.startswith("## "):
            story.append(Paragraph(inline(line[3:]), st["h2"]))
        elif line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), st["h1"]))
        elif re.match(r"^(\s*)([-*]|\d+\.) ", line):
            items, numbered = [], bool(re.match(r"^\s*\d+\. ", line))
            while i < len(lines) and re.match(r"^(\s*)([-*]|\d+\.) ", lines[i]):
                text = re.sub(r"^(\s*)([-*]|\d+\.) ", "", lines[i])
                items.append(ListItem(Paragraph(inline(text), st["body"]), leftIndent=10))
                i += 1
            story.append(ListFlowable(items, bulletType="1" if numbered else "bullet", start=1 if numbered else None,
                                      leftIndent=11, bulletFontSize=size - 1, bulletOffsetY=0))
            continue
        else:  # paragraph: join following plain lines
            para = [line]
            while i + 1 < len(lines) and lines[i + 1].strip() and not re.match(r"^(#|\||\s*[-*] |\s*\d+\. )", lines[i + 1]):
                i += 1
                para.append(lines[i].strip())
            story.append(Paragraph(inline(" ".join(para)), st["body"]))
        i += 1

    pages = {"n": 0}

    def count(canvas, doc):
        pages["n"] = doc.page

    doc = SimpleDocTemplate(str(out_path), pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm,
                            topMargin=1.2 * cm, bottomMargin=1.2 * cm, title="Kuku promo finder - solution")
    doc.build(story, onFirstPage=count, onLaterPages=count)
    return pages["n"]


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "solution.md")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else src.with_suffix(".pdf"))
    md = src.read_text(encoding="utf-8")
    for size in (9.5, 9.0, 8.6, 8.2, 7.8):
        n = build(md, out, size)
        print(f"font {size}pt -> {n} page(s)")
        if n <= MAX_PAGES:
            print(f"OK: {out} is {n} page(s)")
            return
    print(f"WARNING: still {n} pages at the smallest font. Shorten solution.md.")


if __name__ == "__main__":
    main()
