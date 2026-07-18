"""scripts/build_report.py - render the assignment writeup to HTML and PDF.

Primary deliverable is the HTML file; the user can also open it in Chrome / Edge
and "Print > Save as PDF" for the same content via the browser pipeline.

If weasyprint is installed AND its GTK native dependencies are present, a PDF
is produced via CSS-rendered HTML. Otherwise we fall back to a dependency-free
PDF produced directly by fpdf2 (no GTK / system libs needed).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jinja2 import Template

REPORT_DIR = Path("report")
REPORT_MD = REPORT_DIR / "Glance_Internship_Report.md"
REPORT_HTML = REPORT_DIR / "Glance_Internship_Report.html"
REPORT_PDF = REPORT_DIR / "Glance_Internship_Report.pdf"

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{{ title }}</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 920px; margin: 2em auto; padding: 0 1.5em; line-height: 1.65; color: #1a1a1a; }
    h1 { font-size: 2em; border-bottom: 2px solid #111; padding-bottom: 0.3em; margin-top: 0.5em; }
    h2 { font-size: 1.5em; border-bottom: 1px solid #ccc; padding-bottom: 0.2em; margin-top: 2.2em; }
    h3 { font-size: 1.2em; margin-top: 1.8em; color: #333; }
    code { background: #f4f4f4; padding: 0.1em 0.35em; border-radius: 3px; font-size: 0.92em; font-family: "JetBrains Mono", Menlo, monospace; }
    pre { background: #f6f6f6; padding: 1em; border-radius: 5px; overflow-x: auto; font-size: 0.88em; }
    table { border-collapse: collapse; width: 100%; margin: 1.2em 0; font-size: 0.95em; }
    th, td { border: 1px solid #ddd; padding: 0.55em 0.8em; text-align: left; }
    th { background: #f4f4f4; font-weight: 600; }
    tr:nth-child(even) td { background: #fafafa; }
    blockquote { border-left: 4px solid #ccc; padding: 0.2em 1em; color: #555; background: #fafafa; margin: 1em 0; }
    img { max-width: 100%; height: auto; border-radius: 4px; margin: 0.5em 0; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
    hr { border: none; border-top: 1px solid #eee; margin: 2em 0; }
    .meta { color: #666; font-size: 0.92em; margin-bottom: 2em; }
    .meta strong { color: #333; }
    @media print { body { max-width: 100%; margin: 0; padding: 1em; } img { page-break-inside: avoid; } }
  </style>
</head>
<body>
{{ content }}
</body>
</html>"""


UNICODE_SUBS = {
    "\u2013": "-", "\u2014": "-",
    "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"',
    "\u2026": "...", "\u00a0": " ", "\u00b7": "*",
    "\u2190": "<-", "\u2192": "->",
    "\u2191": "^", "\u2193": "v",
    "\u21d0": "<=", "\u21d2": "=>",
    "\u2502": "|", "\u2500": "-",
    "\u251c": "|--", "\u2514": "`--",
    "\u2510": "--+", "\u250c": "+--",
    "\u2524": "--|", "\u253c": "|--|",
    "\u00d7": "x", "\u00f7": "/",
    "\u00b1": "+/-", "\u2248": "~=",
    "\u2260": "!=", "\u2264": "<=", "\u2265": ">=",
    "\u00b0": "deg",
    "\u00a9": "(c)", "\u00ae": "(R)", "\u2122": "TM",
    "\u2728": "*", "\u2705": "[x]", "\u274c": "[ ]",
}


def normalize(text: str) -> str:
    for k, v in UNICODE_SUBS.items():
        text = text.replace(k, v)
    return text.encode("ascii", "replace").decode("ascii")


def strip_inline_md(text: str) -> str:
    return (
        text.replace("**", "")
        .replace("__", "")
        .replace("`", "")
        .replace("_", "")
    )


def render_with_weasyprint(html_text: str) -> bool:
    try:
        from weasyprint import HTML
    except ImportError:
        return False
    except Exception as exc:
        print(f"weasyprint import failed ({exc.__class__.__name__}): {exc}; falling back to fpdf2.")
        return False
    try:
        HTML(string=html_text, base_url=str(REPORT_DIR.resolve())).write_pdf(REPORT_PDF)
        return True
    except Exception as exc:
        print(f"weasyprint PDF render failed ({exc.__class__.__name__}); falling back to fpdf2.")
        return False


def render_with_fpdf2(md_text: str) -> bool:
    try:
        from fpdf import FPDF
    except ImportError:
        return False

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(left=18, top=18, right=18)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    available_w = pdf.w - pdf.l_margin - pdf.r_margin

    pdf.set_font("helvetica", "B", 14)
    pdf.multi_cell(available_w, 9, normalize("Glance ML Internship Assignment"))
    pdf.ln(1)
    pdf.set_font("helvetica", "B", 12)
    pdf.multi_cell(available_w, 7, normalize("Multimodal Fashion & Context Retrieval"))
    pdf.ln(2)
    pdf.set_draw_color(180, 180, 180)
    pdf.line(18, pdf.get_y(), pdf.w - 18, pdf.get_y())
    pdf.ln(4)
    pdf.set_draw_color(0, 0, 0)

    in_code_block = False
    code_buffer: list[str] = []

    def flush_code() -> None:
        nonlocal code_buffer
        if not code_buffer:
            return
        block = "\n".join(code_buffer)
        pdf.set_font("courier", "", 8)
        pdf.set_fill_color(245, 245, 245)
        pdf.multi_cell(available_w, 4, normalize(block), fill=True)
        pdf.set_font("helvetica", "", 10)
        pdf.ln(1)
        code_buffer = []

    def render_table(table_lines: list[str]) -> None:
        """Render a markdown table as a monospace pre-formatted block."""
        available_w = pdf.w - pdf.l_margin - pdf.r_margin
        pdf.set_font("courier", "", 7)
        for ln in table_lines:
            text = normalize(ln)
            pdf.set_xy(pdf.l_margin, pdf.get_y())
            try:
                pdf.multi_cell(w=available_w, h=4, text=text)
            except Exception as exc:
                print(f"!! table render failed: {type(exc).__name__}: {exc}")
                print(f"   available_w={available_w}  line_len={len(text)}  line={text[:80]!r}")
                raise
        pdf.set_font("helvetica", "", 10)
        pdf.ln(2)

    for raw_line in md_text.splitlines():
        line = raw_line.rstrip()

        if line.startswith("```"):
            if in_code_block:
                flush_code()
                in_code_block = False
            else:
                in_code_block = True
            continue
        if in_code_block:
            code_buffer.append(line)
            continue

        if not line.strip():
            pdf.ln(3)
            continue

        if line.startswith("# "):
            pdf.set_font("helvetica", "B", 14)
            pdf.multi_cell(available_w, 9, normalize(strip_inline_md(line[2:].strip())))
            pdf.ln(3)
        elif line.startswith("## "):
            pdf.set_font("helvetica", "B", 12)
            pdf.multi_cell(available_w, 7, normalize(strip_inline_md(line[3:].strip())))
            pdf.ln(2)
        elif line.startswith("### "):
            pdf.set_font("helvetica", "B", 11)
            pdf.multi_cell(available_w, 6, normalize(strip_inline_md(line[4:].strip())))
            pdf.ln(1)
        elif line.startswith("|") and line.strip().endswith("|") and "|" in line[1:]:
            buf = [line]
            for nxt in md_text.splitlines()[md_text.splitlines().index(raw_line) + 1:]:
                if not (nxt.startswith("|") and nxt.strip().endswith("|")):
                    break
                buf.append(nxt)
            if len(buf) >= 2 and set(buf[1].replace(" ", "").replace(":", "")) <= set("|-"):
                render_table(buf)
            else:
                pdf.set_font("helvetica", "", 10)
                for b in buf:
                    pdf.multi_cell(available_w, 6, normalize(strip_inline_md(b)))
            continue
        elif line.lstrip().startswith(("- ", "* ")):
            text = line.lstrip()[2:].strip()
            pdf.set_font("helvetica", "", 10)
            pdf.multi_cell(available_w, 5, "  -  " + normalize(strip_inline_md(text)))
        elif line.startswith("---"):
            pdf.ln(1)
            pdf.set_draw_color(180, 180, 180)
            pdf.line(18, pdf.get_y(), pdf.w - 18, pdf.get_y())
            pdf.ln(3)
            pdf.set_draw_color(0, 0, 0)
        elif line.startswith("![") and "](" in line and line.endswith(")"):
            m = re.match(r"!\[(.*?)\]\((.*?)\)", line)
            if m:
                alt, path = m.group(1), m.group(2)
                abs_path = (REPORT_DIR / path).resolve()
                if abs_path.exists():
                    pdf.image(str(abs_path), w=170)
                    pdf.ln(2)
                    if alt:
                        pdf.set_font("helvetica", "I", 9)
                        pdf.multi_cell(available_w, 5, normalize(alt))
                        pdf.set_font("helvetica", "", 10)
                else:
                    pdf.set_font("helvetica", "I", 10)
                    pdf.multi_cell(available_w, 6, f"[missing image: {path}]")
                    pdf.set_font("helvetica", "", 10)
        else:
            pdf.set_font("helvetica", "", 10)
            pdf.multi_cell(available_w, 5, normalize(strip_inline_md(line)))

    flush_code()
    try:
        pdf.output(str(REPORT_PDF))
    except Exception as exc:
        import traceback
        print(f"!! fpdf2 output() failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
    return True


def _render_loop_with_traceback(pdf, md_text, available_w):
    """Same logic as render_with_fpdf2 but raises immediately on error."""
    in_code_block = False
    code_buffer = []
    def flush_code():
        if not code_buffer:
            return
        block = "\n".join(code_buffer)
        pdf.set_font("courier", "", 8)
        pdf.set_fill_color(245, 245, 245)
        pdf.multi_cell(available_w, 4, normalize(block), fill=True)
        pdf.set_font("helvetica", "", 10)
        pdf.ln(1)
        code_buffer = []
    def render_table(table_lines):
        available_w_local = pdf.w - pdf.l_margin - pdf.r_margin
        pdf.set_font("courier", "", 7)
        for ln in table_lines:
            text = normalize(ln)
            pdf.set_xy(pdf.l_margin, pdf.get_y())
            pdf.multi_cell(w=available_w_local, h=4, text=text)
        pdf.set_font("helvetica", "", 10)
        pdf.ln(2)
    lines = md_text.splitlines()
    for i, raw_line in enumerate(lines):
        line = raw_line.rstrip()
        try:
            if line.startswith("```"):
                if in_code_block:
                    flush_code()
                    in_code_block = False
                else:
                    in_code_block = True
                continue
            if in_code_block:
                code_buffer.append(line)
                continue
            if not line.strip():
                pdf.ln(3)
                continue
            if line.startswith("# "):
                pdf.set_font("helvetica", "B", 14)
                pdf.multi_cell(available_w, 9, normalize(strip_inline_md(line[2:].strip())))
                pdf.ln(3)
            elif line.startswith("## "):
                pdf.set_font("helvetica", "B", 12)
                pdf.multi_cell(available_w, 7, normalize(strip_inline_md(line[3:].strip())))
                pdf.ln(2)
            elif line.startswith("### "):
                pdf.set_font("helvetica", "B", 11)
                pdf.multi_cell(available_w, 6, normalize(strip_inline_md(line[4:].strip())))
                pdf.ln(1)
            elif line.startswith("|") and line.strip().endswith("|") and "|" in line[1:]:
                buf = [line]
                for nxt in lines[i+1:]:
                    if not (nxt.startswith("|") and nxt.rstrip().endswith("|")):
                        break
                    buf.append(nxt.rstrip())
                if len(buf) >= 2 and set(buf[1].replace(" ", "").replace(":", "")) <= set("|-"):
                    render_table(buf)
                else:
                    pdf.set_font("helvetica", "", 10)
                    for b in buf:
                        pdf.multi_cell(available_w, 6, normalize(strip_inline_md(b)))
                continue
            elif line.lstrip().startswith(("- ", "* ")):
                pdf.set_font("helvetica", "", 10)
                text = line.lstrip()[2:].strip()
                pdf.multi_cell(available_w, 5, "  -  " + normalize(strip_inline_md(text)))
            elif line.startswith("---"):
                pdf.ln(1)
                pdf.set_draw_color(180, 180, 180)
                pdf.line(18, pdf.get_y(), pdf.w - 18, pdf.get_y())
                pdf.ln(3)
                pdf.set_draw_color(0, 0, 0)
            elif line.startswith("![") and "](" in line and line.endswith(")"):
                m = re.match(r"!\[(.*?)\]\((.*?)\)", line)
                if m:
                    alt, path = m.group(1), m.group(2)
                    abs_path = (REPORT_DIR / path).resolve()
                    if abs_path.exists():
                        pdf.image(str(abs_path), w=170)
                        pdf.ln(2)
                        if alt:
                            pdf.set_font("helvetica", "I", 9)
                            pdf.multi_cell(available_w, 5, normalize(alt))
                            pdf.set_font("helvetica", "", 10)
            else:
                pdf.set_font("helvetica", "", 10)
                pdf.multi_cell(available_w, 5, normalize(strip_inline_md(line)))
        except Exception as exc:
            print(f"\n!! Line {i+1}: {type(exc).__name__}: {exc}")
            print(f"   Line content: {line[:200]!r}")
            raise
    flush_code()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-pdf", action="store_true", help="skip PDF rendering")
    args = parser.parse_args()

    if not REPORT_MD.exists():
        print(f"missing source: {REPORT_MD}")
        return 1
    try:
        import markdown as md_lib
    except ImportError:
        print("`markdown` package not installed. Run: pip install markdown")
        return 1

    md_text = REPORT_MD.read_text(encoding="utf-8")
    html_content = md_lib.markdown(md_text, extensions=["tables", "fenced_code", "toc", "attr_list"])
    title_match = re.search(r"^#\s+(.+)$", md_text, re.M)
    title = title_match.group(1) if title_match else "Glance Internship Report"
    rendered = Template(TEMPLATE).render(title=title, content=html_content)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_HTML.write_text(rendered, encoding="utf-8")
    print(f"wrote {REPORT_HTML}")

    if args.no_pdf:
        return 0

    if render_with_weasyprint(rendered):
        print(f"wrote {REPORT_PDF}  (via weasyprint)")
    elif render_with_fpdf2(md_text):
        print(f"wrote {REPORT_PDF}  (via fpdf2 — no GTK / system deps)")
    else:
        print("PDF skipped: fpdf2 not installed (pip install fpdf2)")
    return 0


if __name__ == "__main__":
    sys.exit(main())