"""scripts/build_report.py - render the assignment writeup to HTML (+ PDF if available).

Primary deliverable is the HTML file. The user can open it in Chrome / Edge and
"Print > Save as PDF" for a clean PDF without any system dependencies.

If `weasyprint` is installed AND its GTK native dependencies are present, a PDF
is also produced automatically.
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


def render_html() -> int:
    if not REPORT_MD.exists():
        print(f"missing source: {REPORT_MD}")
        return 1
    try:
        import markdown as md_lib
    except ImportError:
        print("`markdown` package not installed. Run: pip install markdown")
        return 1

    text = REPORT_MD.read_text(encoding="utf-8")
    html_content = md_lib.markdown(text, extensions=["tables", "fenced_code", "toc", "attr_list"])
    title_match = re.search(r"^#\s+(.+)$", text, re.M)
    title = title_match.group(1) if title_match else "Glance Internship Report"
    rendered = Template(TEMPLATE).render(title=title, content=html_content)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_HTML.write_text(rendered, encoding="utf-8")
    print(f"wrote {REPORT_HTML}")

    pdf_written = False
    try:
        from weasyprint import HTML
        HTML(string=rendered, base_url=str(REPORT_DIR.resolve())).write_pdf(REPORT_PDF)
        print(f"wrote {REPORT_PDF}")
        pdf_written = True
    except ImportError:
        print("weasyprint not installed (skipped PDF)")
    except Exception as exc:
        print(f"weasyprint PDF render failed ({exc.__class__.__name__}); HTML only.")
        print("  hint: open the HTML file in Chrome / Edge and 'Print > Save as PDF'.")

    if not pdf_written:
        print()
        print("HTML is the primary deliverable. To convert to PDF:")
        print("  1. Open " + str(REPORT_HTML) + " in Chrome or Edge.")
        print("  2. Ctrl+P > Destination: 'Save as PDF' > Save.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-pdf", action="store_true", help="skip PDF rendering")
    args = parser.parse_args()
    if args.no_pdf:
        try:
            import weasyprint  # noqa: F401
            sys.modules["weasyprint"] = None  # type: ignore
        except ImportError:
            pass
    return render_html()


if __name__ == "__main__":
    sys.exit(main())