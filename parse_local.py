"""
parse_local.py  —  CrossBorder Copilot Phase 1
Reads local HTML and PDF files and produces raw_documents.json.

Folder layout expected:
    html_pages/   ← DHL help-center pages saved from browser (Cmd+S)
    pdf_docs/     ← supplementary PDF documents (CBSA, Canada Post, etc.)

Usage:
    python parse_local.py
    python parse_local.py --html-dir my_html --pdf-dir my_pdfs
"""

import json
import logging
import argparse
from pathlib import Path

from bs4 import BeautifulSoup

# pypdf for PDF text extraction (pip install pypdf)
try:
    from pypdf import PdfReader
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("WARNING: pypdf not installed — PDF files will be skipped.")
    print("         Run: pip install pypdf")

# ── Config ────────────────────────────────────────────────────────────────────

HTML_DIR    = Path("html_pages")
PDF_DIR     = Path("pdf_docs")
OUTPUT_FILE = Path("raw_documents.json")

# ── Helpers ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def infer_doc_type(filename: str) -> str:
    f = filename.lower()
    if "faq" in f:                          return "faq"
    if "customs" in f or "clearance" in f:  return "customs"
    if "prohibit" in f or "dangerous" in f \
            or "restricted" in f:           return "policy"
    if "surcharge" in f or "fee" in f:      return "surcharge"
    if "integrat" in f or "api" in f:       return "integration"
    if "undeliver" in f or "return" in f:   return "policy"
    return "guide"


# ── HTML parser ───────────────────────────────────────────────────────────────

def parse_html_file(path: Path) -> dict | None:
    try:
        soup = BeautifulSoup(
            path.read_text(encoding="utf-8", errors="ignore"),
            "html.parser"
        )
    except Exception as exc:
        log.warning("Could not parse %s: %s", path.name, exc)
        return None

    # Strip noise
    for tag in soup.find_all(
        ["nav", "footer", "header", "script", "style", "noscript", "aside", "form"]
    ):
        tag.decompose()

    page_title = soup.title.get_text(strip=True) if soup.title else path.stem

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find(id="main-content")
        or soup.body
    )
    if not main:
        return None

    elements = main.find_all(["h1", "h2", "h3", "h4", "p", "li", "dt", "dd"])
    lines = []
    for el in elements:
        text = el.get_text(separator=" ", strip=True)
        if len(text) < 20:
            continue
        if el.name in ("h1", "h2", "h3", "h4"):
            lines.append(f"\n## {text}\n")
        else:
            lines.append(text)

    full_text = "\n".join(lines).strip()
    if len(full_text) < 100:
        return None

    h2 = main.find("h2")
    section = h2.get_text(strip=True)[:80] if h2 else ""

    return {
        "url":        f"https://www.dhl.com/{path.stem}",
        "page_title": page_title,
        "section":    section,
        "doc_type":   infer_doc_type(path.stem),
        "source":     "html",
        "filename":   path.name,
        "text":       full_text,
    }


# ── PDF parser ────────────────────────────────────────────────────────────────

def parse_pdf_file(path: Path) -> dict | None:
    if not PDF_SUPPORT:
        return None

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        log.warning("Could not open PDF %s: %s", path.name, exc)
        return None

    # Extract text page by page, keep page breaks as section separators
    pages_text = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
            text = text.strip()
            if len(text) > 50:
                pages_text.append(f"\n## Page {i + 1}\n\n{text}")
        except Exception:
            continue

    full_text = "\n".join(pages_text).strip()
    if len(full_text) < 100:
        log.warning("PDF %s yielded very little text (scanned?), skipping", path.name)
        return None

    # Try to get title from PDF metadata
    meta = reader.metadata or {}
    page_title = (
        meta.get("/Title")
        or meta.get("title")
        or path.stem.replace("-", " ").replace("_", " ").title()
    )

    return {
        "url":        f"file://pdf_docs/{path.name}",
        "page_title": str(page_title),
        "section":    "",
        "doc_type":   infer_doc_type(path.stem),
        "source":     "pdf",
        "filename":   path.name,
        "text":       full_text,
        "num_pages":  len(reader.pages),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--html-dir", default=str(HTML_DIR))
    parser.add_argument("--pdf-dir",  default=str(PDF_DIR))
    parser.add_argument("--output",   default=str(OUTPUT_FILE))
    args = parser.parse_args()

    html_dir = Path(args.html_dir)
    pdf_dir  = Path(args.pdf_dir)
    out_path = Path(args.output)

    documents = []

    # HTML pages are excluded — only PDF documents count as knowledge base sources
    # ── Process HTML files ────────────────────────────────────────────────────
    if html_dir.exists():
        html_files = list(html_dir.glob("*.html")) + list(html_dir.glob("*.htm"))
        log.info("Found %d HTML files in %s", len(html_files), html_dir)
        for path in sorted(html_files):
            doc = parse_html_file(path)
            if doc:
                documents.append(doc)
                log.info("  ✓ HTML  %-45s  %d chars", path.name[:45], len(doc["text"]))
            else:
                log.info("  ✗ HTML  %-45s  skipped (no content)", path.name[:45])
    else:
        log.warning("HTML directory not found: %s", html_dir)

    # ── Process PDF files ─────────────────────────────────────────────────────
    if pdf_dir.exists():
        pdf_files = list(pdf_dir.glob("*.pdf")) + list(pdf_dir.glob("*.PDF"))
        log.info("Found %d PDF files in %s", len(pdf_files), pdf_dir)
        for path in sorted(pdf_files):
            doc = parse_pdf_file(path)
            if doc:
                documents.append(doc)
                log.info(
                    "  ✓ PDF   %-45s  %d chars  (%d pages)",
                    path.name[:45], len(doc["text"]), doc.get("num_pages", 0)
                )
            else:
                log.info("  ✗ PDF   %-45s  skipped", path.name[:45])
    else:
        log.info("PDF directory not found (%s) — skipping PDFs", pdf_dir)
        log.info("  Create it with: mkdir %s", pdf_dir)

    if not documents:
        log.error("No documents parsed. Check that html_pages/ or pdf_docs/ exist and contain files.")
        return

    out_path.write_text(
        json.dumps(documents, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # Summary
    from collections import Counter
    by_type   = Counter(d["doc_type"] for d in documents)
    by_source = Counter(d["source"]   for d in documents)
    print(f"\n{'─'*50}")
    print(f"Total documents parsed : {len(documents)}")
    print(f"By source : {dict(by_source)}")
    print(f"By doc_type : {dict(by_type)}")
    print(f"Output → {out_path}")
    print(f"{'─'*50}\n")
    print("Next step:  python ingest.py --reset")


if __name__ == "__main__":
    main()
