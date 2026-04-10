"""
Parse local DHL PDF documents for the CrossBorder Copilot RAG pipeline.

The parser returns one structured record per extracted PDF page. Each record
contains clean text plus metadata that downstream chunking and citation code can
preserve in ChromaDB.

Usage:
    python parse_local.py
    python parse_local.py --pdf-dir pdf_docs --output raw_documents.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader


PDF_DIR = Path("pdf_docs")
OUTPUT_FILE = Path("raw_documents.json")
MIN_PAGE_CHARS = 40

log = logging.getLogger(__name__)
logging.getLogger("pypdf").setLevel(logging.CRITICAL)


@dataclass(frozen=True)
class ParsedDocument:
    """A single parsed page from a local source document."""

    document_id: str
    source_type: str
    source_filename: str
    source_path: str
    source_uri: str
    title: str
    category: str
    page_number: int
    total_pages: int
    raw_text: str
    text: str


def infer_category(path: Path) -> str:
    """Infer a broad support topic from the filename and folder names."""
    value = " ".join(part.lower() for part in (*path.parts, path.stem))

    categories = [
        ("customs", ["customs", "clearance", "declaration", "invoice", "duty", "tax", "cbsa"]),
        ("restricted_goods", ["prohibit", "restricted", "restriction", "dangerous", "battery", "lithium", "knives"]),
        ("surcharges", ["surcharge", "fee", "price", "pricing", "rate", "pricelist", "fuel", "toll"]),
        ("terms_conditions", ["terms", "conditions", "liability", "carriage"]),
        ("returns_exceptions", ["return", "undeliverable", "exception", "delay", "missed"]),
        ("shipping_services", ["parcel", "packet", "direct", "standard", "letterpack", "service"]),
        ("packaging", ["packaging", "protecting goods", "shipment protection"]),
        ("integration", ["integration", "api", "configuration", "how-to", "toolkit"]),
        ("market_research", ["trend", "barometer", "whitepaper", "e-commerce guide"]),
    ]
    for category, needles in categories:
        if any(needle in value for needle in needles):
            return category
    return "general"


def make_document_id(path: Path) -> str:
    """Build a stable document id from the local source path."""
    return hashlib.sha1(path.as_posix().encode("utf-8")).hexdigest()[:16]


def normalize_line(line: str) -> str:
    line = unicodedata.normalize("NFKC", line)
    line = re.sub(r"\s+", " ", line).strip()
    return line


def looks_like_page_number(line: str) -> bool:
    """Identify common standalone page number/footer artifacts."""
    normalized = normalize_line(line)
    if not normalized:
        return True
    patterns = [
        r"^\d{1,4}$",
        r"^-\s*\d{1,4}\s*-$",
        r"^page\s+\d{1,4}(\s+of\s+\d{1,4})?$",
        r"^\d{1,4}\s*/\s*\d{1,4}$",
    ]
    return any(re.match(pattern, normalized, flags=re.IGNORECASE) for pattern in patterns)


def line_signature(line: str) -> str:
    """Normalize a line enough to compare repeated headers/footers."""
    line = normalize_line(line).lower()
    line = re.sub(r"\d+", "#", line)
    return line


def repeated_header_footer_lines(raw_pages: list[str], scan_lines: int = 3) -> set[str]:
    """
    Detect short lines that repeat at the top or bottom of many pages.

    PDF extraction often includes repeating DHL headers, legal footers, or page
    labels. We only remove short repeated candidates from page edges to avoid
    deleting meaningful body text.
    """
    if len(raw_pages) < 4:
        return set()

    candidates: Counter[str] = Counter()
    for page_text in raw_pages:
        lines = [normalize_line(line) for line in page_text.splitlines() if normalize_line(line)]
        edge_lines = lines[:scan_lines] + lines[-scan_lines:]
        for line in edge_lines:
            signature = line_signature(line)
            if 3 <= len(signature) <= 140 and not looks_like_page_number(signature):
                candidates[signature] += 1

    threshold = max(3, int(len(raw_pages) * 0.35))
    return {signature for signature, count in candidates.items() if count >= threshold}


def join_broken_lines(lines: Iterable[str]) -> str:
    """
    Join PDF line breaks where doing so is unlikely to damage structure.

    Bullets, headings, table-like lines, and sentence-ending punctuation keep a
    newline. Sentence continuations become spaces.
    """
    output: list[str] = []
    bullet_re = re.compile(r"^([*•\-–]|\(?[a-zA-Z0-9]{1,3}[\).])\s+")

    for raw_line in lines:
        line = normalize_line(raw_line)
        if not line:
            if output and output[-1] != "":
                output.append("")
            continue
        if looks_like_page_number(line):
            continue

        if not output or output[-1] == "":
            output.append(line)
            continue

        previous = output[-1]
        previous_ends_sentence = bool(re.search(r"[.!?:;)]$", previous))
        current_starts_structure = bool(bullet_re.match(line))
        previous_looks_heading = previous.isupper() and len(previous) < 90
        current_looks_heading = line.isupper() and len(line) < 90
        table_like = previous.count("  ") > 1 or line.count("  ") > 1

        if (
            previous_ends_sentence
            or current_starts_structure
            or previous_looks_heading
            or current_looks_heading
            or table_like
        ):
            output.append(line)
        else:
            output[-1] = f"{previous} {line}"

    text = "\n".join(output)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_page_text(raw_text: str, repeated_lines: set[str] | None = None) -> str:
    """Clean one extracted PDF page while preserving useful structure."""
    repeated_lines = repeated_lines or set()
    normalized = unicodedata.normalize("NFKC", raw_text or "")
    normalized = normalized.replace("\x00", " ").replace("\r", "\n")

    lines = []
    for line in normalized.splitlines():
        clean = normalize_line(line)
        if not clean:
            lines.append("")
            continue
        if line_signature(clean) in repeated_lines:
            continue
        lines.append(clean)

    text = join_broken_lines(lines)
    text = re.sub(r"\bPage\s+\d{1,4}\s+of\s+\d{1,4}\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def best_effort_title(path: Path, reader: PdfReader, raw_pages: list[str]) -> str:
    """Extract a usable title from PDF metadata, first-page text, or filename."""
    metadata = reader.metadata or {}
    metadata_title = metadata.get("/Title") or metadata.get("title")
    if metadata_title and str(metadata_title).strip():
        title = normalize_line(str(metadata_title))
        if len(title) >= 4 and title.lower() not in {"untitled", "microsoft word"}:
            return title[:180]

    if raw_pages:
        for line in raw_pages[0].splitlines()[:12]:
            candidate = normalize_line(line)
            if 8 <= len(candidate) <= 160 and not looks_like_page_number(candidate):
                return candidate

    return path.stem.replace("-", " ").replace("_", " ").title()


def parse_pdf_file(path: Path, min_page_chars: int = MIN_PAGE_CHARS) -> list[dict]:
    """Parse a PDF file into one structured dictionary per non-empty page."""
    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        log.warning("Could not open PDF %s: %s", path.name, exc)
        return []

    raw_pages: list[str] = []
    for page in reader.pages:
        try:
            raw_pages.append(page.extract_text() or "")
        except Exception as exc:
            log.debug("Could not extract one page from %s: %s", path.name, exc)
            raw_pages.append("")

    repeated_lines = repeated_header_footer_lines(raw_pages)
    title = best_effort_title(path, reader, raw_pages)
    document_id = make_document_id(path)
    total_pages = len(raw_pages)
    category = infer_category(path)

    parsed_pages: list[dict] = []
    for index, raw_text in enumerate(raw_pages, start=1):
        text = clean_page_text(raw_text, repeated_lines=repeated_lines)
        if len(text) < min_page_chars:
            continue
        parsed_pages.append(
            asdict(
                ParsedDocument(
                    document_id=document_id,
                    source_type="pdf",
                    source_filename=path.name,
                    source_path=path.as_posix(),
                    source_uri=f"file://{path.as_posix()}",
                    title=title,
                    category=category,
                    page_number=index,
                    total_pages=total_pages,
                    raw_text=raw_text.strip(),
                    text=text,
                )
            )
        )

    if not parsed_pages:
        log.warning("PDF %s yielded no extractable text; it may be scanned or malformed", path.name)
    return parsed_pages


def parse_local_documents(pdf_dir: Path | str = PDF_DIR, min_page_chars: int = MIN_PAGE_CHARS) -> list[dict]:
    """Parse all PDFs from a local directory."""
    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_path}")

    pdf_files = sorted(pdf_path.rglob("*.pdf")) + sorted(pdf_path.rglob("*.PDF"))
    documents: list[dict] = []

    log.info("Found %d PDF files in %s", len(pdf_files), pdf_path)
    for file_path in pdf_files:
        pages = parse_pdf_file(file_path, min_page_chars=min_page_chars)
        documents.extend(pages)
        log.info(
            "Parsed %-55s %4d/%d pages",
            file_path.name[:55],
            len(pages),
            pages[0]["total_pages"] if pages else 0,
        )

    return documents


def write_json(documents: list[dict], output_path: Path | str) -> None:
    """Write parsed page records as UTF-8 JSON."""
    Path(output_path).write_text(json.dumps(documents, ensure_ascii=False, indent=2), encoding="utf-8")


def print_summary(documents: list[dict], output_path: Path | None = None) -> None:
    files = {doc["source_filename"] for doc in documents}
    categories = Counter(doc["category"] for doc in documents)
    print("\nParsed local PDF documents")
    print(f"  Files processed : {len(files)}")
    print(f"  Parsed pages    : {len(documents)}")
    print("  Categories      :")
    for category, count in categories.most_common():
        print(f"    {category:<18} {count}")
    if output_path:
        print(f"  Output          : {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse local PDF files for the RAG knowledge base.")
    parser.add_argument("--pdf-dir", default=str(PDF_DIR), help="Directory containing local PDF files")
    parser.add_argument("--output", default=str(OUTPUT_FILE), help="Where to write parsed JSON")
    parser.add_argument("--min-page-chars", type=int, default=MIN_PAGE_CHARS)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s  %(message)s")

    documents = parse_local_documents(args.pdf_dir, min_page_chars=args.min_page_chars)
    if not documents:
        log.error("No PDF text parsed from %s", args.pdf_dir)
        return

    output_path = Path(args.output)
    write_json(documents, output_path)
    print_summary(documents, output_path)


if __name__ == "__main__":
    main()
