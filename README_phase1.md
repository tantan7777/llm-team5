# Phase 1 — Data Extraction & RAG Pipeline

## Overview

This phase builds the knowledge base for CrossBorder Copilot. It processes DHL shipping PDF documents, chunks the text, embeds it, and stores it in ChromaDB for retrieval by the Phase 2 agent.

---

## File Structure

```
project/
├── pdf_docs/            # DHL PDF documents (56 files)
├── chroma_db/           # ChromaDB vector store (auto-generated, do not edit)
├── raw_documents.json   # Parsed documents before chunking (auto-generated)
├── parse_local.py       # Step 1 — parse PDF files into raw_documents.json
├── ingest.py            # Step 2 — chunk documents and upsert into ChromaDB
├── retriever.py         # Step 3 — query ChromaDB (used by Phase 2 agent)
└── requirements.txt     # Python dependencies
```

---

## Setup

```bash
pip install -r requirements.txt
```

Dependencies: `requests`, `beautifulsoup4`, `lxml`, `chromadb`, `sentence-transformers`, `pypdf`

---

## How to Run (Step by Step)

### Step 1 — Add PDF documents

Download DHL PDF documents manually from the browser and place them in the `pdf_docs/` folder. Manual download is required because DHL's servers block all automated requests (scraping, curl, and archive crawlers all return errors).

### Step 2 — Parse documents

```bash
python parse_local.py # Generate raw_documents.json
```

Reads all PDFs in `pdf_docs/`, extracts clean text, and saves to `raw_documents.json`.

Output example:
```
Total documents parsed : 66
By source : {'pdf': 66}
By doc_type : {'guide': 55, 'customs': 6,  'policy': 2, 'surcharge': 2, 'faq': 1}
```

```bash
python ls pdf_docs/
```
Check all name of pdf files to understand the content we have.

### Step 3 — Ingest into ChromaDB

```bash
python ingest.py --reset # Generate chroma_db/
```

Chunks each document (1500 chars, 100 overlap), embeds using `all-MiniLM-L6-v2`, and upserts into ChromaDB.

Output example:
```
Total chunks: 813
Unique source URLs : 40
Chunks by doc_type:
  guide           411
  customs         67
  surcharge       14
  faq             5
  policy          3
```

### Step 4 — Test retrieval

```bash
python retriever.py "what documents are needed for customs clearance?"
```

---

## How It Works

### parse_local.py

Reads PDF files from `pdf_docs/`:

- Uses `pypdf` to extract text page by page
- Infers `doc_type` from filename keywords: `faq`, `customs`, `policy`, `surcharge`, `integration`, `guide`
- Each document gets metadata: `url`, `page_title`, `section`, `doc_type`, `source`
- Skipped automatically if PDF is scanned (no extractable text)

### ingest.py

Chunks and embeds documents into ChromaDB:

- Splits text into 1500-char chunks with 100-char overlap
- Embeds using ChromaDB's built-in `DefaultEmbeddingFunction` (`all-MiniLM-L6-v2`)
- Each chunk stores metadata: `url`, `page_title`, `section`, `doc_type`, `chunk_index`
- Processes one document at a time to avoid memory issues

Useful flags:
```bash
python ingest.py --reset          # drop and rebuild collection from scratch
python ingest.py --stats          # print collection stats without re-ingesting
python ingest.py --no-smoke-test  # skip retrieval test at the end
```

### retriever.py

Wraps ChromaDB for use by the Phase 2 agent:

```python
from retriever import Retriever

retriever = Retriever()
context, citations, found = retriever.retrieve("what documents do I need to ship to the UK?")
```

Returns:
- `context` — formatted text chunks ready to inject into an LLM prompt
- `citations` — list of `{title, url, doc_type}` dicts for source attribution
- `found` — `True` if at least one relevant chunk was found

Optional `doc_type_filter` to restrict search to a specific document type:

```python
context, citations, found = retriever.retrieve(
    "what items are prohibited?",
    doc_type_filter="policy"
)
```

---

## Knowledge Base

| Source | Count   |
|--------|---------|
| DHL PDF documents | 66      |
| **Chunks in ChromaDB** | **813** |

### Key documents included

- DHL eCommerce Terms and Conditions (US, Canada, Malaysia, Israel)
- DHL Express Rate and Service guides
- Customs Clearance guides
- Duty and Tax Prepayment guide
- Battery Shipping Policy
- Surcharge Policies (peak surcharges, fuel surcharges)
- Prohibited and Restricted Items
- Commercial Invoice templates
- Packaging Guidelines
- DHL e-Commerce Trends Reports (2024, 2025)
- Power of Attorney forms

---

## Common Issues

**`zsh: killed` when running ingest.py**
Killed due to low memory. Close Chrome and other heavy applications before running.

**`No relevant documents found above score threshold`**
Rebuild ChromaDB with `python ingest.py --reset` — embeddings may be mismatched from a previous run.

**PDF shows 0 or very few chars**
PDF is a scanned image and cannot be parsed by `pypdf`. Remove it from `pdf_docs/`.

**`invalid pdf header: b'<!DOC'`**
File was downloaded incorrectly (HTML error page saved instead of real PDF). Delete and re-download from browser.

**`Advanced encoding /SymbolEncoding not implemented yet`**
Some PDFs use special font encodings that `pypdf` cannot decode. These are warnings only — the rest of the text is still extracted normally.
