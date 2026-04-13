# Phase 1 RAG Pipeline

This is the historical milestone document for the original retrieval layer. Use the root `README.md` for current project setup and grading.

## Scope

Phase 1 covered the local DHL knowledge-base pipeline:

- Parse local PDFs from `pdf_docs/`
- Parse saved DHL webpages from `html_pages/`
- Clean and normalize extracted text
- Chunk documents while preserving source metadata
- Embed chunks with SentenceTransformers
- Persist vectors in ChromaDB
- Expose a reusable retrieval API
- Evaluate retrieval quality on DHL support-domain questions

## Setup

From the repository root:

```bash
python -m pip install -r requirements.txt
```

## Parse Local Documents

```bash
python parse_local.py --pdf-dir pdf_docs --html-dir html_pages --output raw_documents.json
```

`parse_local.py` writes one record per PDF page and one record per saved HTML article. Metadata includes source type, source filename, title, inferred category, page number, raw text, and cleaned text.

## Ingest Into ChromaDB

```bash
python ingest.py --reset
```

Defaults:

- PDF input: `pdf_docs/`
- HTML input: `html_pages/`
- Chroma path: `chroma_db/`
- Collection: `dhl_knowledge_base`
- Embedding model: `all-MiniLM-L6-v2`
- Chunk size: `800` characters
- Chunk overlap: `120` characters

Useful commands:

```bash
python ingest.py --stats
python ingest.py --no-html
python ingest.py --chunk-size 900 --chunk-overlap 150
python ingest.py --embedding-model all-MiniLM-L6-v2 --no-smoke-test
python ingest.py --local-files-only
```

The first run may need to download the embedding model from Hugging Face. Once cached, `--local-files-only` avoids network calls.

## Query Retrieval

```bash
python retriever.py "What documents are needed for customs clearance?" -k 5 --show-text
python retriever.py "Can I ship lithium batteries internationally?" --filter category=restricted_goods
```

Python integration:

```python
from retriever import Retriever

retriever = Retriever()
result = retriever.retrieve("How do I prepare a commercial invoice?", k=5)
```

`result["results"]` contains structured chunks with text, score, source filename, title, page range, chunk id, category, URI, and metadata.

## Evaluate Retrieval

```bash
python eval_retrieval.py -k 5
python eval_retrieval.py -k 10 --verbose
```

The evaluator uses DHL support-domain queries such as prohibited items, customs documents, commercial invoices, dangerous goods, surcharges, duties and taxes, shipment protection, and returns.
