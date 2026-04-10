# Phase 1 — RAG Pipeline

This phase builds the local retrieval layer for CrossBorder Copilot. It parses DHL-related PDFs from `pdf_docs/`, chunks them with metadata, embeds the chunks, stores them in ChromaDB, and exposes a reusable retrieval API for a downstream support agent.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Parse PDFs

```bash
python parse_local.py --pdf-dir pdf_docs --output raw_documents.json
```

`parse_local.py` writes one record per PDF page. Metadata includes source filename, title, inferred category, page number, total pages, raw text, and cleaned text.

## Ingest Into ChromaDB

```bash
python ingest.py --reset
```

Defaults:

- PDF input: `pdf_docs/`
- Chroma path: `chroma_db/`
- Collection: `dhl_knowledge_base`
- Embedding model: `all-MiniLM-L6-v2`
- Chunk size: `800` characters
- Chunk overlap: `120` characters

Useful commands:

```bash
python ingest.py --stats
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
python eval_retrieval.py --k 5
python eval_retrieval.py --k 10 --verbose
```

The evaluator uses DHL support-domain queries such as prohibited items, customs documents, commercial invoice, dangerous goods, surcharges, duties and taxes, shipment protection, and returns. It prints retrieved sources, hit or miss, and overall `hit@k`.

## Scope

This phase intentionally covers only the RAG layer:

- local PDF parsing
- text cleaning
- chunking and metadata preservation
- embeddings and ChromaDB storage
- retrieval interface
- retrieval evaluation

It does not implement agent workflows, action logic, frontend UI, or ticketing logic.
