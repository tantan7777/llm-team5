# CrossBorder Copilot

Conversational support assistant for cross-border ecommerce shipping. This repository currently contains the RAG layer for grounding answers in local DHL-related public documents.

## RAG Pipeline

The RAG module ingests local PDFs from `pdf_docs/`, extracts clean page text and metadata, chunks the documents, embeds chunks with a configurable SentenceTransformers model, and persists them in ChromaDB for downstream agent use.

### Files

- `parse_local.py` parses local PDFs into structured page records.
- `ingest.py` runs the parse, chunk, embed, and ChromaDB ingestion pipeline.
- `retriever.py` exposes `Retriever.retrieve(query, k=5, filters=None)`.
- `eval_retrieval.py` runs a small DHL-domain retrieval evaluation.
- `pdf_docs/` contains the local DHL-related PDF knowledge base.
- `chroma_db/` is generated locally after ingestion.

### Install

```bash
python -m pip install -r requirements.txt
```

### Optional: Inspect Parsed PDFs

```bash
python parse_local.py --pdf-dir pdf_docs --output raw_documents.json
```

This writes one JSON record per extracted PDF page with fields such as source filename, title, category, page number, raw text, and cleaned text.

### Ingest Local PDFs

```bash
python ingest.py --reset
```

Useful options:

```bash
python ingest.py --pdf-dir pdf_docs --chroma-dir chroma_db --collection dhl_knowledge_base
python ingest.py --embedding-model all-MiniLM-L6-v2 --chunk-size 800 --chunk-overlap 120
python ingest.py --local-files-only
python ingest.py --stats
```

The ingestion summary reports files processed, parsed pages, chunks created, chunks stored, and final collection size.

### Run One Retrieval Query

```bash
python retriever.py "What documents are needed for customs clearance?" -k 5 --show-text
```

Metadata filtering is supported with `key=value` filters:

```bash
python retriever.py "Can I ship lithium batteries internationally?" --filter category=restricted_goods
```

Python usage:

```python
from retriever import Retriever

retriever = Retriever()
result = retriever.retrieve(
    "What paperwork is needed for a commercial invoice?",
    k=5,
    filters={"category": "customs"},
)

for chunk in result["results"]:
    print(chunk["source_filename"], chunk["page_range"], chunk["score"])
    print(chunk["text"])
```

Each result includes:

- `text`
- `score`
- `source_filename`
- `title`
- `page_number`
- `page_range`
- `chunk_id`
- `document_id`
- `category`
- `source_uri`
- `metadata`

### Run Retrieval Evaluation

```bash
python eval_retrieval.py --k 5
python eval_retrieval.py --k 10 --verbose
```

The evaluation prints retrieved sources for realistic support questions and reports overall `hit@k`.

## Notes

- The default embedding model is `all-MiniLM-L6-v2`, which runs locally through `sentence-transformers`.
- The first run may download the embedding model from Hugging Face. After it is cached, use `--local-files-only` when running without network access.
- ChromaDB is persisted in `chroma_db/`.
- Rerunning ingestion uses stable chunk IDs and removes old chunks for re-ingested documents to avoid duplicate or stale records.
- The current scope is the RAG layer only: parsing, chunking, vector storage, retrieval, evaluation, and usage documentation.
