"""
ingest.py  —  CrossBorder Copilot Phase 1
Reads raw_documents.json, chunks each document, and upserts into ChromaDB.

Chunking strategy:
  - Splits on paragraph → heading → sentence → word boundaries (in that priority)
  - 1500-char chunks with 100-char overlap
  - Preserves semantic coherence better than fixed-window splitting

Usage:
    python ingest.py                          # ingest from raw_documents.json
    python ingest.py --input my_docs.json     # use a different source file
    python ingest.py --reset                  # drop existing collection first
    python ingest.py --stats                  # print collection stats and exit
"""

import json
import hashlib
import logging
import argparse
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions


class RecursiveCharacterTextSplitter:
    """
    Splits text recursively using a hierarchy of separators.
    Tries paragraph breaks first, then headings, then sentences, then words.
    This preserves semantic coherence — a chunk is more likely to contain
    a complete thought rather than cutting mid-sentence.
    """

    def __init__(
        self,
        chunk_size: int = 1500,
        chunk_overlap: int = 100,
        separators: list[str] | None = None,
        **_,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n## ", "\n", ". ", " ", ""]

    def _split_by_separator(self, text: str, separator: str) -> list[str]:
        """Split text by a separator, keeping the separator attached."""
        if separator == "":
            return list(text)
        parts = text.split(separator)
        # Re-attach separator to the beginning of each part (except the first)
        result = [parts[0]]
        for p in parts[1:]:
            result.append(separator + p)
        return [p for p in result if p]

    def _recursive_split(self, text: str, sep_index: int = 0) -> list[str]:
        """Recursively split text, trying coarser separators first."""
        if len(text) <= self.chunk_size:
            return [text]

        # If we've exhausted all separators, hard-cut
        if sep_index >= len(self.separators):
            chunks = []
            start = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                chunks.append(text[start:end])
                start += self.chunk_size - self.chunk_overlap
            return chunks

        separator = self.separators[sep_index]
        parts = self._split_by_separator(text, separator)

        # If this separator didn't help (only 1 part), try the next one
        if len(parts) <= 1:
            return self._recursive_split(text, sep_index + 1)

        # Merge parts into chunks that fit within chunk_size
        chunks = []
        current = ""
        for part in parts:
            if len(current) + len(part) <= self.chunk_size:
                current += part
            else:
                if current:
                    chunks.append(current)
                # If this single part is too big, split it recursively
                if len(part) > self.chunk_size:
                    sub_chunks = self._recursive_split(part, sep_index + 1)
                    chunks.extend(sub_chunks)
                    current = ""
                else:
                    current = part
        if current:
            chunks.append(current)

        # Add overlap between chunks
        if self.chunk_overlap > 0 and len(chunks) > 1:
            overlapped = [chunks[0]]
            for i in range(1, len(chunks)):
                prev = chunks[i - 1]
                overlap_text = prev[-self.chunk_overlap:] if len(prev) > self.chunk_overlap else prev
                overlapped.append(overlap_text + chunks[i])
            chunks = overlapped

        return chunks

    def split_text(self, text: str) -> list[str]:
        """Split text into chunks using recursive separator strategy."""
        raw_chunks = self._recursive_split(text.strip())
        # Clean up: strip whitespace, remove empty chunks
        return [c.strip() for c in raw_chunks if c.strip() and len(c.strip()) > 50]


# ── Config ────────────────────────────────────────────────────────────────────

CHROMA_DIR      = "./chroma_db"           # persisted ChromaDB directory
COLLECTION_NAME = "dhl_knowledge_base"
INPUT_FILE      = Path("raw_documents.json")

# Chunking parameters
CHUNK_SIZE    = 1500   # characters per chunk
CHUNK_OVERLAP = 100    # overlap between consecutive chunks

# Embedding model — runs locally, no API key needed
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

BATCH_SIZE = 64    # ChromaDB upsert batch size

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Chunking ─────────────────────────────────────────────────────────────────

def make_splitter() -> RecursiveCharacterTextSplitter:
    """
    RecursiveCharacterTextSplitter splits on paragraph → heading → sentence → word
    boundaries, preserving semantic coherence better than a fixed-size split.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n## ", "\n", ". ", " ", ""],
    )


def chunk_document(doc: dict, splitter: RecursiveCharacterTextSplitter) -> list[dict]:
    """
    Split one raw document into chunks.
    Each chunk inherits the document's metadata plus a chunk_index.
    """
    chunks_text = splitter.split_text(doc["text"])
    chunks = []
    for i, text in enumerate(chunks_text):
        chunk_id = hashlib.md5(
            f"{doc['url']}::chunk::{i}".encode()
        ).hexdigest()
        chunks.append({
            "id": chunk_id,
            "text": text,
            "metadata": {
                "url":          doc["url"],
                "page_title":   doc.get("page_title", ""),
                "section":      doc.get("section", ""),
                "doc_type":     doc.get("doc_type", "guide"),
                "chunk_index":  i,
                "total_chunks": len(chunks_text),
            },
        })
    return chunks


# ── ChromaDB helpers ──────────────────────────────────────────────────────────

def get_collection(reset: bool = False) -> chromadb.Collection:
    """Create or open the ChromaDB collection with a local embedding function."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embed_fn = embedding_functions.DefaultEmbeddingFunction()

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            log.info("Dropped existing collection '%s'", COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
    )
    return collection


def upsert_in_batches(collection: chromadb.Collection, chunks: list[dict]):
    """Upsert chunks to ChromaDB in batches to avoid memory spikes."""
    total = len(chunks)
    for start in range(0, total, BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        collection.upsert(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )
        log.info("  Upserted %d–%d / %d chunks", start + 1, start + len(batch), total)


# ── Stats helper ──────────────────────────────────────────────────────────────

def print_stats(collection: chromadb.Collection):
    count = collection.count()
    print(f"\nCollection : {collection.name}")
    print(f"Total chunks: {count}")

    if count == 0:
        return

    results = collection.get(limit=min(count, 500), include=["metadatas"])
    from collections import Counter
    doc_types = Counter(m["doc_type"] for m in results["metadatas"])
    urls      = len({m["url"] for m in results["metadatas"]})
    print(f"Unique source URLs : {urls}")
    print("Chunks by doc_type:")
    for dtype, n in doc_types.most_common():
        print(f"  {dtype:<15} {n}")
    print()


# ── Query helper (smoke test) ─────────────────────────────────────────────────

def smoke_test(collection: chromadb.Collection):
    """Run two quick queries to verify retrieval is working."""
    log.info("Running smoke tests...")

    queries = [
        "What documents are needed for customs clearance?",
        "Which items are prohibited or unacceptable for shipping?",
    ]

    for q in queries:
        results = collection.query(
            query_texts=[q],
            n_results=3,
            include=["documents", "metadatas", "distances"],
        )
        print(f"\n── Query: {q}")
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = 1 - dist
            print(f"  score={score:.3f}  [{meta['doc_type']}]  {meta['url']}")
            print(f"  {doc[:120].replace(chr(10), ' ')}...")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=str(INPUT_FILE))
    parser.add_argument("--reset",  action="store_true",
                        help="Drop existing collection before ingesting")
    parser.add_argument("--stats",  action="store_true",
                        help="Print collection statistics and exit")
    parser.add_argument("--no-smoke-test", action="store_true")
    args = parser.parse_args()

    collection = get_collection(reset=args.reset)

    if args.stats:
        print_stats(collection)
        return

    # ── Load raw documents ────────────────────────────────────────────────────
    input_path = Path(args.input)
    if not input_path.exists():
        log.error("Input file not found: %s — run parse_local.py first", input_path)
        return

    raw_docs = json.loads(input_path.read_text(encoding="utf-8"))
    log.info("Loaded %d raw documents from %s", len(raw_docs), input_path)

    # ── Chunk + Ingest (one doc at a time to save memory) ────────────────────
    splitter = make_splitter()
    total_chunks = 0
    for doc in raw_docs:
        log.info("  Processing: %s", doc.get("page_title", doc["url"])[:60])
        chunks = chunk_document(doc, splitter)
        log.info("  Chunked OK: %d chunks", len(chunks))
        if chunks:
            upsert_in_batches(collection, chunks)
            total_chunks += len(chunks)
            log.info("  %s  →  %d chunks", doc.get("page_title", doc["url"])[:60], len(chunks))

    log.info("Ingestion complete. Total chunks: %d, Collection size: %d",
             total_chunks, collection.count())

    print_stats(collection)

    if not args.no_smoke_test:
        smoke_test(collection)


if __name__ == "__main__":
    main()

