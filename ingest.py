"""
Ingest local DHL PDF documents into a persistent ChromaDB collection.

The pipeline parses PDFs from pdf_docs/, chunks cleaned page text, embeds each
chunk, and upserts it into ChromaDB with citation-friendly metadata.

Usage:
    python ingest.py --reset
    python ingest.py --stats
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils import embedding_functions

from parse_local import parse_local_documents


PDF_DIR = Path("pdf_docs")
CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "dhl_knowledge_base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
MIN_CHUNK_CHARS = 120
BATCH_SIZE = 64

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TextChunk:
    """A retrieval chunk with flattened Chroma-compatible metadata."""

    id: str
    text: str
    metadata: dict[str, str | int | float | bool]


class RecursiveCharacterTextSplitter:
    """
    Lightweight recursive splitter for support/policy documents.

    It tries to split at paragraph, heading, sentence, and word boundaries before
    falling back to hard character windows.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        separators: list[str] | None = None,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n# ", "\n", ". ", "; ", ", ", " ", ""]

    def split_text(self, text: str) -> list[str]:
        raw_chunks = self._recursive_split(text.strip(), separator_index=0)
        return [chunk.strip() for chunk in raw_chunks if chunk.strip()]

    def _recursive_split(self, text: str, separator_index: int) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]
        if separator_index >= len(self.separators):
            return self._hard_split(text)

        separator = self.separators[separator_index]
        if not separator:
            return self._hard_split(text)

        parts = self._split_keep_separator(text, separator)
        if len(parts) == 1:
            return self._recursive_split(text, separator_index + 1)

        chunks: list[str] = []
        current = ""
        for part in parts:
            if len(current) + len(part) <= self.chunk_size:
                current += part
                continue

            if current:
                chunks.append(current)
                current = ""

            if len(part) > self.chunk_size:
                chunks.extend(self._recursive_split(part, separator_index + 1))
            else:
                current = part

        if current:
            chunks.append(current)

        return self._add_overlap(chunks)

    @staticmethod
    def _split_keep_separator(text: str, separator: str) -> list[str]:
        parts = text.split(separator)
        if len(parts) <= 1:
            return [text]
        chunks = [parts[0]]
        chunks.extend(f"{separator}{part}" for part in parts[1:])
        return [chunk for chunk in chunks if chunk]

    def _hard_split(self, text: str) -> list[str]:
        chunks = []
        step = self.chunk_size - self.chunk_overlap
        for start in range(0, len(text), step):
            chunks.append(text[start : start + self.chunk_size])
        return chunks

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        if self.chunk_overlap <= 0 or len(chunks) <= 1:
            return chunks

        overlapped = [chunks[0]]
        for chunk in chunks[1:]:
            previous = overlapped[-1]
            overlap = previous[-self.chunk_overlap :]
            overlapped.append(f"{overlap}{chunk}")
        return overlapped


def make_splitter(chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def normalize_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def content_hash(text: str) -> str:
    return hashlib.sha1(normalize_for_hash(text).encode("utf-8")).hexdigest()


def clean_chunk_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def group_pages_by_document(parsed_pages: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for page in parsed_pages:
        grouped[page["document_id"]].append(page)
    for pages in grouped.values():
        pages.sort(key=lambda item: int(item.get("page_number", 0)))
    return grouped


def page_range_label(page_numbers: list[int]) -> str:
    if not page_numbers:
        return ""
    start, end = min(page_numbers), max(page_numbers)
    return str(start) if start == end else f"{start}-{end}"


def chunk_segment(
    segment_text: str,
    segment_pages: list[dict[str, Any]],
    splitter: RecursiveCharacterTextSplitter,
    document_chunk_start: int,
    min_chunk_chars: int,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    page_numbers = [int(page["page_number"]) for page in segment_pages]
    first_page = segment_pages[0]
    text_parts = splitter.split_text(segment_text)

    for local_index, chunk_text in enumerate(text_parts):
        chunk_text = clean_chunk_text(chunk_text)
        if len(chunk_text) < min_chunk_chars and len(text_parts) > 1:
            continue

        chunk_index = document_chunk_start + len(chunks)
        chunk_id = f"{first_page['document_id']}:chunk:{chunk_index:04d}"
        chunk_hash = content_hash(chunk_text)
        pages_label = page_range_label(page_numbers)
        page_start = min(page_numbers)
        page_end = max(page_numbers)

        metadata: dict[str, str | int | float | bool] = {
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "document_id": str(first_page["document_id"]),
            "source_type": str(first_page.get("source_type", "pdf")),
            "source_filename": str(first_page.get("source_filename", "")),
            "source_path": str(first_page.get("source_path", "")),
            "source_uri": str(first_page.get("source_uri", "")),
            "title": str(first_page.get("title", "")),
            "category": str(first_page.get("category", "general")),
            "doc_type": str(first_page.get("category", "general")),
            "page_number": page_start,
            "page_start": page_start,
            "page_end": page_end,
            "page_range": pages_label,
            "content_hash": chunk_hash,
            "chunk_chars": len(chunk_text),
        }
        chunks.append(TextChunk(id=chunk_id, text=chunk_text, metadata=metadata))

    return chunks


def chunk_parsed_documents(
    parsed_pages: list[dict[str, Any]],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    min_chunk_chars: int = MIN_CHUNK_CHARS,
) -> list[TextChunk]:
    """Chunk parsed page records while preserving source and page metadata."""
    splitter = make_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks: list[TextChunk] = []

    for pages in group_pages_by_document(parsed_pages).values():
        document_chunks: list[TextChunk] = []
        buffer_pages: list[dict[str, Any]] = []
        buffer_text: list[str] = []

        def flush_buffer() -> None:
            nonlocal buffer_pages, buffer_text, document_chunks
            if not buffer_pages:
                return
            text = "\n\n".join(buffer_text)
            document_chunks.extend(
                chunk_segment(
                    text,
                    buffer_pages,
                    splitter,
                    document_chunk_start=len(document_chunks),
                    min_chunk_chars=min_chunk_chars,
                )
            )
            buffer_pages = []
            buffer_text = []

        for page in pages:
            text = clean_chunk_text(str(page.get("text", "")))
            if not text:
                continue

            if len(text) < min_chunk_chars:
                buffer_pages.append(page)
                buffer_text.append(text)
                if len("\n\n".join(buffer_text)) >= min_chunk_chars:
                    flush_buffer()
                continue

            flush_buffer()
            document_chunks.extend(
                chunk_segment(
                    text,
                    [page],
                    splitter,
                    document_chunk_start=len(document_chunks),
                    min_chunk_chars=min_chunk_chars,
                )
            )

        flush_buffer()
        all_chunks.extend(merge_final_tiny_chunk(document_chunks, min_chunk_chars=min_chunk_chars))

    return deduplicate_chunks(all_chunks)


def merge_final_tiny_chunk(chunks: list[TextChunk], min_chunk_chars: int) -> list[TextChunk]:
    """Merge a short trailing fragment into the previous chunk when practical."""
    if len(chunks) < 2 or len(chunks[-1].text) >= min_chunk_chars:
        return chunks

    previous = chunks[-2]
    tiny = chunks[-1]
    merged_text = clean_chunk_text(f"{previous.text}\n\n{tiny.text}")
    merged_meta = dict(previous.metadata)
    merged_meta["page_end"] = tiny.metadata.get("page_end", previous.metadata.get("page_end", 0))
    merged_meta["page_range"] = page_range_label(
        [int(previous.metadata.get("page_start", 0)), int(merged_meta["page_end"])]
    )
    merged_meta["chunk_chars"] = len(merged_text)
    merged_meta["content_hash"] = content_hash(merged_text)
    return [*chunks[:-2], TextChunk(id=previous.id, text=merged_text, metadata=merged_meta)]


def deduplicate_chunks(chunks: list[TextChunk]) -> list[TextChunk]:
    """Drop exact duplicate chunks after whitespace/case normalization."""
    seen_hashes: set[str] = set()
    unique_chunks: list[TextChunk] = []
    for chunk in chunks:
        hash_value = str(chunk.metadata["content_hash"])
        if hash_value in seen_hashes:
            continue
        seen_hashes.add(hash_value)
        unique_chunks.append(chunk)
    return unique_chunks


def make_embedding_function(model_name: str = EMBEDDING_MODEL, local_files_only: bool = False):
    """Create the embedding function shared by ingestion and retrieval."""
    kwargs = {"local_files_only": True} if local_files_only else {}
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name, **kwargs)


def get_collection(
    chroma_dir: Path | str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    embedding_model: str = EMBEDDING_MODEL,
    local_files_only: bool = False,
    reset: bool = False,
) -> Collection:
    client = chromadb.PersistentClient(path=str(chroma_dir))
    if reset:
        try:
            client.delete_collection(collection_name)
            log.info("Dropped existing Chroma collection '%s'", collection_name)
        except Exception:
            log.info("No existing Chroma collection named '%s' to drop", collection_name)

    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=make_embedding_function(embedding_model, local_files_only=local_files_only),
        metadata={"hnsw:space": "cosine"},
    )


def delete_existing_documents(collection: Collection, document_ids: set[str]) -> None:
    """Remove existing chunks for re-ingested documents to avoid stale leftovers."""
    for document_id in sorted(document_ids):
        try:
            collection.delete(where={"document_id": document_id})
        except Exception as exc:
            log.debug("Could not delete old chunks for document_id=%s: %s", document_id, exc)


def upsert_chunks(collection: Collection, chunks: list[TextChunk], batch_size: int = BATCH_SIZE) -> int:
    stored = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        collection.upsert(
            ids=[chunk.id for chunk in batch],
            documents=[chunk.text for chunk in batch],
            metadatas=[chunk.metadata for chunk in batch],
        )
        stored += len(batch)
        log.info("Upserted chunks %d-%d / %d", start + 1, start + len(batch), len(chunks))
    return stored


def ingest_documents(
    pdf_dir: Path | str = PDF_DIR,
    chroma_dir: Path | str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    embedding_model: str = EMBEDDING_MODEL,
    local_files_only: bool = False,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    min_chunk_chars: int = MIN_CHUNK_CHARS,
    batch_size: int = BATCH_SIZE,
    reset: bool = False,
    delete_existing: bool = True,
) -> dict[str, int]:
    """Run the full parse -> chunk -> Chroma ingestion pipeline."""
    parsed_pages = parse_local_documents(pdf_dir)
    chunks = chunk_parsed_documents(
        parsed_pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_chars=min_chunk_chars,
    )
    collection = get_collection(
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        embedding_model=embedding_model,
        local_files_only=local_files_only,
        reset=reset,
    )

    document_ids = {str(page["document_id"]) for page in parsed_pages}
    if delete_existing and not reset:
        delete_existing_documents(collection, document_ids)

    stored = upsert_chunks(collection, chunks, batch_size=batch_size)
    summary = {
        "files_processed": len({page["source_filename"] for page in parsed_pages}),
        "parsed_pages": len(parsed_pages),
        "chunks_created": len(chunks),
        "chunks_stored": stored,
        "collection_count": collection.count(),
    }
    return summary


def print_stats(
    chroma_dir: Path | str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    embedding_model: str = EMBEDDING_MODEL,
    local_files_only: bool = False,
) -> None:
    collection = get_collection(
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        embedding_model=embedding_model,
        local_files_only=local_files_only,
    )
    count = collection.count()
    print(f"\nCollection : {collection_name}")
    print(f"Path       : {chroma_dir}")
    print(f"Chunks     : {count}")
    if count == 0:
        return

    sample = collection.get(limit=min(count, 1000), include=["metadatas"])
    metadatas = sample.get("metadatas", [])
    categories = Counter(meta.get("category", "unknown") for meta in metadatas)
    filenames = Counter(meta.get("source_filename", "unknown") for meta in metadatas)
    print(f"Sources    : {len(filenames)} in first {len(metadatas)} chunks")
    print("Categories :")
    for category, value in categories.most_common():
        print(f"  {category:<18} {value}")
    print("Top sources:")
    for filename, value in filenames.most_common(10):
        print(f"  {value:>4}  {filename}")


def smoke_test(
    chroma_dir: Path | str,
    collection_name: str,
    embedding_model: str,
    local_files_only: bool,
    top_k: int = 3,
) -> None:
    from retriever import Retriever

    retriever = Retriever(
        chroma_dir=str(chroma_dir),
        collection_name=collection_name,
        embedding_model=embedding_model,
        local_files_only=local_files_only,
    )
    queries = [
        "What documents are needed for customs clearance?",
        "Which items are prohibited or restricted for international shipping?",
    ]
    for query in queries:
        result = retriever.retrieve(query, k=top_k)
        print(f"\nQuery: {query}")
        for item in result["results"]:
            print(
                f"  score={item['score']:.3f}  "
                f"{item['source_filename']} p.{item['page_range']}  "
                f"[{item['category']}]"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest local DHL PDFs into ChromaDB.")
    parser.add_argument("--pdf-dir", default=str(PDF_DIR), help="Directory containing PDFs")
    parser.add_argument("--chroma-dir", default=str(CHROMA_DIR), help="Persistent ChromaDB directory")
    parser.add_argument("--collection", default=COLLECTION_NAME, help="ChromaDB collection name")
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL, help="SentenceTransformers model name")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load the embedding model from the local Hugging Face cache only",
    )
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    parser.add_argument("--min-chunk-chars", type=int, default=MIN_CHUNK_CHARS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--reset", action="store_true", help="Drop and rebuild the collection")
    parser.add_argument("--stats", action="store_true", help="Print collection stats and exit")
    parser.add_argument("--no-smoke-test", action="store_true", help="Skip sample retrieval queries")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s  %(message)s")

    if args.stats:
        print_stats(args.chroma_dir, args.collection, args.embedding_model, local_files_only=args.local_files_only)
        return

    summary = ingest_documents(
        pdf_dir=args.pdf_dir,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        local_files_only=args.local_files_only,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_chunk_chars=args.min_chunk_chars,
        batch_size=args.batch_size,
        reset=args.reset,
    )

    print("\nIngestion summary")
    print(f"  Files processed       : {summary['files_processed']}")
    print(f"  Parsed documents/pages: {summary['parsed_pages']}")
    print(f"  Chunks created        : {summary['chunks_created']}")
    print(f"  Chunks stored         : {summary['chunks_stored']}")
    print(f"  Collection size       : {summary['collection_count']}")

    print_stats(args.chroma_dir, args.collection, args.embedding_model, local_files_only=args.local_files_only)
    if not args.no_smoke_test:
        smoke_test(args.chroma_dir, args.collection, args.embedding_model, args.local_files_only)


if __name__ == "__main__":
    main()
