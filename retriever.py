"""
Reusable retrieval interface for CrossBorder Copilot's RAG layer.

The public entry point is Retriever.retrieve(query, k=5, filters=None). It
returns structured chunks with text, score, source filename, title, page range,
chunk id, and full metadata for downstream citation handling.

Usage:
    python retriever.py "customs clearance documents" -k 5
    python retriever.py "dangerous goods batteries" --filter category=restricted_goods
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from chromadb.utils import embedding_functions
from ingest import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL


DEFAULT_TOP_K = 5
CONFIDENCE_HIGH = 0.55
CONFIDENCE_MEDIUM = 0.30

log = logging.getLogger(__name__)

def make_embedding_function(model_name: str, local_files_only: bool = False):
    kwargs = {"model_name": model_name}
    if local_files_only:
        kwargs["model_kwargs"] = {"local_files_only": True}
    return embedding_functions.SentenceTransformerEmbeddingFunction(**kwargs)

SYSTEM_PROMPT = """You are CrossBorder Copilot, a support assistant for cross-border ecommerce shipping.
Answer only from the retrieved DHL-related context. If the context is insufficient, say that the knowledge base does not contain enough information and recommend contacting DHL support. Cite sources using the provided source labels."""


class Retriever:
    """Thin wrapper around a persistent ChromaDB collection."""

    def __init__(
        self,
        chroma_dir: str | Path = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
        local_files_only: bool = False,
    ) -> None:
        self.chroma_dir = str(chroma_dir)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.local_files_only = local_files_only
        self.collection = self._load_collection()

    def _load_collection(self) -> Collection:
        client = chromadb.PersistentClient(path=self.chroma_dir)
        embedding_function = make_embedding_function(
            self.embedding_model,
            local_files_only=self.local_files_only,
        )
        try:
            return client.get_collection(
                name=self.collection_name,
                embedding_function=embedding_function,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not open Chroma collection '{self.collection_name}' at {self.chroma_dir}. "
                "Run `python ingest.py --reset` first."
            ) from exc

    def retrieve(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
        where_document: dict[str, Any] | None = None,
        top_k: int | None = None,
        doc_type_filter: str | None = None,
    ) -> dict[str, Any]:
        """
        Retrieve top-k semantically relevant chunks.

        Parameters
        ----------
        query:
            Natural language merchant-support question.
        k:
            Number of chunks to return.
        filters:
            Optional Chroma metadata filters, for example
            {"category": "customs"} or {"source_filename": "sg_commercial_invoice.pdf"}.
        score_threshold:
            Optional minimum similarity score. By default all top-k results are
            returned so downstream callers can make their own confidence choice.
        where_document:
            Optional Chroma document-text filter.
        """
        if top_k is not None:
            k = top_k
        if doc_type_filter:
            filters = dict(filters or {})
            filters["doc_type"] = doc_type_filter

        raw = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=filters,
            where_document=where_document,
            include=["documents", "metadatas", "distances"],
        )

        results = []
        for text, metadata, distance in zip(
            raw.get("documents", [[]])[0],
            raw.get("metadatas", [[]])[0],
            raw.get("distances", [[]])[0],
        ):
            score = distance_to_score(distance)
            if score_threshold is not None and score < score_threshold:
                continue
            results.append(format_result(text, metadata, score))

        confidence = classify_confidence(results)
        return {
            "query": query,
            "found": bool(results),
            "confidence": confidence,
            "results": results,
            "citations": format_citations(results),
            "context": format_context(results),
        }

    def pretty_print(self, retrieval_result: dict[str, Any], show_text: bool = False) -> None:
        pretty_print(retrieval_result, show_text=show_text)

    def build_prompt(self, query: str, retrieval_result: dict[str, Any]) -> str:
        """Build a compact grounded-answer prompt for an LLM caller."""
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"Retrieved context, confidence={retrieval_result['confidence']}:\n"
            f"{retrieval_result['context']}\n\n"
            f"Question: {query}\n"
            "Answer with citations such as [Source 1]."
        )


def distance_to_score(distance: float | int | None) -> float:
    """Convert Chroma cosine distance to a readable similarity score."""
    if distance is None:
        return 0.0
    try:
        value = 1.0 - float(distance)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(value):
        return 0.0
    return round(max(min(value, 1.0), -1.0), 4)


def format_result(text: str, metadata: dict[str, Any], score: float) -> dict[str, Any]:
    source_filename = str(metadata.get("source_filename", ""))
    title = str(metadata.get("title", ""))
    page_range = str(metadata.get("page_range", metadata.get("page_number", "")))
    return {
        "text": text,
        "score": score,
        "source_filename": source_filename,
        "title": title,
        "page_number": metadata.get("page_number", ""),
        "page_range": page_range,
        "chunk_id": metadata.get("chunk_id", ""),
        "document_id": metadata.get("document_id", ""),
        "category": metadata.get("category", ""),
        "doc_type": metadata.get("doc_type", metadata.get("category", "")),
        "source_uri": metadata.get("source_uri", ""),
        "metadata": dict(metadata),
    }


def classify_confidence(results: list[dict[str, Any]]) -> str:
    if not results:
        return "none"
    top_score = float(results[0]["score"])
    if top_score >= CONFIDENCE_HIGH:
        return "high"
    if top_score >= CONFIDENCE_MEDIUM:
        return "medium"
    return "low"


def format_citations(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return unique source citations in result order."""
    citations = []
    seen: set[tuple[str, str]] = set()
    for result in results:
        key = (str(result["source_filename"]), str(result["page_range"]))
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            {
                "source_filename": result["source_filename"],
                "title": result["title"],
                "page_range": result["page_range"],
                "category": result["category"],
                "doc_type": result["doc_type"],
                "score": result["score"],
                "source_uri": result["source_uri"],
            }
        )
    return citations


def format_context(results: list[dict[str, Any]], max_chars_per_chunk: int = 1400) -> str:
    if not results:
        return "(No relevant chunks retrieved.)"

    blocks = []
    for index, result in enumerate(results, start=1):
        text = result["text"]
        if len(text) > max_chars_per_chunk:
            text = f"{text[:max_chars_per_chunk].rstrip()}..."
        blocks.append(
            f"[Source {index}] {result['title']} "
            f"({result['source_filename']}, p. {result['page_range']}, {result['category']})\n"
            f"{text}"
        )
    return "\n\n---\n\n".join(blocks)


def pretty_print(retrieval_result: dict[str, Any], show_text: bool = False) -> None:
    print(f"\nQuery      : {retrieval_result['query']}")
    print(f"Found      : {retrieval_result['found']}")
    print(f"Confidence : {retrieval_result['confidence']}")
    print(f"Results    : {len(retrieval_result['results'])}")

    for index, result in enumerate(retrieval_result["results"], start=1):
        print(
            f"\n[{index}] score={result['score']:.3f}  "
            f"{result['source_filename']} p.{result['page_range']}  "
            f"[{result['category']}]"
        )
        print(f"    title    : {result['title']}")
        print(f"    chunk_id : {result['chunk_id']}")
        if show_text:
            print(f"    text     : {result['text'][:1200]}")


def parse_filter(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("filters must use key=value syntax")
    key, filter_value = value.split("=", 1)
    key = key.strip()
    filter_value = filter_value.strip()
    if not key or not filter_value:
        raise argparse.ArgumentTypeError("filters must use key=value syntax")
    return key, filter_value


def retrieve(query: str, k: int = DEFAULT_TOP_K, filters: dict[str, Any] | None = None) -> dict[str, Any]:
    """Convenience function for simple downstream imports."""
    return Retriever().retrieve(query, k=k, filters=filters)


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the local DHL RAG knowledge base.")
    parser.add_argument("query", nargs="*", help="Question to retrieve context for")
    parser.add_argument("-k", "--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--filter", action="append", type=parse_filter, default=[], help="Metadata filter key=value")
    parser.add_argument("--min-score", type=float, default=None, help="Optional minimum similarity score")
    parser.add_argument("--chroma-dir", default=str(CHROMA_DIR))
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load the embedding model from the local Hugging Face cache only",
    )
    parser.add_argument("--show-text", action="store_true", help="Print retrieved chunk text")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")
    query = " ".join(args.query).strip() or "What documents are needed for customs clearance?"
    filters = dict(args.filter) if args.filter else None

    retriever = Retriever(
        chroma_dir=args.chroma_dir,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        local_files_only=args.local_files_only,
    )
    result = retriever.retrieve(query, k=args.top_k, filters=filters, score_threshold=args.min_score)
    retriever.pretty_print(result, show_text=args.show_text)


if __name__ == "__main__":
    main()
