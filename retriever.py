"""
retriever.py  —  CrossBorder Copilot
Reusable retrieval module consumed by the Phase 2 LangGraph agent.

Usage (standalone smoke test):
    python retriever.py "what documents do I need to ship to Germany?"
"""

from __future__ import annotations
import sys
import logging

import chromadb
from chromadb.utils import embedding_functions

# ── Config (must match ingest.py) ─────────────────────────────────────────────
CHROMA_DIR      = "./chroma_db"
COLLECTION_NAME = "dhl_knowledge_base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K   = 5
MIN_SCORE       = 0.0   # cosine similarity threshold; chunks below this are dropped

log = logging.getLogger(__name__)


class Retriever:
    """
    Thin wrapper around ChromaDB for use by the LangGraph agent.

    Example
    -------
    retriever = Retriever()
    results   = retriever.query("commercial invoice requirements UK")
    context   = retriever.format_context(results)
    """

    def __init__(self):
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        from chromadb import EmbeddingFunction, Embeddings
        import hashlib

        embed_fn = embedding_functions.DefaultEmbeddingFunction()

        self.collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embed_fn,
        )

    # ── Core query ────────────────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        top_k: int = DEFAULT_TOP_K,
        doc_type_filter: str | None = None,
    ) -> list[dict]:
        """
        Retrieve the top-k most relevant chunks.

        Parameters
        ----------
        query_text      : natural language question
        top_k           : number of results to return
        doc_type_filter : restrict to a specific doc_type
                          ("faq", "customs", "policy", "surcharge", "guide")

        Returns
        -------
        List of dicts with keys: text, score, url, page_title, section, doc_type
        """
        where = {"doc_type": doc_type_filter} if doc_type_filter else None

        raw = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
            where=where,
        )

        results = []
        for text, meta, dist in zip(
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        ):
            score = 1.0 - dist          # convert cosine distance → similarity
            if score < MIN_SCORE:
                continue
            results.append({
                "text":       text,
                "score":      round(score, 4),
                "url":        meta.get("url", ""),
                "page_title": meta.get("page_title", ""),
                "section":    meta.get("section", ""),
                "doc_type":   meta.get("doc_type", ""),
            })

        return results

    # ── Formatting helpers ────────────────────────────────────────────────────

    def format_context(self, results: list[dict]) -> str:
        """
        Build a context string to inject into an LLM prompt.
        Each chunk is labelled with its source URL so the LLM can cite it.
        Returns an empty string if no results pass the score threshold.
        """
        if not results:
            return ""

        parts = []
        for i, r in enumerate(results, 1):
            parts.append(
                f"[Source {i}] {r['page_title']} — {r['url']}\n"
                f"{r['text']}"
            )
        return "\n\n---\n\n".join(parts)

    def format_citations(self, results: list[dict]) -> list[dict]:
        """
        Return a compact list of unique citations for the agent to attach
        to its response (deduped by URL).
        """
        seen: set[str] = set()
        citations = []
        for r in results:
            if r["url"] not in seen:
                seen.add(r["url"])
                citations.append({
                    "title": r["page_title"] or r["section"],
                    "url":   r["url"],
                    "doc_type": r["doc_type"],
                })
        return citations

    # ── Convenience: RAG context + citations in one call ─────────────────────

    def retrieve(
        self,
        query_text: str,
        top_k: int = DEFAULT_TOP_K,
        doc_type_filter: str | None = None,
    ) -> tuple[str, list[dict], bool]:
        """
        High-level method used by the agent.

        Returns
        -------
        context    : formatted context string for the LLM prompt
        citations  : list of {title, url, doc_type} dicts
        found      : True if at least one chunk passed the score threshold
        """
        results   = self.query(query_text, top_k=top_k, doc_type_filter=doc_type_filter)
        context   = self.format_context(results)
        citations = self.format_citations(results)
        found     = len(results) > 0
        return context, citations, found


# ── Standalone smoke test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    query = " ".join(sys.argv[1:]) or "What documents are needed for customs clearance?"
    print(f"\nQuery: {query}\n")

    retriever = Retriever()
    context, citations, found = retriever.retrieve(query)

    if not found:
        print("No relevant documents found above score threshold.")
    else:
        print("=== Context ===")
        print(context[:1500])
        print("\n=== Citations ===")
        for c in citations:
            print(f"  [{c['doc_type']}]  {c['title']}")
            print(f"           {c['url']}")
