"""
retriever.py  —  CrossBorder Copilot
Reusable retrieval module consumed by the Phase 2 LangGraph agent.

Features:
  - Cosine similarity retrieval with configurable score threshold
  - Confidence-level classification (high / medium / low / none)
  - No-answer handling: returns found=False when knowledge base has no relevant info
  - Prompt template builder: packages retrieved context into a ready-to-send LLM prompt
  - Source citation formatting with deduplication

Usage (standalone smoke test):
    python retriever.py "what documents do I need to ship to Germany?"
    python retriever.py --verbose "prohibited items for shipping batteries"
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
MIN_SCORE       = 0.25   # cosine similarity threshold; chunks below this are dropped

# Confidence thresholds — used by the agent to decide response strategy
CONFIDENCE_HIGH   = 0.50  # top-1 score >= 0.50 → answer confidently with citations
CONFIDENCE_MEDIUM = 0.25  # top-1 score >= 0.25 → answer with disclaimer
                          # top-1 score <  0.25 → no relevant info found

log = logging.getLogger(__name__)


# ── System prompt template ────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are CrossBorder Copilot, an AI support agent for cross-border ecommerce shipping.
Your knowledge base contains DHL shipping documentation, customs clearance guides, surcharge policies,
prohibited items lists, and business support FAQs.

RULES:
1. Answer ONLY based on the provided context below. Do not use outside knowledge.
2. If the context does not contain enough information to answer the question, say:
   "I don't have enough information in my knowledge base to answer that. Please contact DHL support directly."
3. Always cite your sources using [Source N] references that match the context labels.
4. If the user asks about topics unrelated to shipping, customs, or logistics, politely decline
   and redirect them to the appropriate resource.
5. Be concise and professional. Use bullet points for multi-step instructions.
"""

RAG_PROMPT_TEMPLATE = """{system_prompt}

--- RETRIEVED CONTEXT (confidence: {confidence}) ---
{context}
--- END CONTEXT ---

CITATIONS AVAILABLE:
{citations_block}

USER QUESTION: {query}

Respond based on the context above. Cite sources as [Source N] where applicable.
If confidence is "low" or "none", inform the user that you could not find a reliable answer."""


class Retriever:
    """
    Retrieval module for CrossBorder Copilot.

    Handles:
      - Vector similarity search against ChromaDB
      - Score-based filtering and confidence classification
      - Context formatting with source labels
      - Full prompt construction for the LLM

    Example
    -------
    retriever = Retriever()
    result    = retriever.retrieve("commercial invoice requirements UK")
    prompt    = retriever.build_prompt("commercial invoice requirements UK", result)
    """

    def __init__(self, chroma_dir: str = CHROMA_DIR):
        client = chromadb.PersistentClient(path=chroma_dir)
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
        List of dicts with keys: text, score, url, page_title, section, doc_type, relevance
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

            # Tag each chunk with a relevance label so the LLM knows what to trust
            if score >= CONFIDENCE_HIGH:
                relevance = "high"
            elif score >= CONFIDENCE_MEDIUM:
                relevance = "medium"
            else:
                relevance = "low"

            results.append({
                "text":       text,
                "score":      round(score, 4),
                "relevance":  relevance,
                "url":        meta.get("url", ""),
                "page_title": meta.get("page_title", ""),
                "section":    meta.get("section", ""),
                "doc_type":   meta.get("doc_type", ""),
            })

        return results

    # ── Confidence classification ─────────────────────────────────────────────

    @staticmethod
    def classify_confidence(results: list[dict]) -> str:
        """
        Classify overall retrieval confidence based on the top-1 result score.

        Returns
        -------
        "high"   : top-1 score >= 0.50 — answer confidently
        "medium" : top-1 score >= 0.25 — answer with disclaimer
        "low"    : results exist but all below 0.25 — weak match
        "none"   : no results passed MIN_SCORE — refuse to answer
        """
        if not results:
            return "none"
        top_score = results[0]["score"]
        if top_score >= CONFIDENCE_HIGH:
            return "high"
        elif top_score >= CONFIDENCE_MEDIUM:
            return "medium"
        else:
            return "low"

    # ── Formatting helpers ────────────────────────────────────────────────────

    def format_context(self, results: list[dict]) -> str:
        """
        Build a context string to inject into an LLM prompt.
        Each chunk is labelled with source, relevance, and doc_type
        so the LLM can prioritize high-relevance chunks and cite properly.
        """
        if not results:
            return "(No relevant documents found in the knowledge base.)"

        parts = []
        for i, r in enumerate(results, 1):
            parts.append(
                f"[Source {i}] (relevance: {r['relevance']}, type: {r['doc_type']})\n"
                f"Title: {r['page_title']}\n"
                f"URL: {r['url']}\n"
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
                    "title":    r["page_title"] or r["section"],
                    "url":      r["url"],
                    "doc_type": r["doc_type"],
                    "score":    r["score"],
                })
        return citations

    def _citations_block(self, citations: list[dict]) -> str:
        """Format citations as a readable block for the prompt."""
        if not citations:
            return "(No sources available)"
        lines = []
        for i, c in enumerate(citations, 1):
            lines.append(f"  [Source {i}] {c['title']} — {c['url']}")
        return "\n".join(lines)

    # ── Prompt builder ────────────────────────────────────────────────────────

    def build_prompt(
        self,
        query: str,
        results: list[dict],
        confidence: str | None = None,
    ) -> str:
        """
        Construct a complete LLM prompt with system instructions,
        retrieved context, citation list, and the user question.

        Parameters
        ----------
        query      : the user's original question
        results    : list of retrieved chunks from self.query()
        confidence : override confidence level; auto-classified if None

        Returns
        -------
        A fully formatted prompt string ready to send to the LLM.
        """
        if confidence is None:
            confidence = self.classify_confidence(results)

        context   = self.format_context(results)
        citations = self.format_citations(results)

        return RAG_PROMPT_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT.strip(),
            confidence=confidence,
            context=context,
            citations_block=self._citations_block(citations),
            query=query,
        )

    # ── High-level retrieve (used by Phase 2 agent) ───────────────────────────

    def retrieve(
        self,
        query_text: str,
        top_k: int = DEFAULT_TOP_K,
        doc_type_filter: str | None = None,
    ) -> dict:
        """
        High-level retrieval method used by the agent.

        Returns
        -------
        dict with keys:
            context    : formatted context string for the LLM prompt
            citations  : list of {title, url, doc_type, score} dicts
            found      : True if at least one chunk passed the score threshold
            confidence : "high", "medium", "low", or "none"
            results    : raw list of retrieved chunks (for further processing)
            prompt     : complete LLM prompt ready to send
        """
        results    = self.query(query_text, top_k=top_k, doc_type_filter=doc_type_filter)
        context    = self.format_context(results)
        citations  = self.format_citations(results)
        confidence = self.classify_confidence(results)
        found      = len(results) > 0
        prompt     = self.build_prompt(query_text, results, confidence)

        return {
            "context":    context,
            "citations":  citations,
            "found":      found,
            "confidence": confidence,
            "results":    results,
            "prompt":     prompt,
        }


# ── Standalone smoke test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    verbose = "--verbose" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--verbose"]
    query = " ".join(args) or "What documents are needed for customs clearance?"

    print(f"\nQuery: {query}\n")

    retriever = Retriever()
    result = retriever.retrieve(query)

    print(f"Found: {result['found']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Chunks retrieved: {len(result['results'])}")

    if result["results"]:
        print(f"Top score: {result['results'][0]['score']}")

    if not result["found"]:
        print("\nNo relevant documents found above score threshold.")
        print("The agent would respond: 'I don't have enough information in my")
        print("knowledge base to answer that. Please contact DHL support directly.'")
    else:
        print("\n=== Context (first 1500 chars) ===")
        print(result["context"][:1500])
        print("\n=== Citations ===")
        for c in result["citations"]:
            print(f"  [{c['doc_type']}] (score: {c['score']})  {c['title']}")
            print(f"           {c['url']}")

    if verbose:
        print("\n=== Full LLM Prompt ===")
        print(result["prompt"])

