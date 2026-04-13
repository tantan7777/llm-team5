"""
Knowledge-base retrieval tool for DHL-related local documents.
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.tools import tool


ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


@lru_cache(maxsize=1)
def _get_retriever():
    from retriever import Retriever

    return Retriever()


@tool
def knowledge_base(query: str, top_k: int = 5) -> str:
    """Search the local DHL shipping knowledge base for supporting context.

    Args:
        query: Natural-language shipping, customs, duties, document, surcharge,
            restriction, or DHL support question to search for.
        top_k: Number of relevant chunks to retrieve. Use 3-8.

    Returns:
        A JSON string containing confidence, source citations, and context.
    """
    clean_query = query.strip()
    if not clean_query:
        return json.dumps({"error": "query is required"})

    safe_top_k = min(max(int(top_k or 5), 1), 8)

    try:
        result: dict[str, Any] = _get_retriever().retrieve(clean_query, k=safe_top_k)
    except Exception as exc:
        return json.dumps(
            {
                "error": str(exc),
                "hint": "Run `python ingest.py --reset` from the repository root if the Chroma collection is missing.",
            }
        )

    compact_results = [
        {
            "source": item["source_filename"],
            "title": item["title"],
            "page_range": item["page_range"],
            "score": item["score"],
            "text": item["text"][:900],
        }
        for item in result.get("results", [])
    ]

    return json.dumps(
        {
            "query": result.get("query", clean_query),
            "found": result.get("found", False),
            "confidence": result.get("confidence", "none"),
            "citations": result.get("citations", []),
            "results": compact_results,
        }
    )
