"""
Simple retrieval-quality evaluation for the CrossBorder Copilot RAG layer.

Each case specifies expected source filename fragments and/or expected keywords.
A query is a hit when any top-k chunk matches at least one expected source or
contains at least one expected keyword.

Usage:
    python eval_retrieval.py
    python eval_retrieval.py --k 10 --verbose
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from retriever import Retriever


GOLD_CASES: list[dict[str, Any]] = [
    {
        "query": "Which items are prohibited or restricted for international shipping?",
        "expected_sources": ["Restricted-and-prohibited-items", "prohibited", "restriction"],
        "expected_keywords": ["prohibited", "restricted", "dangerous goods"],
    },
    {
        "query": "What documents are needed for customs clearance?",
        "expected_sources": ["Customs_Checklist", "customs", "commercial_invoice"],
        "expected_keywords": ["customs", "commercial invoice", "clearance"],
    },
    {
        "query": "How do I prepare a commercial invoice for an international shipment?",
        "expected_sources": ["commercial_invoice", "Customs_Checklist"],
        "expected_keywords": ["commercial invoice", "description", "value", "origin"],
    },
    {
        "query": "Can DHL ship lithium batteries or dangerous goods?",
        "expected_sources": ["battery", "dangerous", "Restricted-and-prohibited-items"],
        "expected_keywords": ["battery", "lithium", "dangerous goods"],
    },
    {
        "query": "What peak season or fuel surcharges might apply?",
        "expected_sources": ["surcharge", "Surcharges", "peak"],
        "expected_keywords": ["surcharge", "peak", "fuel"],
    },
    {
        "query": "What does duty and tax prepayment mean for DHL eCommerce shipments?",
        "expected_sources": ["duty-and-tax-prepayment"],
        "expected_keywords": ["duty", "tax", "prepayment"],
    },
    {
        "query": "What guidance exists for shipment value protection?",
        "expected_sources": ["shipment-protection", "limitation-of-liability"],
        "expected_keywords": ["shipment protection", "liability", "value"],
    },
    {
        "query": "What happens when a shipment cannot be delivered or needs return handling?",
        "expected_sources": ["undeliverable", "terms-and-conditions", "return"],
        "expected_keywords": ["undeliverable", "return", "delivery"],
    },
    {
        "query": "What are common causes of customs delays or clearance issues?",
        "expected_sources": ["customs", "clearance", "Customs-Industry-Guide"],
        "expected_keywords": ["customs", "clearance", "delay"],
    },
    {
        "query": "What DHL eCommerce services exist for parcels shipped from Canada to the US?",
        "expected_sources": ["Canada_to_U.S", "parcel-international-direct", "ca-parcel"],
        "expected_keywords": ["parcel", "international", "direct", "canada"],
    },
]


def text_matches_keywords(text: str, keywords: list[str]) -> list[str]:
    haystack = text.lower()
    return [keyword for keyword in keywords if keyword.lower() in haystack]


def source_matches(filename: str, expected_sources: list[str]) -> list[str]:
    lower = filename.lower()
    return [source for source in expected_sources if source.lower() in lower]


def evaluate_case(retriever: Retriever, case: dict[str, Any], k: int) -> dict[str, Any]:
    result = retriever.retrieve(case["query"], k=k)
    retrieved_sources = [item["source_filename"] for item in result["results"]]
    combined_text = " ".join(
        f"{item['source_filename']} {item['title']} {item['category']} {item['text']}"
        for item in result["results"]
    )

    keyword_hits = text_matches_keywords(combined_text, case["expected_keywords"])
    source_hits: list[str] = []
    for filename in retrieved_sources:
        source_hits.extend(source_matches(filename, case["expected_sources"]))

    hit = bool(keyword_hits or source_hits)
    return {
        "query": case["query"],
        "hit": hit,
        "confidence": result["confidence"],
        "top_score": result["results"][0]["score"] if result["results"] else 0.0,
        "retrieved_sources": retrieved_sources,
        "keyword_hits": sorted(set(keyword_hits)),
        "source_hits": sorted(set(source_hits)),
    }


def run_evaluation(
    k: int = 5,
    verbose: bool = False,
    export_path: str | None = None,
    chroma_dir: str | None = None,
    collection_name: str | None = None,
    embedding_model: str | None = None,
    local_files_only: bool = False,
) -> list[dict[str, Any]]:
    kwargs = {}
    if chroma_dir:
        kwargs["chroma_dir"] = chroma_dir
    if collection_name:
        kwargs["collection_name"] = collection_name
    if embedding_model:
        kwargs["embedding_model"] = embedding_model
    kwargs["local_files_only"] = local_files_only
    retriever = Retriever(**kwargs)
    results = [evaluate_case(retriever, case, k=k) for case in GOLD_CASES]

    print(f"\nRetrieval evaluation, hit@{k}")
    print("=" * 80)
    for index, result in enumerate(results, start=1):
        status = "HIT" if result["hit"] else "MISS"
        retrieved = ", ".join(result["retrieved_sources"][:3])
        print(f"{index:02d}. {status:<4} score={result['top_score']:.3f} conf={result['confidence']:<6} {result['query']}")
        print(f"    sources: {retrieved}")
        if verbose:
            print(f"    source hits : {result['source_hits']}")
            print(f"    keyword hits: {result['keyword_hits']}")

    hits = sum(1 for result in results if result["hit"])
    hit_rate = hits / len(results) if results else 0.0
    print("=" * 80)
    print(f"Overall hit@{k}: {hits}/{len(results)} = {hit_rate:.1%}")

    if export_path:
        with open(export_path, "w", encoding="utf-8") as handle:
            json.dump({"k": k, "hit_rate": hit_rate, "results": results}, handle, indent=2)
        print(f"Exported results to {export_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate local DHL retrieval quality.")
    parser.add_argument("--k", type=int, default=5, help="Top-k retrieval cutoff")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--export", default=None, help="Optional JSON export path")
    parser.add_argument("--chroma-dir", default=None)
    parser.add_argument("--collection", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    run_evaluation(
        k=args.k,
        verbose=args.verbose,
        export_path=args.export,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        local_files_only=args.local_files_only,
    )


if __name__ == "__main__":
    main()
