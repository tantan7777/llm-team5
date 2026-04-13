"""
Simple retrieval-quality evaluation for the CrossBorder Copilot RAG layer.

Each case specifies expected source filename fragments and/or expected keywords.
A query passes when any top-k chunk matches at least one expected source or
contains at least one expected keyword.

Usage:
    python eval_retrieval.py
    python eval_retrieval.py --k 10 --verbose
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from retriever import DEFAULT_TOP_K, Retriever


GOLD_CASES: list[dict[str, Any]] = [
    {
        "id": 1,
        "category": "restricted_goods",
        "query": "Which items are prohibited or restricted for international shipping?",
        "expected_sources": ["Restricted-and-prohibited-items", "prohibited", "restriction"],
        "expected_keywords": ["prohibited", "restricted", "dangerous goods"],
    },
    {
        "id": 2,
        "category": "customs",
        "query": "What documents are needed for customs clearance?",
        "expected_sources": ["Customs_Checklist", "customs", "commercial_invoice"],
        "expected_keywords": ["customs", "commercial invoice", "clearance"],
    },
    {
        "id": 3,
        "category": "customs",
        "query": "How do I prepare a commercial invoice for an international shipment?",
        "expected_sources": ["commercial_invoice", "Customs_Checklist"],
        "expected_keywords": ["commercial invoice", "description", "value", "origin"],
    },
    {
        "id": 4,
        "category": "restricted_goods",
        "query": "Can DHL ship lithium batteries or dangerous goods?",
        "expected_sources": ["battery", "dangerous", "Restricted-and-prohibited-items"],
        "expected_keywords": ["battery", "lithium", "dangerous goods"],
    },
    {
        "id": 5,
        "category": "surcharges",
        "query": "What peak season or fuel surcharges might apply?",
        "expected_sources": ["surcharge", "Surcharges", "peak"],
        "expected_keywords": ["surcharge", "peak", "fuel"],
    },
    {
        "id": 6,
        "category": "duties_taxes",
        "query": "What does duty and tax prepayment mean for DHL eCommerce shipments?",
        "expected_sources": ["duty-and-tax-prepayment"],
        "expected_keywords": ["duty", "tax", "prepayment"],
    },
    {
        "id": 7,
        "category": "claims_liability",
        "query": "What guidance exists for shipment value protection?",
        "expected_sources": ["shipment-protection", "limitation-of-liability"],
        "expected_keywords": ["shipment protection", "liability", "value"],
    },
    {
        "id": 8,
        "category": "delivery",
        "query": "What happens when a shipment cannot be delivered or needs return handling?",
        "expected_sources": ["undeliverable", "terms-and-conditions", "return"],
        "expected_keywords": ["undeliverable", "return", "delivery"],
    },
    {
        "id": 9,
        "category": "customs",
        "query": "What are common causes of customs delays or clearance issues?",
        "expected_sources": ["customs", "clearance", "Customs-Industry-Guide"],
        "expected_keywords": ["customs", "clearance", "delay"],
    },
    {
        "id": 10,
        "category": "services",
        "query": "What DHL eCommerce services exist for parcels shipped from Canada to the US?",
        "expected_sources": ["Canada_to_U.S", "parcel-international-direct", "ca-parcel"],
        "expected_keywords": ["parcel", "international", "direct", "canada"],
    },
]


def _matches_any(haystack: str, needles: list[str]) -> list[str]:
    lower = haystack.lower()
    return [needle for needle in needles if needle.lower() in lower]


def evaluate_case(retriever: Retriever, case: dict[str, Any], k: int = DEFAULT_TOP_K, verbose: bool = False) -> dict[str, Any]:
    """Run and score one retrieval case."""
    result = retriever.retrieve(case["query"], k=k)
    results = result.get("results", [])

    source_hits: list[str] = []
    keyword_hits: list[str] = []

    for item in results:
        source_text = " ".join(
            [
                str(item.get("source_filename", "")),
                str(item.get("title", "")),
                str(item.get("source_uri", "")),
            ]
        )
        source_hits.extend(_matches_any(source_text, case["expected_sources"]))
        keyword_hits.extend(_matches_any(str(item.get("text", "")), case["expected_keywords"]))

    source_hits = sorted(set(source_hits))
    keyword_hits = sorted(set(keyword_hits))
    scores = [float(item.get("score", 0.0)) for item in results]

    eval_result = {
        "id": case["id"],
        "query": case["query"],
        "category": case["category"],
        "passed": bool(result.get("found")) and bool(source_hits or keyword_hits),
        "confidence": result.get("confidence", "none"),
        "found": bool(result.get("found")),
        "num_results": len(results),
        "top_score": round(scores[0], 4) if scores else 0.0,
        "avg_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "source_match": bool(source_hits),
        "keyword_match": bool(keyword_hits),
        "source_hits": source_hits,
        "keyword_hits": keyword_hits,
    }

    if verbose:
        eval_result["citations"] = result.get("citations", [])

    return eval_result


def run_evaluation(k: int = DEFAULT_TOP_K, verbose: bool = False, export_path: str | None = None, quiet: bool = False) -> list[dict[str, Any]]:
    """Run all retrieval cases and optionally print/export a summary."""
    retriever = Retriever()
    results = [evaluate_case(retriever, case, k=k, verbose=verbose) for case in GOLD_CASES]

    total = len(results)
    passed = sum(1 for item in results if item["passed"])
    failed_cases = [item for item in results if not item["passed"]]

    if not quiet:
        print("=" * 70)
        print("  CrossBorder Copilot - Retrieval Evaluation")
        print("=" * 70)
        for item in results:
            status = "PASS" if item["passed"] else "FAIL"
            print(
                f"  [{status}] #{item['id']:02d} "
                f"conf={item['confidence']:<6} "
                f"score={item['top_score']:.3f} "
                f"source={'yes' if item['source_match'] else 'no '} "
                f"kw={'yes' if item['keyword_match'] else 'no '} "
                f"| {item['query'][:55]}"
            )
            if verbose:
                for citation in item.get("citations", [])[:3]:
                    print(f"        - {citation['title'][:60]} ({citation['source_filename']})")

        print("\n" + "=" * 70)
        print("  SUMMARY")
        print("=" * 70)
        print(f"  Total: {total} | Passed: {passed} | Failed: {total - passed}")
        print(f"  Overall accuracy: {(passed / total * 100):.1f}%" if total else "  Overall accuracy: 0.0%")

        categories = Counter(item["category"] for item in results)
        print("\n  By category:")
        for category in sorted(categories):
            category_results = [item for item in results if item["category"] == category]
            category_passed = sum(1 for item in category_results if item["passed"])
            print(f"    {category:<20} {category_passed}/{len(category_results)} passed")

        if failed_cases:
            print("\n  Failure analysis:")
            for item in failed_cases:
                print(f"    #{item['id']:02d} [{item['category']}] conf={item['confidence']}")
                print(f"        query: {item['query'][:80]}")
                print(f"        source hits: {item['source_hits']}")
                print(f"        keyword hits: {item['keyword_hits']}")
                print(f"        top score: {item['top_score']:.3f}")

        print("=" * 70)

    if export_path:
        export_data = {
            "summary": {
                "total": total,
                "passed": passed,
                "failed": total - passed,
                "accuracy": round(passed / total * 100, 1) if total else 0.0,
            },
            "results": results,
        }
        Path(export_path).write_text(json.dumps(export_data, indent=2), encoding="utf-8")
        if not quiet:
            print(f"Results exported to {export_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the local DHL retrieval index.")
    parser.add_argument("-k", "--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--export", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    run_evaluation(k=args.top_k, verbose=args.verbose, export_path=args.export)


if __name__ == "__main__":
    main()
