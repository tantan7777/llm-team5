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


def evaluate_case(retriever: Retriever, case: dict, verbose: bool = False) -> dict:
    """Run a single test case and score it."""
    result = retriever.retrieve(case["query"])

    # Check doc_type match
    retrieved_types = {r["doc_type"] for r in result["results"]}
    type_match = bool(set(case["expected_types"]) & retrieved_types) if case["expected_types"] else True

    # Check keyword match
    all_text = " ".join(r["text"].lower() for r in result["results"])
    keyword_hits = [kw for kw in case["expected_keywords"] if kw.lower() in all_text]
    keyword_match = len(keyword_hits) > 0 if case["expected_keywords"] else True

    # For out-of-scope queries, success = low confidence or no results
    if case["category"] == "out_of_scope":
        is_pass = result["confidence"] in ("none", "low")
    else:
        is_pass = type_match and keyword_match and result["found"]

    scores = [r["score"] for r in result["results"]]

    eval_result = {
        "id":              case["id"],
        "query":           case["query"],
        "category":        case["category"],
        "passed":          is_pass,
        "confidence":      result["confidence"],
        "found":           result["found"],
        "num_results":     len(result["results"]),
        "top_score":       scores[0] if scores else 0.0,
        "avg_score":       round(sum(scores) / len(scores), 4) if scores else 0.0,
        "type_match":      type_match,
        "keyword_match":   keyword_match,
        "keyword_hits":    keyword_hits,
        "retrieved_types": list(retrieved_types),
        "expected_types":  case["expected_types"],
    }

    if verbose:
        eval_result["citations"] = result["citations"]

    return eval_result


def run_evaluation(verbose: bool = False, export_path: str | None = None):
    """Run all test cases and print a summary report."""
    retriever = Retriever()
    results = []

    print("=" * 70)
    print("  CrossBorder Copilot — Retrieval Evaluation")
    print("=" * 70)

    for case in TEST_CASES:
        eval_result = evaluate_case(retriever, case, verbose)
        results.append(eval_result)

        status = "PASS" if eval_result["passed"] else "FAIL"
        icon = "✓" if eval_result["passed"] else "✗"
        print(
            f"  {icon} [{status}]  #{eval_result['id']:02d}  "
            f"conf={eval_result['confidence']:<6}  "
            f"score={eval_result['top_score']:.3f}  "
            f"type={'✓' if eval_result['type_match'] else '✗'}  "
            f"kw={'✓' if eval_result['keyword_match'] else '✗'}  "
            f"| {eval_result['query'][:55]}"
        )

        if verbose and eval_result["citations"]:
            for c in eval_result["citations"][:3]:
                print(f"        → [{c['doc_type']}] {c['title'][:50]}")

    # ── Summary ───────────────────────────────────────────────────────────────
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed_cases = [r for r in results if not r["passed"]]

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Total: {total}  |  Passed: {passed}  |  Failed: {total - passed}")
    print(f"  Overall accuracy: {passed / total * 100:.1f}%")

    # By category
    from collections import Counter
    categories = Counter(r["category"] for r in results)
    print("\n  By category:")
    for cat in sorted(categories.keys()):
        cat_results = [r for r in results if r["category"] == cat]
        cat_passed = sum(1 for r in cat_results if r["passed"])
        print(f"    {cat:<22} {cat_passed}/{len(cat_results)} passed")

    # Confidence distribution
    conf_dist = Counter(r["confidence"] for r in results)
    print("\n  Confidence distribution:")
    for level in ["high", "medium", "low", "none"]:
        print(f"    {level:<8} {conf_dist.get(level, 0)}")

    # Average scores
    retrieval_results = [r for r in results if r["category"] != "out_of_scope"]
    if retrieval_results:
        avg_top = sum(r["top_score"] for r in retrieval_results) / len(retrieval_results)
        print(f"\n  Avg top-1 score (in-scope queries): {avg_top:.3f}")

    # Failure analysis
    if failed_cases:
        print("\n  FAILURE ANALYSIS:")
        for r in failed_cases:
            print(f"    #{r['id']:02d} [{r['category']}] conf={r['confidence']}")
            print(f"        Query: {r['query'][:60]}")
            print(f"        Expected types: {r['expected_types']}")
            print(f"        Retrieved types: {r['retrieved_types']}")
            print(f"        Keyword hits: {r['keyword_hits']}")
            print(f"        Top score: {r['top_score']:.3f}")
            print()

    print("=" * 70)
    return results

    # Export
    if export_path:
        export_data = {
            "summary": {
                "total": total,
                "passed": passed,
                "failed": total - passed,
                "accuracy": round(passed / total * 100, 1),
            },
            "results": results,
        }
        with open(export_path, "w") as f:
            json.dump(export_data, f, indent=2, default=str)
        print(f"\n  Results exported to {export_path}")
        return results

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
