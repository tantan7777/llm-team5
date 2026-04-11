"""
eval_retrieval.py  —  CrossBorder Copilot
Evaluates RAG retrieval quality across 15 representative test cases.

Measures:
  - Retrieval accuracy: did the top-5 results contain a relevant chunk?
  - Confidence distribution: how often do we get high/medium/low/none
  - Citation quality: are the cited doc_types correct for the query?
  - Failure analysis: which queries fail and why?

Usage:
    python eval_retrieval.py                  # run all test cases
    python eval_retrieval.py --verbose        # show full context for each case
    python eval_retrieval.py --export results.json  # save results to file
"""

from __future__ import annotations
import sys
import json
import logging
import argparse
from retriever import Retriever

logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")


# ── Test Cases ────────────────────────────────────────────────────────────────
# Each test case has:
#   query            : the user's question
#   expected_types   : which doc_types SHOULD appear in results (at least one match = pass)
#   expected_keywords: keywords that SHOULD appear in retrieved text (at least one match = pass)
#   category         : what this test is evaluating

TEST_CASES = [
    # ── Successful knowledge retrieval (7 cases) ──────────────────────────────
    {
        "id": 1,
        "query": "What documents are needed for customs clearance?",
        "expected_types": ["customs"],
        "expected_keywords": ["commercial invoice", "customs", "clearance", "declaration"],
        "category": "knowledge_retrieval",
    },
    {
        "id": 2,
        "query": "Which items are prohibited for international shipping with DHL?",
        "expected_types": ["policy"],
        "expected_keywords": ["prohibited", "restricted", "dangerous", "items"],
        "category": "knowledge_retrieval",
    },
    {
        "id": 3,
        "query": "How do I prepare a commercial invoice for shipping to the UK?",
        "expected_types": ["customs", "guide"],
        "expected_keywords": ["commercial invoice", "value", "description", "origin"],
        "category": "knowledge_retrieval",
    },
    {
        "id": 4,
        "query": "What are DHL's surcharges for peak season shipping?",
        "expected_types": ["surcharge"],
        "expected_keywords": ["surcharge", "peak", "fee"],
        "category": "knowledge_retrieval",
    },
    {
        "id": 5,
        "query": "Can I ship lithium batteries internationally with DHL?",
        "expected_types": ["policy", "guide"],
        "expected_keywords": ["battery", "lithium", "dangerous goods"],
        "category": "knowledge_retrieval",
    },
    {
        "id": 6,
        "query": "What happens to undeliverable shipments in Canada?",
        "expected_types": ["policy", "guide", "faq"],
        "expected_keywords": ["undeliverable", "return", "shipment"],
        "category": "knowledge_retrieval",
    },
    {
        "id": 7,
        "query": "How does DHL eCommerce parcel tracking work?",
        "expected_types": ["faq", "guide"],
        "expected_keywords": ["tracking", "status", "shipment"],
        "category": "knowledge_retrieval",
    },

    # ── Doc-type filtered retrieval (2 cases) ─────────────────────────────────
    {
        "id": 8,
        "query": "What is the duty and tax prepayment option?",
        "expected_types": ["surcharge", "faq", "guide"],
        "expected_keywords": ["duty", "tax", "prepayment", "DDP"],
        "category": "filtered_retrieval",
    },
    {
        "id": 9,
        "query": "What are the terms and conditions for DHL eCommerce in Canada?",
        "expected_types": ["guide", "policy"],
        "expected_keywords": ["terms", "conditions", "Canada", "liability"],
        "category": "filtered_retrieval",
    },

    # ── Edge cases and harder queries (3 cases) ───────────────────────────────
    {
        "id": 10,
        "query": "I want to ship skincare products from Canada to the UK, what paperwork do I need?",
        "expected_types": ["customs", "guide"],
        "expected_keywords": ["customs", "invoice", "clearance", "documents"],
        "category": "complex_query",
    },
    {
        "id": 11,
        "query": "My shipment is stuck in customs, what should I do?",
        "expected_types": ["customs", "faq", "guide"],
        "expected_keywords": ["customs", "clearance", "delay", "hold"],
        "category": "complex_query",
    },
    {
        "id": 12,
        "query": "How much does it cost to ship a 5kg package from China to the US?",
        "expected_types": ["guide", "surcharge"],
        "expected_keywords": ["price", "rate", "cost", "weight", "parcel"],
        "category": "complex_query",
    },

    # ── Out-of-scope / no-answer queries (3 cases) ────────────────────────────
    {
        "id": 13,
        "query": "What is the weather forecast in Toronto this weekend?",
        "expected_types": [],
        "expected_keywords": [],
        "category": "out_of_scope",
    },
    {
        "id": 14,
        "query": "Can you help me write a Python script to sort a list?",
        "expected_types": [],
        "expected_keywords": [],
        "category": "out_of_scope",
    },
    {
        "id": 15,
        "query": "What is the capital of France?",
        "expected_types": [],
        "expected_keywords": [],
        "category": "out_of_scope",
    },
]


# ── Evaluation Logic ──────────────────────────────────────────────────────────

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="Show citations for each case")
    parser.add_argument("--export", type=str, default=None, help="Export results to JSON file")
    args = parser.parse_args()
    run_evaluation(verbose=args.verbose, export_path=args.export)
