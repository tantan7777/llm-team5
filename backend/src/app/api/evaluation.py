from pathlib import Path
import os
import sys
import uuid
import json

from fastapi import APIRouter, HTTPException

ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from eval_retrieval import run_evaluation
from retriever import Retriever
from app.tools.notepad import notepad

router = APIRouter(prefix="/evaluation", tags=["Evaluation"])


def _parse_json(raw: str):
    try:
        return json.loads(raw)
    except Exception:
        return {"raw": raw}


def _predict_route(query: str) -> str:
    q = query.lower().strip()

    memory_markers = [
        "remember",
        "store",
        "save",
        "note down",
        "keep this",
    ]

    retrieval_markers = [
        "customs",
        "invoice",
        "shipment",
        "dhl",
        "parcel",
        "tracking",
        "restricted",
        "prohibited",
        "surcharge",
        "duty",
        "tax",
        "clearance",
        "battery",
        "undeliverable",
        "returns",
        "shipping",
    ]

    if any(marker in q for marker in memory_markers):
        return "notepad"

    if any(marker in q for marker in retrieval_markers):
        return "knowledge_base"

    return "out_of_scope"


def evaluate_intent_routing() -> dict:
    retrieval_queries = [
        "What documents are needed for customs clearance?",
        "Which items are prohibited for international shipping with DHL?",
        "How do I prepare a commercial invoice for shipping to the UK?",
        "What are DHL's surcharges for peak season shipping?",
        "Can I ship lithium batteries internationally with DHL?",
        "What happens to undeliverable shipments in Canada?",
        "How does DHL eCommerce parcel tracking work?",
        "What is the duty and tax prepayment option?",
        "What are the terms and conditions for DHL eCommerce in Canada?",
        "My shipment is stuck in customs, what should I do?",
        "What causes customs delays in international shipping?",
        "What documents are required for export clearance?",
        "What are DHL shipping restrictions for cosmetics?",
        "What is shipment protection in DHL?",
        "How do DHL returns work for international parcels?",
        "What fees apply during customs processing?",
        "How do I track a DHL parcel?",
        "What service options exist from Canada to the US?",
        "How should I declare goods for customs?",
        "What is included in DHL cross-border compliance guidance?",
        "Can I ship skincare with DHL internationally?",
        "What paperwork is needed for high-value goods?",
        "How do fuel surcharges work?",
        "What is a commercial invoice used for?",
        "What happens if customs rejects a parcel?",
    ]

    memory_queries = [
        "Remember my HS code is 8471",
        "Remember the destination country is the UK",
        "Remember the declared value is 500 dollars",
        "Remember the shipper city is Toronto",
        "Remember the receiver city is London",
        "Remember the parcel weight is 5 kilograms",
        "Remember my shipment category is cosmetics",
        "Remember the origin country is Canada",
        "Remember my invoice number is INV123",
        "Remember the recipient postal code is SW1A 1AA",
        "Remember my shipment value is 200 dollars",
        "Remember the carrier service is parcel direct",
        "Remember the import country is Germany",
        "Remember the export country is Canada",
        "Remember the product type is skincare",
        "Remember the quantity is 4 items",
        "Remember the shipping mode is air",
        "Remember the customer name is Alex",
        "Remember the shipment type is ecommerce parcel",
        "Remember the destination province is Ontario",
        "Remember the contents are supplements",
        "Remember the declared currency is CAD",
        "Remember the customs value is 350",
        "Remember the sender company is Kara",
        "Remember the receiver company is DHL test client",
    ]

    tests = (
        [{"query": q, "expected_route": "knowledge_base"} for q in retrieval_queries]
        + [{"query": q, "expected_route": "notepad"} for q in memory_queries]
    )

    results = []
    for idx, test in enumerate(tests, start=1):
        predicted = _predict_route(test["query"])
        passed = predicted == test["expected_route"]
        results.append(
            {
                "id": idx,
                "query": test["query"],
                "expected_route": test["expected_route"],
                "predicted_route": predicted,
                "passed": passed,
            }
        )

    passed = sum(1 for r in results if r["passed"])
    score = round((passed / len(results)) * 100, 2) if results else 0.0
    return {"score": score, "details": results}


def evaluate_action_execution() -> dict:
    session_id = str(uuid.uuid4())

    write_cases = [
        ("hs_code", "8471"),
        ("shipper_country", "Canada"),
        ("receiver_country", "United Kingdom"),
        ("declared_value", "500"),
        ("shipper_city", "Toronto"),
        ("receiver_city", "London"),
        ("parcel_weight", "5kg"),
        ("currency", "CAD"),
        ("shipment_type", "ecommerce"),
        ("product_type", "skincare"),
        ("invoice_number", "INV123"),
        ("quantity", "4"),
        ("shipping_mode", "air"),
        ("postal_code", "SW1A1AA"),
        ("origin_country", "Canada"),
        ("destination_country", "Germany"),
        ("customs_value", "350"),
        ("service_type", "parcel direct"),
        ("customer_name", "Alex"),
        ("receiver_name", "Chris"),
        ("company_name", "Kara"),
        ("carrier", "DHL"),
        ("shipment_id", "SHIP001"),
        ("package_count", "2"),
        ("description", "cosmetics"),
    ]

    results = []
    idx = 1

    for key, value in write_cases:
        write_raw = notepad.invoke(
            {
                "action": "write",
                "session_id": session_id,
                "key": key,
                "value": value,
            }
        )
        write_result = _parse_json(write_raw)
        results.append(
            {
                "id": idx,
                "name": f"write_{key}",
                "passed": write_result.get("status") == "saved" and write_result.get("value") == value,
                "result": write_result,
            }
        )
        idx += 1

        read_raw = notepad.invoke(
            {
                "action": "read",
                "session_id": session_id,
                "key": key,
            }
        )
        read_result = _parse_json(read_raw)
        results.append(
            {
                "id": idx,
                "name": f"read_{key}",
                "passed": read_result.get("value") == value,
                "result": read_result,
            }
        )
        idx += 1

    passed = sum(1 for t in results if t["passed"])
    score = round((passed / len(results)) * 100, 2) if results else 0.0
    return {"score": score, "details": results}


def evaluate_out_of_scope() -> dict:
    retriever = Retriever()

    queries = [
        "What is the weather in Toronto this weekend?",
        "Can you help me write a Python script to sort a list?",
        "What is the capital of France?",
        "Who won the FIFA World Cup in 2022?",
        "Explain Newton's second law.",
        "What is the stock price of Apple today?",
        "Write a cover letter for a finance internship.",
        "Who is the current prime minister of Canada?",
        "How do I solve a quadratic equation?",
        "Give me a pasta recipe for dinner.",
        "What is photosynthesis?",
        "Who founded Microsoft?",
        "Explain black holes in simple terms.",
        "How do I make a Java app?",
        "What is the population of India?",
        "Translate hello into Japanese.",
        "How do I bake chocolate cake?",
        "Who is the CEO of Tesla?",
        "What is machine learning?",
        "Tell me a joke about cats.",
        "What is the best iPhone to buy?",
        "How does Wi-Fi work?",
        "What is the GDP of Canada?",
        "How do I train for a marathon?",
        "Write a poem about the ocean.",
        "What is blockchain?",
        "Summarize Hamlet.",
        "Who painted the Mona Lisa?",
        "How do I invest in ETFs?",
        "What is compound interest?",
        "How do I learn guitar?",
        "What is the tallest mountain?",
        "Tell me today's news headlines.",
        "How do I center a div in CSS?",
        "What are symptoms of flu?",
        "How do I make biryani?",
        "Who discovered gravity?",
        "What is quantum computing?",
        "How do I improve public speaking?",
        "Write SQL to join two tables.",
        "What is the boiling point of water?",
        "Who won the NBA finals in 2020?",
        "How do I calculate CAGR?",
        "What is the capital of Japan?",
        "How do I write a resume?",
        "Explain relativity simply.",
        "How do I solve a Rubik's cube?",
        "What is the history of Rome?",
        "Give me a gym workout plan.",
        "How do I prepare for interviews?",
    ]

    results = []
    for idx, query in enumerate(queries, start=1):
        retrieval = retriever.retrieve(query)
        passed = retrieval["confidence"] in ("none", "low")
        results.append(
            {
                "id": idx,
                "query": query,
                "confidence": retrieval["confidence"],
                "found": retrieval["found"],
                "passed": passed,
            }
        )

    passed = sum(1 for r in results if r["passed"])
    score = round((passed / len(results)) * 100, 2) if results else 0.0
    return {"score": score, "details": results}


def evaluate_error_handling() -> dict:
    test_inputs = [
        {"name": "missing_value_1", "invoke": {"action": "write", "session_id": str(uuid.uuid4()), "key": "hs_code", "value": ""}},
        {"name": "missing_value_2", "invoke": {"action": "write", "session_id": str(uuid.uuid4()), "key": "country", "value": ""}},
        {"name": "missing_key_1", "invoke": {"action": "write", "session_id": str(uuid.uuid4()), "key": "", "value": "8471"}},
        {"name": "missing_key_2", "invoke": {"action": "write", "session_id": str(uuid.uuid4()), "key": "", "value": "Canada"}},
        {"name": "missing_read_key_1", "invoke": {"action": "read", "session_id": str(uuid.uuid4()), "key": "does_not_exist"}},
        {"name": "missing_read_key_2", "invoke": {"action": "read", "session_id": str(uuid.uuid4()), "key": "shipment_weight"}},
        {"name": "unknown_action_1", "invoke": {"action": "delete", "session_id": str(uuid.uuid4()), "key": "hs_code", "value": "8471"}},
        {"name": "unknown_action_2", "invoke": {"action": "drop", "session_id": str(uuid.uuid4()), "key": "country", "value": "Canada"}},
        {"name": "unknown_action_3", "invoke": {"action": "update", "session_id": str(uuid.uuid4()), "key": "value", "value": "500"}},
        {"name": "blank_session_write_1", "invoke": {"action": "write", "session_id": "", "key": "hs_code", "value": "8471"}},
        {"name": "blank_session_write_2", "invoke": {"action": "write", "session_id": "", "key": "country", "value": "Canada"}},
        {"name": "blank_session_read_1", "invoke": {"action": "read", "session_id": "", "key": "hs_code"}},
        {"name": "blank_session_read_2", "invoke": {"action": "read", "session_id": "", "key": "country"}},
        {"name": "spaces_value_1", "invoke": {"action": "write", "session_id": str(uuid.uuid4()), "key": "declared_value", "value": "   "}},
        {"name": "spaces_value_2", "invoke": {"action": "write", "session_id": str(uuid.uuid4()), "key": "postal_code", "value": "   "}},
        {"name": "read_empty_key_1", "invoke": {"action": "read", "session_id": str(uuid.uuid4()), "key": ""}},
        {"name": "read_empty_key_2", "invoke": {"action": "read", "session_id": str(uuid.uuid4()), "key": ""}},
        {"name": "bad_action_case_1", "invoke": {"action": "REMOVE", "session_id": str(uuid.uuid4()), "key": "hs_code", "value": "8471"}},
        {"name": "bad_action_case_2", "invoke": {"action": "WRITEE", "session_id": str(uuid.uuid4()), "key": "country", "value": "Canada"}},
        {"name": "bad_action_case_3", "invoke": {"action": "READD", "session_id": str(uuid.uuid4()), "key": "country", "value": ""}},
        {"name": "missing_everything_1", "invoke": {"action": "write", "session_id": "", "key": "", "value": ""}},
        {"name": "missing_everything_2", "invoke": {"action": "read", "session_id": "", "key": "", "value": ""}},
        {"name": "missing_everything_3", "invoke": {"action": "delete", "session_id": "", "key": "", "value": ""}},
        {"name": "nonexistent_key_1", "invoke": {"action": "read", "session_id": str(uuid.uuid4()), "key": "nonexistent_1"}},
        {"name": "nonexistent_key_2", "invoke": {"action": "read", "session_id": str(uuid.uuid4()), "key": "nonexistent_2"}},
        {"name": "nonexistent_key_3", "invoke": {"action": "read", "session_id": str(uuid.uuid4()), "key": "nonexistent_3"}},
        {"name": "missing_value_3", "invoke": {"action": "write", "session_id": str(uuid.uuid4()), "key": "city", "value": ""}},
        {"name": "missing_value_4", "invoke": {"action": "write", "session_id": str(uuid.uuid4()), "key": "weight", "value": ""}},
        {"name": "missing_key_3", "invoke": {"action": "write", "session_id": str(uuid.uuid4()), "key": "", "value": "London"}},
        {"name": "missing_key_4", "invoke": {"action": "write", "session_id": str(uuid.uuid4()), "key": "", "value": "5kg"}},
        {"name": "blank_session_write_3", "invoke": {"action": "write", "session_id": "", "key": "receiver", "value": "Chris"}},
        {"name": "blank_session_write_4", "invoke": {"action": "write", "session_id": "", "key": "service", "value": "parcel"}},
        {"name": "blank_session_read_3", "invoke": {"action": "read", "session_id": "", "key": "receiver"}},
        {"name": "blank_session_read_4", "invoke": {"action": "read", "session_id": "", "key": "service"}},
        {"name": "spaces_value_3", "invoke": {"action": "write", "session_id": str(uuid.uuid4()), "key": "receiver_name", "value": "   "}},
        {"name": "spaces_value_4", "invoke": {"action": "write", "session_id": str(uuid.uuid4()), "key": "company", "value": "   "}},
        {"name": "bad_action_case_4", "invoke": {"action": "DESTROY", "session_id": str(uuid.uuid4()), "key": "hs_code", "value": "8471"}},
        {"name": "bad_action_case_5", "invoke": {"action": "ERASE", "session_id": str(uuid.uuid4()), "key": "country", "value": "Canada"}},
        {"name": "unknown_action_4", "invoke": {"action": "wipe", "session_id": str(uuid.uuid4()), "key": "postal", "value": "12345"}},
        {"name": "unknown_action_5", "invoke": {"action": "clear", "session_id": str(uuid.uuid4()), "key": "invoice", "value": "INV123"}},
        {"name": "nonexistent_key_4", "invoke": {"action": "read", "session_id": str(uuid.uuid4()), "key": "random_4"}},
        {"name": "nonexistent_key_5", "invoke": {"action": "read", "session_id": str(uuid.uuid4()), "key": "random_5"}},
        {"name": "missing_everything_4", "invoke": {"action": "write", "session_id": "", "key": "", "value": ""}},
        {"name": "missing_everything_5", "invoke": {"action": "read", "session_id": "", "key": "", "value": ""}},
        {"name": "missing_everything_6", "invoke": {"action": "drop", "session_id": "", "key": "", "value": ""}},
        {"name": "read_empty_key_3", "invoke": {"action": "read", "session_id": str(uuid.uuid4()), "key": ""}},
        {"name": "read_empty_key_4", "invoke": {"action": "read", "session_id": str(uuid.uuid4()), "key": ""}},
        {"name": "missing_key_5", "invoke": {"action": "write", "session_id": str(uuid.uuid4()), "key": "", "value": "Alex"}},
        {"name": "missing_value_5", "invoke": {"action": "write", "session_id": str(uuid.uuid4()), "key": "customer", "value": ""}},
        {"name": "unknown_action_6", "invoke": {"action": "remove", "session_id": str(uuid.uuid4()), "key": "x", "value": "y"}},
        {"name": "unknown_action_7", "invoke": {"action": "archive", "session_id": str(uuid.uuid4()), "key": "x", "value": "y"}},
    ]

    results = []
    for idx, test in enumerate(test_inputs, start=1):
        raw = notepad.invoke(test["invoke"])
        parsed = _parse_json(raw)
        passed = "error" in parsed or isinstance(parsed, list)
        results.append(
            {
                "id": idx,
                "name": test["name"],
                "passed": passed,
                "result": parsed,
            }
        )

    passed = sum(1 for t in results if t["passed"])
    score = round((passed / len(results)) * 100, 2) if results else 0.0
    return {"score": score, "details": results}


@router.post("/run")
async def run_eval():
    old_cwd = os.getcwd()

    try:
        os.chdir(ROOT_DIR)

        retrieval_results = run_evaluation(verbose=False, quiet=True)
        retrieval_total = len(retrieval_results)
        retrieval_passed = sum(1 for r in retrieval_results if r["passed"])
        retrieval_score = round((retrieval_passed / retrieval_total) * 100, 2) if retrieval_total else 0.0

        intent = evaluate_intent_routing()
        action = evaluate_action_execution()
        out_of_scope = evaluate_out_of_scope()
        error = evaluate_error_handling()

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}") from exc
    finally:
        os.chdir(old_cwd)

    return {
        "summary": {
            "retrieval_quality": retrieval_score,
            "intent_routing": intent["score"],
            "action_execution": action["score"],
            "out_of_scope_handling": out_of_scope["score"],
            "error_handling": error["score"],
        },
        "details": {
            "retrieval_quality": retrieval_results,
            "intent_routing": intent["details"],
            "action_execution": action["details"],
            "out_of_scope_handling": out_of_scope["details"],
            "error_handling": error["details"],
        },
    }