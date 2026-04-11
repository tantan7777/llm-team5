"""
shipment_inquiry_mcp.py — Mock Shipment Inquiry Agent MCP server.

Dependencies: fastmcp, langchain-openai
Run with: python shipment_inquiry_mcp.py
"""

import json
import logging
import sqlite3
from pathlib import Path

from fastmcp import FastMCP
from langchain_openai import ChatOpenAI

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("shipment-inquiry")

# ── Database setup ────────────────────────────────────────────────────────────
DB_FILE = Path(__file__).parent / "shipments.db"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    logger.info("Initialising database at %s", DB_FILE)
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id          TEXT PRIMARY KEY,
                destination       TEXT NOT NULL,
                status            TEXT NOT NULL,
                item              TEXT NOT NULL,
                exception_details TEXT,
                internal_notes    TEXT NOT NULL DEFAULT '[]'
            )
        """)
        count = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        if count == 0:
            logger.info("Empty database — seeding with mock orders...")
            seed = [
                ("ORD-001", "Canada",    "Held at Customs", "Lithium Battery Pack",       "Shipment flagged due to missing UN38.3 certification for lithium batteries."),
                ("ORD-002", "Germany",   "In Transit",      "Industrial Machinery Parts",  None),
                ("ORD-003", "Australia", "Held at Customs", "Wooden Furniture",            "Biosecurity hold — wood packaging material requires ISPM-15 phytosanitary certificate."),
                ("ORD-004", "Japan",     "Delivered",       "Consumer Electronics",        None),
                ("ORD-005", "Brazil",    "Held at Customs", "Pharmaceutical Samples",      "ANVISA import permit missing. Controlled substance declaration required."),
            ]
            conn.executemany(
                "INSERT INTO orders (order_id, destination, status, item, exception_details) VALUES (?,?,?,?,?)",
                seed,
            )
            conn.commit()
            logger.info("Seeded %d orders into database.", len(seed))
        else:
            logger.info("Database already contains %d order(s) — skipping seed.", count)


# ── LLM ───────────────────────────────────────────────────────────────────────
logger.info("Initialising LLM client...")
llm = ChatOpenAI(
    base_url="https://rsm-8430-finalproject.bjlkeng.io/v1",
    api_key="1012837405",
    temperature=0.0,
)
logger.info("LLM client ready.")

# ── MCP server ────────────────────────────────────────────────────────────────
mcp = FastMCP("shipment-inquiry")

BLACKLISTED_KEYWORDS = {"REFUND", "CANCEL", "DELETE", "SHIPMENT_STATUS_CHANGE"}


def _get_order(order_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM orders WHERE order_id = ?", (order_id,)
        ).fetchone()
    if not row:
        return None
    order = dict(row)
    order["internal_notes"] = json.loads(order["internal_notes"])
    return order


@mcp.tool()
def list_all_orders() -> list[dict]:
    """Return a summary of all orders (order_id, destination, status, item)."""
    logger.info("TOOL CALL  list_all_orders()")
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT order_id, destination, status, item FROM orders"
        ).fetchall()
    result = [dict(r) for r in rows]
    logger.info("TOOL RESULT  Returning %d order(s):", len(result))
    for o in result:
        logger.info("  %-10s  %-12s  %-20s  %s", o["order_id"], o["status"], o["destination"], o["item"])
    return result


@mcp.tool()
def get_order_status(order_id: str) -> dict:
    """Return full details for a specific order.

    Args:
        order_id: The order ID to look up, e.g. 'ORD-001'.
    """
    logger.info("TOOL CALL  get_order_status(order_id='%s')", order_id)
    order = _get_order(order_id)
    if not order:
        logger.warning("TOOL RESULT  Order '%s' not found.", order_id)
        return {"error": f"Order '{order_id}' not found."}
    logger.info("TOOL RESULT  %s | %s | %s | exception=%s | notes=%s",
        order["order_id"], order["status"], order["item"],
        order["exception_details"] or "None",
        order["internal_notes"],
    )
    return order


@mcp.tool()
def diagnose_order(order_id: str) -> str:
    """Use the LLM to explain why an order is held at customs and suggest remediation.

    Args:
        order_id: The order ID to diagnose.
    """
    logger.info("TOOL CALL  diagnose_order(order_id='%s')", order_id)
    order = _get_order(order_id)

    if not order:
        logger.warning("TOOL RESULT  Order '%s' not found.", order_id)
        return f"Order '{order_id}' not found."

    if order["status"] != "Held at Customs":
        logger.info("TOOL RESULT  Order '%s' is not held at customs (status: %s) — skipping diagnosis.", order_id, order["status"])
        return f"Order '{order_id}' is not held at customs (current status: {order['status']})."

    if not order["exception_details"]:
        logger.warning("TOOL RESULT  Order '%s' has no exception details — cannot diagnose.", order_id)
        return "No exception details available to diagnose."

    logger.info("Sending exception details to LLM for diagnosis...")
    logger.info("  Item       : %s", order["item"])
    logger.info("  Destination: %s", order["destination"])
    logger.info("  Exception  : %s", order["exception_details"])

    prompt = (
        f"A shipment of '{order['item']}' destined for {order['destination']} "
        f"is held at customs. Exception details: {order['exception_details']}\n\n"
        "In 2-3 concise bullet points, explain the likely cause and what the shipper "
        "should do to resolve this customs hold."
    )
    response = llm.invoke(prompt).content
    logger.info("TOOL RESULT  LLM diagnosis for '%s':\n%s", order_id, response)
    return response


@mcp.tool()
def update_order_notes(order_id: str, note: str) -> dict:
    """Append a note to an order's internal_notes list.

    Guard rails — the update will be rejected if:
    - The order does not exist.
    - The note is shorter than 10 or longer than 200 characters.
    - The note contains blacklisted keywords: REFUND, CANCEL, DELETE, SHIPMENT_STATUS_CHANGE.

    Args:
        order_id: The order to update.
        note:     The note to append.
    """
    logger.info("TOOL CALL  update_order_notes(order_id='%s', note='%s')", order_id, note)

    # Guard 1 — order must exist
    order = _get_order(order_id)
    if not order:
        logger.warning("GUARD FAIL  Order '%s' does not exist.", order_id)
        return {"error": f"Order '{order_id}' not found."}

    # Guard 2 — note length
    if len(note) < 10:
        logger.warning("GUARD FAIL  Note too short (%d chars) for order '%s'.", len(note), order_id)
        return {"error": "Note is too short (minimum 10 characters)."}
    if len(note) > 200:
        logger.warning("GUARD FAIL  Note too long (%d chars) for order '%s'.", len(note), order_id)
        return {"error": "Note is too long (maximum 200 characters)."}

    # Guard 3 — blacklisted keywords
    triggered = [kw for kw in BLACKLISTED_KEYWORDS if kw in note.upper()]
    if triggered:
        logger.warning("GUARD FAIL  Blacklisted keyword(s) %s found in note for order '%s'.", triggered, order_id)
        return {"error": f"Note contains blacklisted keyword(s): {triggered}. Update rejected."}

    notes = order["internal_notes"]
    notes.append(note)

    with get_conn() as conn:
        conn.execute(
            "UPDATE orders SET internal_notes = ? WHERE order_id = ?",
            (json.dumps(notes), order_id),
        )
        conn.commit()

    logger.info("TOOL RESULT  Note appended to order '%s'. Total notes: %d.", order_id, len(notes))
    return {"status": "success", "order_id": order_id, "notes": notes}


if __name__ == "__main__":
    init_db()
    logger.info("Starting Shipment Inquiry MCP server on port 8002 (SSE)...")
    mcp.run(transport="sse", host="0.0.0.0", port=8002)