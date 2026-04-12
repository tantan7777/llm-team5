"""
mcp/ticket_agent_mcp.py — Ticket Agent MCP Server

Exposes a single tool: create_escalation_ticket
Runs via SSE on host 0.0.0.0, port 8004.

Run from the mcp/ directory:
    python ticket_agent_mcp.py
"""

import os
import sys
import logging
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("LLM_BASE_URL")
API_KEY  = os.getenv("LLM_API_KEY")

# ── Logging setup ─────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RED    = "\033[38;5;196m"
YELLOW = "\033[38;5;220m"
GREEN  = "\033[38;5;82m"
CYAN   = "\033[38;5;51m"
BLUE   = "\033[38;5;39m"
PURPLE = "\033[38;5;141m"
GRAY   = "\033[38;5;245m"
WHITE  = "\033[38;5;255m"
ORANGE = "\033[38;5;208m"


class PrettyFormatter(logging.Formatter):
    LEVEL_STYLES = {
        "DEBUG":    f"{DIM}{GRAY}",
        "INFO":     f"{CYAN}",
        "WARNING":  f"{YELLOW}",
        "ERROR":    f"{RED}{BOLD}",
        "CRITICAL": f"{RED}{BOLD}",
    }

    def format(self, record):
        style = self.LEVEL_STYLES.get(record.levelname, "")
        ts    = self.formatTime(record, "%H:%M:%S")
        level = f"{style}{record.levelname:<8}{RESET}"
        msg   = record.getMessage()
        return f"{DIM}{GRAY}{ts}{RESET}  {level}  {CYAN}[TICKET-AGENT]{RESET}  {msg}"


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(PrettyFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger("ticket-agent")


def _banner():
    print(f"""
{ORANGE}{BOLD}╔══════════════════════════════════════════════════════╗
║         CrossBorder Copilot — Ticket Agent           ║
║                   MCP Server v1.0                    ║
╚══════════════════════════════════════════════════════╝{RESET}
""")


def _section(title: str):
    bar = "─" * 54
    logger.info(f"{BOLD}{WHITE}{bar}{RESET}")
    logger.info(f"{BOLD}{WHITE}  {title}{RESET}")
    logger.info(f"{BOLD}{WHITE}{bar}{RESET}")


def _kv(key: str, value: str, color: str = CYAN):
    logger.info(f"  {DIM}{GRAY}{key:<24}{RESET}{color}{value}{RESET}")


# ── Boot ──────────────────────────────────────────────────────────────────────

_banner()
_section("STARTUP")

from sqlmodel import Field, Session, SQLModel, create_engine
from fastmcp import FastMCP
from langchain_openai import ChatOpenAI

logger.info(f"{GREEN}✔{RESET}  Core imports OK")

# ── Database setup ────────────────────────────────────────────────────────────

_mcp_dir = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(_mcp_dir, "tickets.db")
DB_URL   = f"sqlite:///{DB_PATH}"

_kv("Database path", DB_PATH)
_kv("Database URL",  DB_URL)


class SupportTicket(SQLModel, table=True):
    id:               Optional[int] = Field(default=None, primary_key=True)
    customer_name:    str
    issue_summary:    str
    priority:         str
    status:           str  = Field(default="Open")
    created_at:       str  = Field(
        default_factory=lambda: datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    )


engine = create_engine(DB_URL, echo=False)
SQLModel.metadata.create_all(engine)
logger.info(f"{GREEN}✔{RESET}  SQLite database initialised  →  tickets.db")

# ── LLM setup ─────────────────────────────────────────────────────────────────

_kv("LLM base URL", BASE_URL or "(not set)", YELLOW if not BASE_URL else CYAN)
_kv("LLM API key",  "***" + (API_KEY[-4:] if API_KEY else "(not set)"))

_llm_ready = bool(BASE_URL and API_KEY)
if _llm_ready:
    llm = ChatOpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        temperature=0.0,
    )
    logger.info(f"{GREEN}✔{RESET}  LLM client initialised")
else:
    logger.warning(
        f"{YELLOW}⚠{RESET}  LLM_BASE_URL or LLM_API_KEY not set — "
        f"falling back to truncated issue summary for SMS"
    )

SMS_SUMMARY_SYSTEM_PROMPT = """You summarize shipment support issues for SMS alerts.
Return exactly one concise line.
Do not use bullets, labels, quotes, or extra commentary.
Keep the summary under 12 words if possible.
Extract only the core issue keywords so the alert is easy to scan.
"""

def summarize_issue_for_sms(issue_summary: str) -> str:
    """Return a short one-line summary suitable for a Twilio SMS body."""
    if not _llm_ready:
        fallback = issue_summary.strip().replace("\n", " ")
        return fallback[:60]

    prompt = f"""Summarize this support escalation issue into a short SMS-safe issue line:

Issue:
{issue_summary}

Return only the concise issue line."""
    try:
        response = llm.invoke([
            {"role": "system", "content": SMS_SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])
        concise = str(response.content).strip().replace("\n", " ")
        return concise[:60] if concise else issue_summary.strip().replace("\n", " ")[:60]
    except Exception as exc:
        logger.warning(
            f"{YELLOW}⚠{RESET}  LLM issue summarization failed — using fallback. Error: {exc}"
        )
        return issue_summary.strip().replace("\n", " ")[:60]

# ── Twilio config ─────────────────────────────────────────────────────────────

TWILIO_SID  = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM = os.getenv("TWILIO_FROM_NUMBER")
MOBILE      = os.getenv("MOBILE")

_kv("Twilio SID",   ("***" + TWILIO_SID[-4:])  if TWILIO_SID  else f"{YELLOW}(not set){RESET}")
_kv("Twilio FROM",  TWILIO_FROM                 if TWILIO_FROM else f"{YELLOW}(not set){RESET}")
_kv("Alert mobile", MOBILE                      if MOBILE      else f"{YELLOW}(not set){RESET}")

_twilio_ready = all([TWILIO_SID, TWILIO_AUTH, TWILIO_FROM, MOBILE])
if _twilio_ready:
    logger.info(f"{GREEN}✔{RESET}  Twilio credentials loaded — SMS alerts enabled")
else:
    logger.warning(
        f"{YELLOW}⚠{RESET}  One or more Twilio env vars missing — "
        f"SMS alerts will be skipped"
    )

# ── MCP server ────────────────────────────────────────────────────────────────

mcp = FastMCP("ticket-service")
logger.info(f"{GREEN}✔{RESET}  FastMCP instance created  →  name=ticket-service")


# ── Tool ──────────────────────────────────────────────────────────────────────

@mcp.tool()
def create_escalation_ticket(
    customer_name:  str,
    issue_summary:  str,
    priority:       str,
) -> str:
    """Create a support escalation ticket for high-value shipment errors or explicit
    human-support requests. Persists to SQLite and sends an SMS alert to the on-call
    Production Support Analyst via Twilio.

    Args:
        customer_name:  Full name of the affected customer.
        issue_summary:  Clear description of the issue requiring escalation.
        priority:       Ticket priority — one of: Low, Medium, High, Critical.

    Returns:
        A confirmation message with the assigned Ticket ID for the agent to relay to the user.
    """
    _section("TOOL INVOKED  →  create_escalation_ticket")
    _kv("Customer",      customer_name, WHITE)
    _kv("Priority",      priority,
        RED if priority.lower() == "critical" else
        ORANGE if priority.lower() == "high" else
        YELLOW if priority.lower() == "medium" else CYAN)
    _kv("Issue summary", issue_summary[:60] + ("…" if len(issue_summary) > 60 else ""), WHITE)

    logger.info(
        f"\n  {ORANGE}Escalation requested for user: {BOLD}{customer_name}{RESET}\n"
    )

    # ── Step 1: Persist to SQLite ─────────────────────────────────────────────
    logger.info(f"{BLUE}[1/4]{RESET}  Persisting ticket to SQLite…")

    ticket = SupportTicket(
        customer_name=customer_name,
        issue_summary=issue_summary,
        priority=priority,
    )

    with Session(engine) as session:
        session.add(ticket)
        session.commit()
        session.refresh(ticket)
        ticket_id = ticket.id

    logger.info(
        f"{GREEN}✔{RESET}  Writing Ticket ID {BOLD}{ticket_id}{RESET} to SQLite…  "
        f"{DIM}{GRAY}status=Open  priority={priority}  "
        f"created_at={ticket.created_at}{RESET}"
    )

    # ── Step 2: Summarize issue for SMS ───────────────────────────────────────
    logger.info(f"{BLUE}[2/4]{RESET}  Generating concise issue summary for SMS…")
    concise_issue = summarize_issue_for_sms(issue_summary)
    _kv("SMS issue line", concise_issue, WHITE)

    # ── Step 3: Twilio SMS alert ──────────────────────────────────────────────
    logger.info(f"{BLUE}[3/4]{RESET}  Preparing Twilio SMS alert…")

    sms_body = f"🚨 Ticket #{ticket_id} | {customer_name} | {priority.upper()} | {concise_issue}"

    if _twilio_ready:
        logger.info(
            f"  Twilio API attempt  →  from={TWILIO_FROM}  to={MOBILE}  "
            f"chars={len(sms_body)}"
        )
        try:
            from twilio.rest import Client
            client  = Client(TWILIO_SID, TWILIO_AUTH)
            message = client.messages.create(
                body=sms_body,
                from_=TWILIO_FROM,
                to=MOBILE,
            )
            logger.info(
                f"{GREEN}✔{RESET}  Twilio SMS sent successfully.  "
                f"SID: {BOLD}{message.sid}{RESET}"
            )
            sms_status = f"SMS alert dispatched to on-call analyst (SID: {message.sid})."
        except Exception as exc:
            logger.error(
                f"{RED}✘{RESET}  Twilio SMS failed  →  {RED}{exc}{RESET}"
            )
            sms_status = f"SMS alert failed ({exc}). Ticket persisted — manual follow-up required."
    else:
        logger.warning(
            f"{YELLOW}⚠{RESET}  Twilio not configured — skipping SMS alert"
        )
        sms_status = "SMS alert skipped (Twilio not configured)."

    # ── Step 4: Build response ────────────────────────────────────────────────
    logger.info(f"{BLUE}[4/4]{RESET}  Building response for main agent…")

    response = (
        f"Escalation ticket created successfully.\n"
        f"  Ticket ID        : #{ticket_id}\n"
        f"  Customer         : {customer_name}\n"
        f"  Priority         : {priority}\n"
        f"  Status           : Open\n"
        f"  Created          : {ticket.created_at}\n"
        f"  SMS issue line   : {concise_issue}\n"
        f"  {sms_status}"
    )

    logger.info(
        f"\n  {BOLD}{GREEN}✔  Ticket #{ticket_id} live — response dispatched to main agent{RESET}\n"
    )
    logger.info(
        f"  {DIM}{GRAY}Response preview → {response[:100].replace(chr(10), ' ')}…{RESET}\n"
    )

    return response


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _section("SERVER STARTING")
    _kv("Transport", "SSE")
    _kv("Host",      "0.0.0.0")
    _kv("Port",      "8004")
    _kv("Endpoint",  "http://0.0.0.0:8004/sse", GREEN)
    logger.info("")
    logger.info(
        f"  {BOLD}{GREEN}Ready — waiting for connections{RESET}  "
        f"{DIM}{GRAY}(Ctrl+C to stop){RESET}\n"
    )
    mcp.run(transport="sse", host="0.0.0.0", port=8004)