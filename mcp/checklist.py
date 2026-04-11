"""
mcp/checklist_agent_mcp.py — Checklist Agent MCP Server

Exposes a single tool: generate_shipping_checklist
Runs via SSE on host 0.0.0.0, port 8003.

Run from the mcp/ directory:
    python checklist_agent_mcp.py

Or from the project root:
    python mcp/checklist_agent_mcp.py
"""

import sys
import os
import time
import logging
from dotenv import load_dotenv

load_dotenv()
BASE_URL = os.getenv("LLM_BASE_URL")
API_KEY  = os.getenv("LLM_API_KEY")

# Allow imports from the project root regardless of where this script is invoked from.
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.append(_root)

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
        name  = f"{DIM}{GRAY}{record.name}{RESET}"
        msg   = record.getMessage()
        return f"{DIM}{GRAY}{ts}{RESET}  {level}  {name}  {msg}"


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(PrettyFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger("checklist-agent")


def _banner():
    print(f"""
{YELLOW}{BOLD}╔══════════════════════════════════════════════════════╗
║        CrossBorder Copilot — Checklist Agent         ║
║                   MCP Server v1.0                    ║
╚══════════════════════════════════════════════════════╝{RESET}
""")


def _section(title: str):
    bar = "─" * 54
    logger.info(f"{BOLD}{WHITE}{bar}{RESET}")
    logger.info(f"{BOLD}{WHITE}  {title}{RESET}")
    logger.info(f"{BOLD}{WHITE}{bar}{RESET}")


def _kv(key: str, value: str, color: str = CYAN):
    logger.info(f"  {DIM}{GRAY}{key:<22}{RESET}{color}{value}{RESET}")


# ── Boot sequence ─────────────────────────────────────────────────────────────

_banner()
_section("STARTUP")

from retriever import Retriever
from langchain_openai import ChatOpenAI
from fastmcp import FastMCP

logger.info(f"{GREEN}✔{RESET}  Core imports OK")

# ── LLM ───────────────────────────────────────────────────────────────────────

_kv("LLM base URL", BASE_URL or "(not set)", YELLOW if not BASE_URL else CYAN)
_kv("LLM API key",  "***" + (API_KEY[-4:] if API_KEY else "(not set)"))
_kv("Temperature",  "0.2")

if not BASE_URL or not API_KEY:
    logger.error(f"{RED}LLM_BASE_URL or LLM_API_KEY not set — check your .env{RESET}")
    sys.exit(1)

llm = ChatOpenAI(base_url=BASE_URL, api_key=API_KEY, temperature=0.2)
logger.info(f"{GREEN}✔{RESET}  LLM client initialised")

SYSTEM_PROMPT = """You are a meticulous cross-border logistics expert with deep knowledge of
international trade regulations, customs procedures, import/export compliance, and shipping
documentation requirements.

When generating checklists:
- Be precise and cite sources using notation like [Source 1], [Source 2] where context is provided.
- If the retrieved context does not contain specific details for a rule or requirement, explicitly
  state: "Specific details not found in available sources — verify with official customs authority."
- Never hallucinate regulations, tariff codes, or policy details.
- Structure your output with clear sections and bullet points.
"""

# ── Retriever ─────────────────────────────────────────────────────────────────

_chroma_dir = os.path.join(_root, "chroma_db")
_kv("Project root",  _root)
_kv("chroma_db path", _chroma_dir)

if not os.path.isdir(_chroma_dir):
    logger.warning(
        f"{YELLOW}⚠{RESET}  chroma_db not found at expected path — "
        f"retriever may return empty results"
    )
else:
    logger.info(f"{GREEN}✔{RESET}  chroma_db directory found")

retriever = Retriever(chroma_dir=_chroma_dir)
logger.info(f"{GREEN}✔{RESET}  Retriever initialised")

# ── MCP server ────────────────────────────────────────────────────────────────

mcp = FastMCP("checklist-agent")
logger.info(f"{GREEN}✔{RESET}  FastMCP instance created  →  name=checklist-agent")


# ── Tool ──────────────────────────────────────────────────────────────────────

@mcp.tool()
def generate_shipping_checklist(
    item_description: str,
    origin: str,
    destination: str,
) -> str:
    """Generate a structured shipping compliance checklist for a cross-border shipment.

    Args:
        item_description: Description of the goods being shipped (e.g. 'lithium-ion laptop batteries').
        origin: Country or region of origin (e.g. 'Canada').
        destination: Country or region of destination (e.g. 'Germany').

    Returns:
        A structured, bulleted checklist covering required documents, restrictions/prohibitions,
        and recommended next steps, grounded in retrieved regulatory context.
    """
    t_start = time.perf_counter()

    _section("TOOL INVOKED  →  generate_shipping_checklist")
    _kv("Item",        item_description, WHITE)
    _kv("Origin",      origin,           WHITE)
    _kv("Destination", destination,      WHITE)

    # ── Step 1: Build retrieval query ─────────────────────────────────────────
    query = (
        f"shipping requirements checklist for {item_description} "
        f"from {origin} to {destination}"
    )
    logger.info(f"\n  {DIM}{GRAY}RAG query →{RESET} {PURPLE}{query}{RESET}\n")

    # ── Step 2: Retrieve chunks ───────────────────────────────────────────────
    logger.info(f"{BLUE}[1/3]{RESET}  Querying vector store  {DIM}{GRAY}(top_k=5){RESET}")
    t_ret = time.perf_counter()
    chunks = retriever.retrieve(query, top_k=5)
    ret_ms = (time.perf_counter() - t_ret) * 1000

    if chunks:
        logger.info(
            f"{GREEN}✔{RESET}  Retrieved {BOLD}{len(chunks)}{RESET} chunk(s)  "
            f"{DIM}{GRAY}{ret_ms:.0f} ms{RESET}"
        )
        for i, chunk in enumerate(chunks, 1):
            preview = chunk[:90].replace("\n", " ").strip()
            logger.info(f"     {DIM}{GRAY}[Source {i}]{RESET}  {preview}{DIM}…{RESET}")
        context_blocks = "\n\n".join(
            f"[Source {i+1}] {chunk}" for i, chunk in enumerate(chunks)
        )
        context_section = f"RETRIEVED REGULATORY CONTEXT:\n{context_blocks}"
    else:
        logger.warning(
            f"{YELLOW}⚠{RESET}  No chunks returned from vector store  "
            f"{DIM}{GRAY}({ret_ms:.0f} ms){RESET} — proceeding with empty context"
        )
        context_section = (
            "RETRIEVED REGULATORY CONTEXT:\n"
            "No relevant documents were found in the local knowledge base for this query."
        )

    # ── Step 3: Build prompt ──────────────────────────────────────────────────
    logger.info(f"{BLUE}[2/3]{RESET}  Building prompt")

    user_prompt = f"""{context_section}

---

Using the context above (and only the context above for specific regulatory claims), generate a
comprehensive shipping compliance checklist for the following shipment:

- Item: {item_description}
- Origin: {origin}
- Destination: {destination}

Your checklist must include these three sections:

## 1. Required Documents
List every document typically required for this shipment. Cite the source (e.g. [Source 2]) for
each requirement where the context supports it. If a document type is standard practice not
covered by the retrieved context, note it as general industry practice.

## 2. Restrictions & Prohibitions
List any known restrictions, prohibitions, licensing requirements, or special conditions that
apply to this item/route. If the context does not contain specific restriction details for this
route, explicitly state that and advise the shipper to verify with the relevant customs authority.

## 3. Next Steps
Provide a prioritised, actionable list of next steps the shipper should take before dispatch.
Include any recommended verifications with official sources.
"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]

    prompt_chars = sum(len(m["content"]) for m in messages)
    logger.info(f"     Prompt assembled  {DIM}{GRAY}({prompt_chars:,} chars total){RESET}")

    # ── Step 4: LLM call ──────────────────────────────────────────────────────
    logger.info(
        f"{BLUE}[3/3]{RESET}  Calling LLM  "
        f"{DIM}{GRAY}base_url={BASE_URL}  temp=0.2{RESET}"
    )
    t_llm = time.perf_counter()
    response = llm.invoke(messages)
    llm_s = time.perf_counter() - t_llm

    content   = response.content
    total_s   = time.perf_counter() - t_start

    logger.info(
        f"{GREEN}✔{RESET}  LLM response received  "
        f"{DIM}{GRAY}{llm_s:.2f}s  |  {len(content):,} chars{RESET}"
    )

    # ── Per-call summary ──────────────────────────────────────────────────────
    logger.info(f"\n  {BOLD}{GREEN}✔  Checklist generated successfully{RESET}")
    _kv("Retrieval time",  f"{ret_ms:.0f} ms",         GREEN)
    _kv("LLM latency",     f"{llm_s:.2f} s",           GREEN)
    _kv("Total wall time", f"{total_s:.2f} s",          GREEN)
    _kv("Output length",   f"{len(content):,} chars",   GREEN)
    _kv("Chunks used",
        str(len(chunks)) if chunks else "0  (no RAG context)",
        GREEN if chunks else YELLOW,
    )
    logger.info("")

    return content


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _section("SERVER STARTING")
    _kv("Transport", "SSE")
    _kv("Host",      "0.0.0.0")
    _kv("Port",      "8003")
    _kv("Endpoint",  "http://0.0.0.0:8003/sse", GREEN)
    logger.info("")
    logger.info(
        f"  {BOLD}{GREEN}Ready — waiting for connections{RESET}  "
        f"{DIM}{GRAY}(Ctrl+C to stop){RESET}\n"
    )
    mcp.run(transport="sse", host="0.0.0.0", port=8003)
