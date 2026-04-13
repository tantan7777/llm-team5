"""
app/core/config.py — Centralised configuration.
"""

from __future__ import annotations

import logging
import os

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")

# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "").strip()
LLM_API_KEY  = os.environ.get("LLM_API_KEY", "").strip()
LLM_TEMP     = float(os.environ.get("LLM_TEMP", "0.1"))
MAX_TOKENS   = int(os.environ.get("MAX_TOKENS", "10000"))
LLM_CONFIGURED = bool(LLM_BASE_URL and LLM_API_KEY)
# ── Database ──────────────────────────────────────────────────────────────────
DB_PATH    = os.environ.get("DB_PATH", "crossborder.db")
SQLITE_URL = f"sqlite:///{DB_PATH}"

# ── MCP ───────────────────────────────────────────────────────────────────────
# Comma-separated list of MCP server URLs, e.g.:
MCP_SERVER_URLS: list[str] = [
    u.strip()
    for u in os.environ.get("MCP_SERVER_URLS", "").split(",")
    if u.strip()
]

#Logging the loaded configuration for verification
logging.info(f"LLM_BASE_URL: {LLM_BASE_URL}")
logging.info(f"LLM_API_KEY: {'***' if LLM_API_KEY else '(not set)'}")
logging.info(f"LLM_CONFIGURED: {LLM_CONFIGURED}")
logging.info(f"LLM_TEMP: {LLM_TEMP}")
logging.info(f"MAX_TOKENS: {MAX_TOKENS}")
logging.info(f"DB_PATH: {DB_PATH}")
logging.info(f"SQLITE_URL: {SQLITE_URL}")
logging.info(f"MCP_SERVER_URLS: {MCP_SERVER_URLS}")

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are CrossBorder Copilot, an expert assistant specialising in
cross-border shipping, customs regulations, import/export compliance, tariff
classification (HS codes), duties & taxes, trade agreements, and logistics
optimisation.

Guidelines:
- Provide accurate, actionable advice tailored to the specific countries involved.
- Use the knowledge_base tool for DHL policy, customs document, surcharge, service,
  restricted-goods, and support-process questions before giving a factual answer.
- Cite retrieved sources when the knowledge_base tool returns citations.
- When unsure, say so clearly rather than guessing.
- Use the notepad tool to remember key details (addresses, HS codes, declared values,
  deadlines) so you can refer to them later without asking the user again.
- The conversation history includes a SESSION_ID system message. Use that exact value
  as the session_id when calling the notepad tool.
- Keep answers concise but thorough; use bullet points where helpful.
- Always clarify which regulations are current as of your knowledge cutoff and advise
  users to verify with official sources for time-sensitive matters."""
