"""
app/factory.py — FastAPI application factory & lifespan manager.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.core.config import DB_PATH, LLM_CONFIGURED, MCP_SERVER_URLS
from app.core.agent import build_agent
from app.db.database import init_db
from app.api import chat, notepad as notepad_router, evaluation

logger = logging.getLogger("crossborder-copilot")

from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────────
    init_db()

    extra_tools: list = []
    mcp_client = None

    if MCP_SERVER_URLS:
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
            server_config = {
                f"mcp_{i}": {"url": url, "transport": "sse"}
                for i, url in enumerate(MCP_SERVER_URLS)
            }
            mcp_client = MultiServerMCPClient(server_config)
            extra_tools = await mcp_client.get_tools()
            logger.info("Loaded %d MCP tool(s).", len(extra_tools))
        except ImportError:
            logger.warning("langchain-mcp-adapters not installed; skipping MCP tools.")
        except Exception as exc:
            logger.warning("MCP connection failed (%s); continuing without MCP tools.", exc)

    app.state.agent = None
    app.state.startup_error = None

    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        if LLM_CONFIGURED:
            try:
                app.state.agent = build_agent(extra_tools, checkpointer)
            except Exception as exc:
                app.state.startup_error = str(exc)
                logger.exception("Chat agent startup failed: %s", exc)
        else:
            app.state.startup_error = (
                "LLM_BASE_URL and LLM_API_KEY are not set. Chat endpoints are disabled."
            )
            logger.warning(app.state.startup_error)

        app.state.mcp_client = mcp_client

        logger.info("CrossBorder Copilot ready.")
        yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("Shutdown complete.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="CrossBorder Copilot",
        description="LangGraph-powered cross-border shipping assistant with MCP + SQLite memory.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.agent = None
    app.state.startup_error = None
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.include_router(chat.router)
    app.include_router(notepad_router.router)
    app.include_router(evaluation.router)

    def startup_error() -> str | None:
        error = getattr(app.state, "startup_error", None)
        if error:
            return error
        if not LLM_CONFIGURED:
            return "LLM_BASE_URL and LLM_API_KEY are not set. Chat endpoints are disabled."
        return None

    @app.get("/", tags=["Health"])
    async def root():
        return {
            "service":    "CrossBorder Copilot",
            "status":     "running" if getattr(app.state, "agent", None) else "degraded",
            "chat_ready": bool(getattr(app.state, "agent", None)),
            "llm_configured": LLM_CONFIGURED,
            "startup_error": startup_error(),
            "mcp_servers": MCP_SERVER_URLS or [],
        }

    @app.get("/health", tags=["Health"])
    async def health():
        return {
            "service": "CrossBorder Copilot",
            "status": "ok",
            "chat_ready": bool(getattr(app.state, "agent", None)),
            "llm_configured": LLM_CONFIGURED,
            "database_path": DB_PATH,
            "startup_error": startup_error(),
        }

    return app
