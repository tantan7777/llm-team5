"""
app/api/chat.py — /chat routes.

Key behaviours:
  - If no session_id is supplied (or it's blank), one is auto-generated (UUID4).
  - The session_id is injected as the very first SystemMessage in the thread the
    first time a session is used, so it always appears at the top of /chat/history
    and can be copied directly for notepad use.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, HTTPException, Request
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

logger = logging.getLogger("crossborder-copilot")
router = APIRouter(tags=["Chat"])

# Track which sessions have already had their ID injected so we only do it once.
_sessions_initialised: set[str] = set()


# ── Pydantic models ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query:      str
    session_id: str = ""          # blank → auto-generate


class ChatResponse(BaseModel):
    session_id:      str
    response:        str
    tool_calls_made: list[str] = []


class HistoryMessage(BaseModel):
    role:    str
    content: str


class HistoryResponse(BaseModel):
    session_id: str
    messages:   list[HistoryMessage]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_session_id(raw: str) -> str:
    """Return raw if non-empty, otherwise generate a new UUID4."""
    return raw.strip() if raw.strip() else str(uuid.uuid4())


async def _ensure_session_initialised(agent, session_id: str) -> None:
    """
    On the first invocation of a session, inject a silent SystemMessage that
    records the session_id.  This makes it the first item in chat history so
    users can copy it without interrogating the bot.
    """
    if session_id in _sessions_initialised:
        return

    config = {"configurable": {"thread_id": session_id}}
    seed_message = SystemMessage(
        content=f"SESSION_ID: {session_id}"
    )
    # Write directly to the checkpointer via a no-op agent invocation seed.
    # We use update_state so no LLM call is made.
    await agent.aupdate_state(config, {"messages": [seed_message]})
    _sessions_initialised.add(session_id)
    logger.info("Session initialised: %s", session_id)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/chat/invoke", response_model=ChatResponse)
async def chat_invoke(req: ChatRequest, request: Request):
    """Send a message and receive the agent's response.

    If *session_id* is omitted or blank a new one is generated automatically
    and returned in the response — use it for all subsequent requests in the
    same conversation.
    """
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        detail = getattr(
            request.app.state,
            "startup_error",
            "Chat agent is not initialised.",
        ) or "Chat agent is not initialised. Set LLM_BASE_URL and LLM_API_KEY, then restart the backend."
        raise HTTPException(status_code=503, detail=detail)

    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty.")

    session_id = _resolve_session_id(req.session_id)
    config     = {"configurable": {"thread_id": session_id}}

    # Inject session_id as first history entry (idempotent).
    await _ensure_session_initialised(agent, session_id)

    input_state = {"messages": [HumanMessage(content=req.query)]}

    try:
        result = await agent.ainvoke(input_state, config)
    except Exception as exc:
        logger.exception("Agent error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}")

    final_ai = next(
        (m for m in reversed(result["messages"]) if isinstance(m, AIMessage)),
        None,
    )
    response_text = final_ai.content if final_ai else "(no response)"

    tool_calls_made = [
        tc["name"]
        for m in result["messages"]
        if isinstance(m, AIMessage)
        for tc in (m.tool_calls or [])
    ]

    return ChatResponse(
        session_id=session_id,
        response=response_text,
        tool_calls_made=tool_calls_made,
    )


@router.get("/chat/history/{session_id}", response_model=HistoryResponse)
async def chat_history(session_id: str, request: Request):
    """Retrieve stored conversation history for a session.

    The first entry will always be a *system* message containing the
    session_id in the form ``SESSION_ID: <id>`` — copy it from there for
    notepad operations.
    """
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        detail = getattr(
            request.app.state,
            "startup_error",
            "Chat agent is not initialised.",
        ) or "Chat agent is not initialised. Set LLM_BASE_URL and LLM_API_KEY, then restart the backend."
        raise HTTPException(status_code=503, detail=detail)

    config = {"configurable": {"thread_id": session_id}}
    try:
        state = await agent.aget_state(config)
    except Exception as exc:
        logger.exception("History fetch error: %s", exc)
        raise HTTPException(status_code=500, detail=f"History error: {exc}")

    messages: list[HistoryMessage] = []
    for msg in (state.values.get("messages") or []):
        if isinstance(msg, HumanMessage):
            messages.append(HistoryMessage(role="user",      content=msg.content))
        elif isinstance(msg, AIMessage):
            messages.append(HistoryMessage(role="assistant", content=msg.content or ""))
        elif isinstance(msg, SystemMessage):
            messages.append(HistoryMessage(role="system",    content=msg.content))

    return HistoryResponse(session_id=session_id, messages=messages)
