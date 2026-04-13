"""
app/core/agent.py — LangGraph agent construction.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, trim_messages
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from app.core.config import LLM_API_KEY, LLM_BASE_URL, LLM_CONFIGURED, LLM_TEMP, MAX_TOKENS, SYSTEM_PROMPT
from app.tools.knowledge_base import knowledge_base
from app.tools.notepad import notepad


def _build_llm(tools: list) -> ChatOpenAI:
    if not LLM_CONFIGURED:
        raise RuntimeError(
            "LLM_BASE_URL and LLM_API_KEY must be set before the chat agent can be started."
        )
    llm = ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=LLM_TEMP,
        max_tokens=MAX_TOKENS,
    )
    return llm.bind_tools(tools) if tools else llm


def _trim(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Trim message history to stay within token budget."""
    return trim_messages(
        messages,
        max_tokens=MAX_TOKENS,
        token_counter=len,   # lightweight; swap for tiktoken if desired
        strategy="last",
        include_system=True,
        allow_partial=False,
    )


def build_agent(extra_tools: list, checkpointer):
    """Compile and return a LangGraph StateGraph agent."""
    all_tools      = [knowledge_base, notepad] + extra_tools
    llm_with_tools = _build_llm(all_tools)

    def agent_node(state: MessagesState):
        trimmed = _trim(state["messages"])
        response = llm_with_tools.invoke(
            [SystemMessage(content=SYSTEM_PROMPT)] + trimmed
        )
        return {"messages": [response]}

    def should_continue(state: MessagesState):
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    tool_node = ToolNode(all_tools)

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=checkpointer)
