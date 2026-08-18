"""
api/routers/chat.py
───────────────────
POST /api/v1/chat   — Main agent endpoint with multi-turn session support.
GET  /api/v1/health — Liveness check.
"""

import logging
from typing import Any

from fastapi import APIRouter, Request, HTTPException, Depends
from langchain_core.messages import AIMessage, HumanMessage

from agent.llm import message_text
from api.auth import AuthContext, require_api_key
from api.models import ChatRequest, ChatResponse, HealthResponse
from observability import Timer, metrics

log = logging.getLogger(__name__)
router = APIRouter()


def _extract_tool_calls(messages: list) -> list[str]:
    """Return tool names invoked in this conversation turn."""
    names: list[str] = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            for tc in getattr(msg, "tool_calls", []):
                if tc["name"] not in names:
                    names.append(tc["name"])
    return names


@router.post("/chat", response_model=ChatResponse, summary="Send a message to the agent")
def chat(request: Request, body: ChatRequest, auth: AuthContext = Depends(require_api_key)) -> ChatResponse:
    """
    Send a message to ProcureGuard AI and receive a response.

    Requires an `X-API-Key` header — the caller's role and identity are
    resolved from the key server-side (see api/auth.py), not from the
    request body.

    - **session_id**: Unique ID to maintain conversation history across calls.
    - **message**: Your question or command in natural language.

    The agent will use the appropriate tools, enforce compliance guard rules,
    and return a structured response including any risk flags raised.
    """
    graph:            Any = request.app.state.graph
    session_store:    Any = request.app.state.session_store
    chat_rate_limiter: Any = request.app.state.chat_rate_limiter

    # Rate-limited per client address — there's no real auth yet, so
    # session_id/role are self-reported and not usable as a limiter key.
    client_key = request.client.host if request.client else "unknown"
    allowed, retry_after = chat_rate_limiter.check(client_key)
    if not allowed:
        metrics.increment("chat_rate_limited_total")
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please slow down and try again shortly.",
            headers={"Retry-After": str(int(retry_after) + 1)},
        )

    if graph is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No LLM configured. Set a provider key in .env (GROQ_API_KEY or "
                "GOOGLE_API_KEY), or set LLM_PROVIDER=ollama to run locally."
            ),
        )

    # Get or create session state
    state = session_store.get_or_create(body.session_id, role=auth.role, identity=auth.identity)

    # Track message count before invocation (to extract new messages only)
    prev_count = len(state["messages"])

    # Append new user message and invoke graph
    state["messages"] = list(state["messages"]) + [HumanMessage(content=body.message)]

    metrics.increment("chat_requests_total")
    try:
        with Timer() as t:
            state = graph.invoke(state)
        metrics.record_latency("chat_agent_invoke", t.elapsed)
    except Exception as exc:
        metrics.increment("chat_errors_total")
        log.exception("Agent invocation failed for session '%s'.", body.session_id)
        raise HTTPException(status_code=500, detail=f"Agent error: {str(exc)}")

    # Persist updated state
    session_store.update(body.session_id, state)

    # Extract new messages from this turn
    new_messages = list(state["messages"])[prev_count:]

    # Get last AI reply
    reply = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage):
            reply = message_text(msg.content)
            break

    # Collect tool calls made in this turn
    tool_calls_made = _extract_tool_calls(new_messages)

    # Collect risk flags added this turn
    risk_flags: list[str] = list(state.get("risk_flags", []))

    log.info(
        "Chat [session=%s role=%s tools=%s flags=%d] → %d chars reply",
        body.session_id, auth.role, tool_calls_made, len(risk_flags), len(reply),
    )

    return ChatResponse(
        session_id=body.session_id,
        reply=reply or "(No response generated.)",
        risk_flags=risk_flags,
        tool_calls_made=tool_calls_made,
    )


@router.get("/health", response_model=HealthResponse, summary="Server health check")
def health(request: Request) -> HealthResponse:
    """Returns server status, loaded model name, DB path, and active session count."""
    return HealthResponse(
        status="ok",
        model_name=request.app.state.model_name,
        db_path=request.app.state.db_path,
        active_sessions=request.app.state.session_store.active_count(),
    )
