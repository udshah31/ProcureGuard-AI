"""
agent/state.py
──────────────
Defines the shared AgentState for the ProcureGuard LangGraph.

Fields:
    messages    – full conversation history (append-only via add_messages)
    role        – the requesting user's role (requester | approver | finance)
    context     – structured facts extracted mid-conversation
                  e.g. {"po_number": "PO-001", "vendor": "Acme", "amount": 75000}
    risk_flags  – compliance warnings accumulated during the session
                  e.g. ["Amount exceeds $50k threshold", "Vendor has active disputes"]
"""

from typing import Annotated, Sequence
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # ── Core conversation history ─────────────────────────────────────────────
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # ── Session context ───────────────────────────────────────────────────────
    role: str                   # "requester" | "approver" | "finance" | "admin"
    context: dict               # structured facts extracted during the session
    risk_flags: list[str]       # compliance warnings accumulated this session
