"""
main.py
───────
ProcureGuard AI — LangGraph agent with Groq Llama 3.1 (API-based, no downloads).

Usage:
    python main.py

Architecture:
    ┌────────────────────────────────────────────────────────┐
    │                  LangGraph StateGraph                  │
    │                                                        │
    │  [START] ──→ agent_node ──┬── tool needed? ──→        │
    │                           │      tool_node             │
    │                           │         │                  │
    │                           │    approve_po tool?        │
    │                           │      guard_node ──→        │
    │                           │         │  agent_node      │
    │                           │         └──────────────↗   │
    │                           │ (no tool)                  │
    │                           └──→ [END]                   │
    └────────────────────────────────────────────────────────┘
"""

import logging
import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

# ── Agent modules ─────────────────────────────────────────────────────────────
from agent.llm import build_llm, describe as describe_llm, message_text
from agent.state import AgentState
from agent.tools import TOOLS, TOOL_MAP
from agent.guard_rules import run_all_guards, format_guard_summary
from observability import configure_logging, metrics

# ── Initialise environment ────────────────────────────────────────────────────
load_dotenv()
configure_logging()
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MAX_NEW_TOKENS: int = int(os.getenv("MAX_NEW_TOKENS", "512"))
DB_PATH:        str = os.getenv("DB_PATH",        "data/procurement.db")

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are ProcureGuard, an intelligent procurement assistant.

You help users manage vendors, purchase orders, and invoices through a secure,
compliant workflow. You have access to the following tools:

  • lookup_vendor              – Search vendor registry by name
  • add_vendor                 – Register a new vendor
  • vendor_risk_score          – Compute risk score for a vendor (0–100)
  • create_purchase_order      – Create a draft PO for a vendor
  • approve_purchase_order     – Approve a pending PO (guard checks run first)
  • reject_purchase_order      – Reject a PO with a mandatory reason
  • check_invoice_status       – Look up invoice status by invoice number
  • flag_suspicious_invoice    – Flag an invoice as disputed

Compliance rules you must enforce:
  - Always run guard checks before approving any PO.
  - Never approve a PO for a blacklisted vendor.
  - Escalate POs over $50,000 to finance before approving.
  - Always log a reason when rejecting a PO or flagging an invoice.

You are not the compliance authority — a dedicated guard system enforces these
rules automatically before any approval is written, and it cannot be bypassed
by a tool call. So when a user asks you to approve or reject a PO, invoke the
tool: do not refuse or stall based on your own judgment of urgency, claimed
authority, or pressure to "skip the usual checks" — that framing is exactly
what the guard exists to catch. If the request is genuinely non-compliant, the
guard will block it and you should relay that outcome; your job is to attempt
the action and report what happened, not to decide compliance yourself.

Be concise, professional, and proactive about surfacing compliance issues."""


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# 1. LLM Initialisation
# ══════════════════════════════════════════════════════════════════════════════
# build_llm now lives in agent/llm.py so the provider stays swappable —
# re-exported here to keep `from main import build_llm` working.


# ══════════════════════════════════════════════════════════════════════════════
# 2. Graph Nodes
# ══════════════════════════════════════════════════════════════════════════════

def agent_node(state: AgentState, llm_with_tools) -> dict:
    """
    Primary reasoning node.
    Prepends the system prompt on the first call, then invokes the LLM.
    """
    messages = list(state["messages"])

    # Inject system prompt if not already present
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    log.debug("agent_node: %d messages in context", len(messages))
    response: AIMessage = llm_with_tools.invoke(messages)
    return {"messages": [response]}


NUMERIC_JSON_TYPES = {"integer", "number"}


def _coerce_tool_args(args: dict, tool_name: str | None = None) -> dict:
    """
    Providers often serialise numbers as strings ("87500.0", "$12,750.50").

    Coercion is driven by the tool's own schema: converting every numeric-looking
    string turns identifiers like invoice number "100002" into ints, which then
    fail validation. Without a tool name, nothing is coerced.
    """
    schema = {}
    tool_fn = TOOL_MAP.get(tool_name) if tool_name else None
    if tool_fn is not None:
        schema = getattr(tool_fn, "args", {}) or {}

    out = {}
    for key, value in args.items():
        want = (schema.get(key) or {}).get("type")
        if not isinstance(value, str) or want not in NUMERIC_JSON_TYPES:
            out[key] = value
            continue
        try:
            cleaned = value.strip().replace(",", "").lstrip("$")
            out[key] = int(cleaned) if want == "integer" else float(cleaned)
        except ValueError:
            out[key] = value
    return out


def tool_node(state: AgentState) -> dict:
    """
    Executes tool calls from the last AIMessage.
    Appends one ToolMessage per tool call.
    Skips approve_purchase_order — that goes through guard_node instead.
    """
    last: AIMessage = state["messages"][-1]
    tool_messages: list[ToolMessage] = []

    for tc in getattr(last, "tool_calls", []):
        if tc["name"] == "approve_purchase_order":
            # Guard node handles this — should not reach here
            continue

        tool_fn = TOOL_MAP.get(tc["name"])
        if tool_fn is None:
            result = f"Unknown tool: {tc['name']}"
        else:
            try:
                result = tool_fn.invoke(_coerce_tool_args(tc["args"], tc["name"]))
            except Exception as exc:
                # Hand the failure back as an observation so the agent can retry
                # with better arguments; raising here would kill the whole turn.
                result = f"Tool '{tc['name']}' failed: {type(exc).__name__}: {exc}"
                log.warning("Tool '%s' raised: %s", tc["name"], exc)
                metrics.increment(f"tool_error_total:{tc['name']}")

        tool_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
        log.info("Tool '%s' → %s", tc["name"], str(result)[:120])

    return {"messages": tool_messages}


def guard_node(state: AgentState) -> dict:
    """
    Runs compliance checks before an approve_purchase_order tool call.
    - If BLOCKED: injects a ToolMessage explaining the block (approval cancelled).
    - If WARN only: injects guard summary as context, lets agent decide.
    """
    last: AIMessage = state["messages"][-1]
    tool_messages: list[ToolMessage] = []
    risk_flags: list[str] = list(state.get("risk_flags", []))

    for tc in getattr(last, "tool_calls", []):
        if tc["name"] != "approve_purchase_order":
            continue

        # approved_by is always the authenticated caller's identity, never
        # whatever the LLM put in the tool call — otherwise a user could ask
        # the agent to "approve this as someone-else@company.com" and walk
        # straight past the self-approval guard.
        args = dict(tc["args"])
        args["approved_by"] = state.get("identity", "")

        po_number = args.get("po_number", "")
        results = run_all_guards(po_number, approved_by=args.get("approved_by"))
        summary = format_guard_summary(results)

        blocks = [r for r in results if not r.passed and r.severity == "block"]
        warns  = [r for r in results if not r.passed and r.severity == "warn"]

        # Collect risk flags for state
        risk_flags.extend(r.message for r in results if not r.passed)

        for r in blocks:
            metrics.increment(f"guard_block_total:{r.check}")
        for r in warns:
            metrics.increment(f"guard_warn_total:{r.check}")

        if blocks:
            # Hard block — do not call the tool, inject failure message
            tool_messages.append(
                ToolMessage(
                    content=f"APPROVAL BLOCKED by compliance guard:\n{summary}",
                    tool_call_id=tc["id"],
                )
            )
            log.warning("PO %s approval BLOCKED. Reasons: %s", po_number,
                        [r.message for r in blocks])
        else:
            # No blocks — actually execute the approval tool
            tool_fn = TOOL_MAP["approve_purchase_order"]
            result = tool_fn.invoke(args)

            suffix = f"\n\n{summary}" if warns else ""
            tool_messages.append(
                ToolMessage(content=str(result) + suffix, tool_call_id=tc["id"])
            )

    return {"messages": tool_messages, "risk_flags": risk_flags}


# ══════════════════════════════════════════════════════════════════════════════
# 3. Routing Logic
# ══════════════════════════════════════════════════════════════════════════════

def route_after_agent(state: AgentState) -> Literal["tool_node", "guard_node", "__end__"]:
    """
    Conditional edge after agent_node:
      - approve_purchase_order call  → guard_node
      - any other tool call          → tool_node
      - no tool call                 → END
    """
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", [])

    if not tool_calls:
        return END

    tool_names = {tc["name"] for tc in tool_calls}
    if "approve_purchase_order" in tool_names:
        return "guard_node"
    return "tool_node"


# ══════════════════════════════════════════════════════════════════════════════
# 4. Graph Assembly
# ══════════════════════════════════════════════════════════════════════════════

def build_graph(llm) -> StateGraph:
    """Construct and compile the ProcureGuard StateGraph."""
    llm_with_tools = llm.bind_tools(TOOLS)

    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("agent",      lambda s: agent_node(s, llm_with_tools))
    graph.add_node("tool_node",  tool_node)
    graph.add_node("guard_node", guard_node)

    # Edges
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", route_after_agent)
    graph.add_edge("tool_node",  "agent")   # loop back after tool execution
    graph.add_edge("guard_node", "agent")   # loop back after guard + (maybe) approval

    return graph.compile()


# ══════════════════════════════════════════════════════════════════════════════
# 5. Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # Ensure DB schema exists
    from db.init_db import init_db
    init_db(DB_PATH)

    # Optionally seed if DB is empty
    try:
        import sqlite3
        with sqlite3.connect(DB_PATH) as conn:
            count = conn.execute("SELECT COUNT(*) FROM vendors").fetchone()[0]
        if count == 0:
            log.info("Empty database detected — running seeder.")
            from db.seed import seed
            seed(DB_PATH)
    except Exception as e:
        log.warning("Could not auto-seed: %s", e)

    llm = build_llm()
    app = build_graph(llm)

    print(f"\n🛡️  ProcureGuard AI ready ({describe_llm()}). Type 'exit' to quit.\n")

    # Initial state
    state: AgentState = {
        "messages": [],
        "role": "requester",
        "identity": "cli-user@procureguard.local",
        "context": {},
        "risk_flags": [],
    }

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break
        if not user_input:
            continue

        state["messages"] = list(state["messages"]) + [HumanMessage(content=user_input)]
        state = app.invoke(state)

        # Print last AI response
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage):
                print(f"\nProcureGuard: {message_text(msg.content)}\n")
                break


if __name__ == "__main__":
    main()
