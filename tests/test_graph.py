"""
tests/test_graph.py
───────────────────
Graph-level tests driven by a scripted fake LLM.

The point is to pin down the parts of the agent we control — routing, guard
enforcement, and tool-arg coercion — without paying for or depending on a live
Groq call. Model quality is measured separately in evals/.
"""

import sqlite3

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from main import build_graph, route_after_agent, _coerce_tool_args


# ── Fake LLM ──────────────────────────────────────────────────────────────────

class FakeLLM:
    """Returns pre-scripted AIMessages in order. Records what it was sent."""

    def __init__(self, responses: list[AIMessage]):
        self._responses = list(responses)
        self.calls: list[list] = []

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        if not self._responses:
            return AIMessage(content="No further action.")
        return self._responses.pop(0)


def tool_call_message(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id}],
    )


def initial_state(text: str, role: str = "approver", identity: str = "approver@company.com") -> dict:
    return {
        "messages": [HumanMessage(content=text)],
        "role": role,
        "identity": identity,
        "context": {},
        "risk_flags": [],
    }


def tool_outputs(state: dict) -> list[str]:
    return [m.content for m in state["messages"] if isinstance(m, ToolMessage)]


# ── Routing ───────────────────────────────────────────────────────────────────

def test_no_tool_call_ends_the_graph():
    state = {"messages": [AIMessage(content="Hello.")]}
    assert route_after_agent(state) == "__end__"


def test_approval_is_routed_through_the_guard():
    state = {"messages": [tool_call_message("approve_purchase_order", {"po_number": "PO-1"})]}
    assert route_after_agent(state) == "guard_node"


def test_other_tools_go_straight_to_the_tool_node():
    state = {"messages": [tool_call_message("lookup_vendor", {"name": "Acme"})]}
    assert route_after_agent(state) == "tool_node"


def test_approval_batched_with_other_tools_still_hits_the_guard():
    """An approval must never slip through by being bundled with a safe call."""
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "lookup_vendor", "args": {"name": "Acme"}, "id": "a"},
                    {"name": "approve_purchase_order", "args": {"po_number": "PO-1"}, "id": "b"},
                ],
            )
        ]
    }
    assert route_after_agent(state) == "guard_node"


# ── Argument coercion ─────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "tool_name, raw, expected",
    [
        # Declared numeric → coerced, including money formatting from the model
        ("create_purchase_order", {"amount": "87500.0"}, {"amount": 87500.0}),
        ("create_purchase_order", {"amount": "12,750.50"}, {"amount": 12750.50}),
        ("create_purchase_order", {"amount": "$15000"}, {"amount": 15000.0}),
        ("create_purchase_order", {"amount": 1234.5}, {"amount": 1234.5}),
        # Declared string → left alone even when it looks like a number
        ("check_invoice_status", {"invoice_number": "100002"}, {"invoice_number": "100002"}),
        ("approve_purchase_order", {"po_number": "100002"}, {"po_number": "100002"}),
        ("lookup_vendor", {"name": "Acme Corp"}, {"name": "Acme Corp"}),
        # Unparseable numeric stays as-is rather than raising
        ("create_purchase_order", {"amount": "a lot"}, {"amount": "a lot"}),
    ],
)
def test_coercion_follows_the_tool_schema(tool_name, raw, expected):
    assert _coerce_tool_args(raw, tool_name) == expected


def test_unknown_tool_name_coerces_nothing():
    assert _coerce_tool_args({"amount": "100"}, "no_such_tool") == {"amount": "100"}


def test_numeric_identifier_survives_a_full_tool_call(db, make_vendor, make_po):
    """Regression: blanket coercion turned invoice_number into an int and the
    Pydantic ValidationError took down the whole turn."""
    llm = FakeLLM(
        [
            tool_call_message("check_invoice_status", {"invoice_number": "100002"}),
            AIMessage(content="Checked."),
        ]
    )
    result = build_graph(llm).invoke(initial_state("check invoice 100002"))

    assert any("not found" in out.lower() for out in tool_outputs(result))
    assert not any("ValidationError" in out for out in tool_outputs(result))


def test_requested_by_ignores_a_spoofed_tool_call_argument(db, make_vendor):
    """tool_node must always use the authenticated caller's identity for
    requested_by, never whatever the LLM put in the tool call — otherwise a
    user could ask the agent to raise a PO "on behalf of" someone else."""
    make_vendor("Acme Corp")

    llm = FakeLLM(
        [
            tool_call_message(
                "create_purchase_order",
                {
                    "vendor_name": "Acme Corp",
                    "description": "Spoof test",
                    "amount": 1000.0,
                    "requested_by": "someone-else@company.com",
                },
            ),
            AIMessage(content="Created."),
        ]
    )
    build_graph(llm).invoke(
        initial_state("Create a PO for Acme Corp", identity="real-caller@company.com")
    )

    row = db.execute(
        "SELECT requested_by FROM purchase_orders WHERE description = 'Spoof test'"
    ).fetchone()
    assert row["requested_by"] == "real-caller@company.com"


def test_a_failing_tool_is_reported_not_raised(db):
    """One bad argument should cost a turn, not the conversation."""
    llm = FakeLLM(
        [
            tool_call_message("check_invoice_status", {"wrong_arg": 1}),
            AIMessage(content="Let me try again."),
        ]
    )
    result = build_graph(llm).invoke(initial_state("check an invoice"))

    assert any("failed" in out.lower() for out in tool_outputs(result))
    assert result["messages"][-1].content == "Let me try again."


# ── End-to-end through the compiled graph ─────────────────────────────────────

def test_blacklisted_vendor_approval_is_blocked_and_db_is_unchanged(
    db, make_vendor, make_po
):
    vendor_id = make_vendor("ShadyDeals LLC", status="blacklisted")
    make_po("PO-BAD", vendor_id, amount=5_000.0, status="pending")

    llm = FakeLLM([tool_call_message("approve_purchase_order", {"po_number": "PO-BAD"})])
    result = build_graph(llm).invoke(initial_state("Approve PO-BAD"))

    assert any("APPROVAL BLOCKED" in out for out in tool_outputs(result))

    status = db.execute(
        "SELECT status FROM purchase_orders WHERE po_number = 'PO-BAD'"
    ).fetchone()["status"]
    assert status == "pending"


def test_blocked_approval_writes_no_audit_row(db, make_vendor, make_po):
    vendor_id = make_vendor("ShadyDeals LLC", status="blacklisted")
    make_po("PO-BAD", vendor_id, amount=5_000.0, status="pending")

    llm = FakeLLM([tool_call_message("approve_purchase_order", {"po_number": "PO-BAD"})])
    build_graph(llm).invoke(initial_state("Approve PO-BAD"))

    count = db.execute(
        "SELECT COUNT(*) AS n FROM audit_log WHERE action = 'approved'"
    ).fetchone()["n"]
    assert count == 0


def test_clean_approval_updates_the_po(db, make_vendor, make_po):
    vendor_id = make_vendor("Acme Corp")
    make_po("PO-OK", vendor_id, amount=5_000.0, status="pending")

    llm = FakeLLM(
        [
            tool_call_message(
                "approve_purchase_order",
                {"po_number": "PO-OK", "approved_by": "manager@company.com"},
            )
        ]
    )
    build_graph(llm).invoke(initial_state("Approve PO-OK", identity="approver@company.com"))

    row = db.execute(
        "SELECT status, approved_by FROM purchase_orders WHERE po_number = 'PO-OK'"
    ).fetchone()
    assert row["status"] == "approved"
    assert row["approved_by"] == "approver@company.com"


def test_approved_by_ignores_a_spoofed_tool_call_argument(db, make_vendor, make_po):
    """guard_node must always use the authenticated caller's identity, never
    whatever approved_by value the LLM put in the tool call — otherwise a
    user could ask the agent to approve "as" someone else and bypass the
    self-approval guard."""
    vendor_id = make_vendor("Acme Corp")
    make_po("PO-SPOOF", vendor_id, amount=5_000.0, status="pending", created_at=None)

    llm = FakeLLM(
        [
            tool_call_message(
                "approve_purchase_order",
                {"po_number": "PO-SPOOF", "approved_by": "someone-else@company.com"},
            )
        ]
    )
    build_graph(llm).invoke(initial_state("Approve PO-SPOOF", identity="real-caller@company.com"))

    row = db.execute(
        "SELECT approved_by FROM purchase_orders WHERE po_number = 'PO-SPOOF'"
    ).fetchone()
    assert row["approved_by"] == "real-caller@company.com"


def test_over_threshold_po_is_approved_but_carries_a_warning(db, make_vendor, make_po):
    """A warn-level guard escalates, it does not stop the approval."""
    vendor_id = make_vendor("Acme Corp")
    make_po("PO-BIG", vendor_id, amount=90_000.0, status="pending")

    llm = FakeLLM(
        [
            tool_call_message(
                "approve_purchase_order",
                {"po_number": "PO-BIG", "approved_by": "cfo@company.com"},
            )
        ]
    )
    result = build_graph(llm).invoke(initial_state("Approve PO-BIG"))

    assert any("exceeds" in out for out in tool_outputs(result))
    status = db.execute(
        "SELECT status FROM purchase_orders WHERE po_number = 'PO-BIG'"
    ).fetchone()["status"]
    assert status == "approved"


def test_guard_failures_surface_as_risk_flags(db, make_vendor, make_po):
    vendor_id = make_vendor("ShadyDeals LLC", status="blacklisted")
    make_po("PO-BAD", vendor_id, amount=90_000.0, status="pending")

    llm = FakeLLM([tool_call_message("approve_purchase_order", {"po_number": "PO-BAD"})])
    result = build_graph(llm).invoke(initial_state("Approve PO-BAD"))

    assert len(result["risk_flags"]) >= 2


def test_unguarded_tool_runs_and_loops_back_to_the_agent(db, make_vendor):
    make_vendor("Acme Corp")

    llm = FakeLLM(
        [
            tool_call_message("lookup_vendor", {"name": "Acme"}),
            AIMessage(content="Acme Corp is an active vendor."),
        ]
    )
    result = build_graph(llm).invoke(initial_state("Look up Acme"))

    assert any("Acme Corp" in out for out in tool_outputs(result))
    assert result["messages"][-1].content == "Acme Corp is an active vendor."


def test_system_prompt_is_injected_once(db):
    llm = FakeLLM(
        [
            tool_call_message("lookup_vendor", {"name": "Acme"}),
            AIMessage(content="Done."),
        ]
    )
    build_graph(llm).invoke(initial_state("Look up Acme"))

    for sent in llm.calls:
        assert sum(isinstance(m, SystemMessage) for m in sent) == 1


def test_unknown_tool_name_does_not_crash_the_graph(db):
    llm = FakeLLM(
        [
            tool_call_message("delete_all_vendors", {}),
            AIMessage(content="I cannot do that."),
        ]
    )
    result = build_graph(llm).invoke(initial_state("Delete everything"))

    assert any("Unknown tool" in out for out in tool_outputs(result))
