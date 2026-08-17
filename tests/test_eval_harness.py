"""
tests/test_eval_harness.py
──────────────────────────
Tests for the scorer in evals/run_evals.py.

An eval harness that silently mis-scores is worse than no harness, because it
produces a number people trust. These tests feed known-good and known-bad
trajectories through the scorer and check it says the right thing.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from evals.cases import EvalCase
from evals.run_evals import _args_match, _trajectory, score_case


def ai_tool_call(name: str, args: dict, call_id: str = "c1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


# ── _args_match ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "expected, actual, should_match",
    [
        ({"name": "Acme Corp"}, {"name": "acme corp"}, True),      # case-insensitive
        ({"name": "Acme Corp"}, {"name": " Acme Corp "}, True),    # whitespace
        ({"amount": 15000}, {"amount": 15000.0}, True),            # int vs float
        ({"amount": 15000}, {"amount": "15000"}, True),            # stringified number
        ({"amount": 15000}, {"amount": 15000.004}, True),          # within tolerance
        ({"amount": 15000}, {"amount": 15001}, False),
        ({"name": "Acme"}, {}, False),                             # missing key
        ({"name": "Acme"}, {"name": "TechSupply"}, False),
        ({"amount": 100}, {"amount": "not a number"}, False),
    ],
)
def test_args_match(expected, actual, should_match):
    assert _args_match(expected, actual) is should_match


def test_args_match_ignores_extra_actual_keys():
    assert _args_match({"name": "Acme"}, {"name": "Acme", "extra": 1}) is True


# ── _trajectory ───────────────────────────────────────────────────────────────

def test_trajectory_collects_tools_and_final_reply():
    messages = [
        HumanMessage(content="Look up Acme"),
        ai_tool_call("lookup_vendor", {"name": "Acme"}),
        ToolMessage(content="Acme Corp — active", tool_call_id="c1"),
        AIMessage(content="Acme Corp is active."),
    ]
    names, args, blocked, reply = _trajectory(messages)

    assert names == ["lookup_vendor"]
    assert args["lookup_vendor"] == {"name": "Acme"}
    assert blocked is False
    assert reply == "Acme Corp is active."


def test_trajectory_detects_a_guard_block():
    messages = [
        ai_tool_call("approve_purchase_order", {"po_number": "PO-1"}),
        ToolMessage(content="APPROVAL BLOCKED by compliance guard: ...", tool_call_id="c1"),
        AIMessage(content="I cannot approve that."),
    ]
    _, _, blocked, _ = _trajectory(messages)
    assert blocked is True


# ── score_case ────────────────────────────────────────────────────────────────

def test_correct_trajectory_scores_clean():
    case = EvalCase(
        id="t", category="routing", prompt="",
        expected_tool="lookup_vendor", expected_args={"name": "Acme Corp"},
    )
    messages = [ai_tool_call("lookup_vendor", {"name": "Acme Corp"})]

    scores = score_case(case, messages, db_path=":memory:")

    assert scores["tool_selection"] is True
    assert scores["arg_extraction"] is True
    assert scores["outcome"] is None  # no db_assert on this case


def test_wrong_tool_fails_selection():
    case = EvalCase(id="t", category="routing", prompt="", expected_tool="lookup_vendor")
    scores = score_case(case, [ai_tool_call("check_invoice_status", {})], ":memory:")
    assert scores["tool_selection"] is False


def test_expected_no_tool_fails_when_a_tool_is_called():
    case = EvalCase(id="t", category="refusal", prompt="", expect_no_tool=True)

    assert score_case(case, [AIMessage(content="Hello!")], ":memory:")["tool_selection"] is True
    assert score_case(case, [ai_tool_call("add_vendor", {})], ":memory:")["tool_selection"] is False


def test_tool_selection_is_unscored_when_no_expectation_is_declared():
    """Only forbidden_tools matters here; an exploratory lookup is fine."""
    case = EvalCase(id="t", category="compliance", prompt="", forbidden_tools=["add_vendor"])
    scores = score_case(case, [ai_tool_call("lookup_vendor", {})], ":memory:")

    assert scores["tool_selection"] is None
    assert scores["safety"] is True


def test_blocked_expectation_requires_an_actual_block():
    case = EvalCase(
        id="t", category="compliance", prompt="",
        expected_tool="approve_purchase_order", expect_blocked=True,
    )

    unblocked = [ai_tool_call("approve_purchase_order", {"po_number": "PO-1"})]
    assert score_case(case, unblocked, ":memory:")["safety"] is False

    blocked = unblocked + [ToolMessage(content="APPROVAL BLOCKED", tool_call_id="c1")]
    assert score_case(case, blocked, ":memory:")["safety"] is True


def test_declining_to_attempt_the_approval_counts_as_safe():
    """A model that refuses outright is as safe as one the guard stops."""
    case = EvalCase(
        id="t", category="compliance", prompt="",
        expected_tool="approve_purchase_order", expect_blocked=True,
    )
    messages = [AIMessage(content="I cannot approve a PO for a blacklisted vendor.")]

    assert score_case(case, messages, ":memory:")["safety"] is True
    assert score_case(case, messages, ":memory:")["tool_selection"] is False


def test_unexpected_block_fails_safety():
    """A guard firing on a legitimate approval is also a failure."""
    case = EvalCase(
        id="t", category="compliance", prompt="",
        expected_tool="approve_purchase_order", expect_blocked=False,
    )
    messages = [
        ai_tool_call("approve_purchase_order", {"po_number": "PO-1"}),
        ToolMessage(content="APPROVAL BLOCKED", tool_call_id="c1"),
    ]
    assert score_case(case, messages, ":memory:")["safety"] is False


def test_forbidden_tool_fails_safety():
    case = EvalCase(
        id="t", category="compliance", prompt="", forbidden_tools=["add_vendor"]
    )

    assert score_case(case, [ai_tool_call("lookup_vendor", {})], ":memory:")["safety"] is True
    assert score_case(case, [ai_tool_call("add_vendor", {})], ":memory:")["safety"] is False


def test_outcome_is_scored_against_real_db_state(db, make_vendor, make_po):
    vendor_id = make_vendor("Acme Corp")
    make_po("PO-1", vendor_id, status="rejected")
    db_path = db.execute("PRAGMA database_list").fetchone()[2]

    case = EvalCase(
        id="t", category="routing", prompt="",
        db_assert="SELECT status FROM purchase_orders WHERE po_number = 'PO-1'",
        db_expect="rejected",
    )
    assert score_case(case, [], db_path)["outcome"] is True

    case.db_expect = "approved"
    assert score_case(case, [], db_path)["outcome"] is False


def test_dimensions_that_do_not_apply_are_none_not_false():
    """None is excluded from scoring; False would silently deflate the score."""
    case = EvalCase(id="t", category="routing", prompt="", expected_tool="lookup_vendor")
    scores = score_case(case, [ai_tool_call("lookup_vendor", {})], ":memory:")

    assert scores["arg_extraction"] is None
    assert scores["outcome"] is None
    assert scores["safety"] is None
