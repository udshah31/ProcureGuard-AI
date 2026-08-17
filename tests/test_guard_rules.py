"""
tests/test_guard_rules.py
─────────────────────────
Unit tests for the compliance guard layer.

The guards are the part of this system that must never regress: a false
negative here means a blacklisted vendor gets paid. Each check is tested at
its boundary, then run_all_guards is tested as an orchestrator.
"""

import pytest

from agent import guard_rules
from agent.guard_rules import (
    check_amount_threshold,
    check_duplicate_po,
    check_self_approval,
    check_vendor_status,
    format_guard_summary,
    run_all_guards,
)


# ── check_amount_threshold ────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "amount, should_pass",
    [
        (0.0, True),
        (49_999.99, True),
        (50_000.0, True),        # threshold is exclusive
        (50_000.01, False),
        (1_000_000.0, False),
    ],
)
def test_amount_threshold_boundary(amount, should_pass):
    result = check_amount_threshold(amount)
    assert result.passed is should_pass
    assert result.check == "amount_threshold"


def test_amount_threshold_warns_never_blocks():
    """A large amount escalates to finance, it does not hard-block approval."""
    assert check_amount_threshold(500_000.0).severity == "warn"


def test_amount_threshold_reads_configured_limit(monkeypatch):
    monkeypatch.setattr(guard_rules, "AMOUNT_THRESHOLD", 100.0)
    assert check_amount_threshold(150.0).passed is False
    assert check_amount_threshold(50.0).passed is True


# ── check_vendor_status ───────────────────────────────────────────────────────

def test_blacklisted_vendor_blocks(make_vendor):
    vendor_id = make_vendor("ShadyDeals LLC", status="blacklisted")
    result = check_vendor_status(vendor_id)

    assert result.passed is False
    assert result.severity == "block"
    assert "ShadyDeals LLC" in result.message


def test_inactive_vendor_warns_but_does_not_block(make_vendor):
    vendor_id = make_vendor("OldParts Co", status="inactive")
    result = check_vendor_status(vendor_id)

    assert result.passed is False
    assert result.severity == "warn"


def test_active_vendor_passes(make_vendor):
    result = check_vendor_status(make_vendor("Acme Corp"))
    assert result.passed is True


def test_unknown_vendor_blocks(db):
    """A dangling vendor_id must fail closed, not fail open."""
    result = check_vendor_status(9999)
    assert result.passed is False
    assert result.severity == "block"


# ── check_duplicate_po ────────────────────────────────────────────────────────

def test_duplicate_detected_within_amount_band(make_vendor, make_po):
    vendor_id = make_vendor("Acme Corp")
    make_po("PO-1", vendor_id, amount=10_000.0)

    result = check_duplicate_po(vendor_id, amount=10_500.0)  # +5%

    assert result.passed is False
    assert result.severity == "warn"
    assert "PO-1" in result.message


def test_amount_outside_twenty_percent_band_is_not_duplicate(make_vendor, make_po):
    vendor_id = make_vendor("Acme Corp")
    make_po("PO-1", vendor_id, amount=10_000.0)

    assert check_duplicate_po(vendor_id, amount=13_000.0).passed is True  # +30%


def test_duplicate_outside_time_window_ignored(make_vendor, make_po):
    vendor_id = make_vendor("Acme Corp")
    make_po("PO-OLD", vendor_id, amount=10_000.0, created_at="2020-01-01 00:00:00")

    assert check_duplicate_po(vendor_id, amount=10_000.0).passed is True


def test_rejected_and_closed_pos_are_not_duplicates(make_vendor, make_po):
    vendor_id = make_vendor("Acme Corp")
    make_po("PO-R", vendor_id, amount=10_000.0, status="rejected")
    make_po("PO-C", vendor_id, amount=10_000.0, status="closed")

    assert check_duplicate_po(vendor_id, amount=10_000.0).passed is True


def test_duplicates_are_scoped_to_one_vendor(make_vendor, make_po):
    acme = make_vendor("Acme Corp")
    tech = make_vendor("TechSupply Inc")
    make_po("PO-1", acme, amount=10_000.0)

    assert check_duplicate_po(tech, amount=10_000.0).passed is True


def test_po_under_review_is_not_its_own_duplicate(make_vendor, make_po):
    """Regression: without exclude_po_id every PO matched itself."""
    vendor_id = make_vendor("Acme Corp")
    po_id = make_po("PO-1", vendor_id, amount=10_000.0)

    assert check_duplicate_po(vendor_id, 10_000.0, exclude_po_id=po_id).passed is True
    assert check_duplicate_po(vendor_id, 10_000.0).passed is False


# ── run_all_guards ────────────────────────────────────────────────────────────

def test_missing_po_returns_single_blocking_result(db):
    results = run_all_guards("PO-DOES-NOT-EXIST")

    assert len(results) == 1
    assert results[0].check == "po_lookup"
    assert results[0].severity == "block"


@pytest.mark.parametrize("typed", ["po-clean", "PO-clean", "  PO-CLEAN  "])
def test_po_number_lookup_is_case_and_whitespace_insensitive(
    make_vendor, make_po, typed
):
    """Regression: SQLite '=' is case-sensitive, so 'po-clean' used to read as a
    missing PO and hard-block a legitimate approval."""
    vendor_id = make_vendor("Acme Corp")
    make_po("PO-CLEAN", vendor_id, amount=5_000.0)

    results = run_all_guards(typed)

    assert [r.check for r in results] != ["po_lookup"]
    assert all(r.passed for r in results)


def test_clean_po_passes_every_check(make_vendor, make_po):
    vendor_id = make_vendor("Acme Corp")
    make_po("PO-CLEAN", vendor_id, amount=5_000.0)

    results = run_all_guards("PO-CLEAN")

    assert len(results) == 3
    assert all(r.passed for r in results)


def test_all_three_checks_can_fail_together(make_vendor, make_po):
    vendor_id = make_vendor("ShadyDeals LLC", status="blacklisted")
    make_po("PO-DUP", vendor_id, amount=80_000.0)
    make_po("PO-BAD", vendor_id, amount=80_000.0)

    failed = {r.check for r in run_all_guards("PO-BAD") if not r.passed}

    assert failed == {"amount_threshold", "vendor_status", "duplicate_po"}


def test_blacklist_blocks_even_when_amount_is_small(make_vendor, make_po):
    """Severity must come from the vendor check, not be diluted by warnings."""
    vendor_id = make_vendor("ShadyDeals LLC", status="blacklisted")
    make_po("PO-SMALL", vendor_id, amount=10.0)

    results = run_all_guards("PO-SMALL")
    blocks = [r for r in results if not r.passed and r.severity == "block"]

    assert len(blocks) == 1


# ── check_self_approval ───────────────────────────────────────────────────────

def test_approving_your_own_po_is_blocked():
    result = check_self_approval("john.doe@company.com", "john.doe@company.com")

    assert result.passed is False
    assert result.severity == "block"


@pytest.mark.parametrize(
    "requester, approver",
    [
        ("John.Doe@company.com", "john.doe@company.com"),   # casing
        ("  john.doe@company.com  ", "john.doe@company.com"),  # padding
    ],
)
def test_self_approval_check_is_not_fooled_by_formatting(requester, approver):
    assert check_self_approval(requester, approver).passed is False


def test_different_people_pass():
    assert check_self_approval("john.doe@company.com", "manager@company.com").passed is True


@pytest.mark.parametrize("requester", [None, "", "   "])
def test_unknown_requester_does_not_block(requester):
    """Legacy rows have no requester; don't block on data we never captured."""
    assert check_self_approval(requester, "manager@company.com").passed is True


def test_self_approval_is_enforced_end_to_end(make_vendor, db):
    vendor_id = make_vendor("Acme Corp")
    db.execute(
        """
        INSERT INTO purchase_orders (po_number, vendor_id, amount, status, requested_by)
        VALUES ('PO-SELF', ?, 1000.0, 'pending', 'john.doe@company.com')
        """,
        (vendor_id,),
    )
    db.commit()

    results = run_all_guards("PO-SELF", approved_by="john.doe@company.com")
    blocks = [r for r in results if not r.passed and r.severity == "block"]

    assert [r.check for r in blocks] == ["self_approval"]


def test_self_approval_check_is_skipped_when_no_approver_given(make_vendor, db):
    vendor_id = make_vendor("Acme Corp")
    db.execute(
        """
        INSERT INTO purchase_orders (po_number, vendor_id, amount, status, requested_by)
        VALUES ('PO-SELF', ?, 1000.0, 'pending', 'john.doe@company.com')
        """,
        (vendor_id,),
    )
    db.commit()

    checks = {r.check for r in run_all_guards("PO-SELF")}

    assert "self_approval" not in checks


# ── format_guard_summary ──────────────────────────────────────────────────────

def test_summary_reports_all_clear_when_nothing_failed(make_vendor, make_po):
    vendor_id = make_vendor("Acme Corp")
    make_po("PO-CLEAN", vendor_id, amount=100.0)

    summary = format_guard_summary(run_all_guards("PO-CLEAN"))

    assert "All compliance checks passed" in summary


def test_summary_states_approval_is_barred_when_blocked(make_vendor, make_po):
    vendor_id = make_vendor("ShadyDeals LLC", status="blacklisted")
    make_po("PO-BAD", vendor_id, amount=100.0)

    summary = format_guard_summary(run_all_guards("PO-BAD"))

    assert "BLOCKED" in summary
    assert "cannot be approved" in summary


def test_summary_allows_progress_when_only_warnings(make_vendor, make_po):
    vendor_id = make_vendor("Acme Corp")
    make_po("PO-BIG", vendor_id, amount=90_000.0)

    summary = format_guard_summary(run_all_guards("PO-BIG"))

    assert "exceeds" in summary
    assert "No blocking issues" in summary


def test_summary_does_not_double_prefix_severity(make_vendor, make_po):
    vendor_id = make_vendor("ShadyDeals LLC", status="blacklisted")
    make_po("PO-BAD", vendor_id, amount=100.0)

    summary = format_guard_summary(run_all_guards("PO-BAD"))

    assert "BLOCKED — 🚫 BLOCKED" not in summary
