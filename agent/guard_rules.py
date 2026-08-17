"""
agent/guard_rules.py
─────────────────────
Compliance guard layer — runs BEFORE any purchase order approval.

Each check returns a GuardResult(passed, message).
The agent collects failures and either warns the user or blocks the action
depending on severity.

Checks:
    check_amount_threshold    – POs > $50,000 require finance sign-off
    check_vendor_status       – Blocked if vendor is blacklisted
    check_duplicate_po        – Warn if similar PO exists within last 30 days

Usage:
    results = run_all_guards(po_number="PO-123456")
    failures = [r for r in results if not r.passed]
"""

import json
import logging
import os
import sqlite3
from dataclasses import dataclass

log = logging.getLogger(__name__)

DB_PATH: str = os.getenv("DB_PATH", "data/procurement.db")

AMOUNT_THRESHOLD: float = float(os.getenv("GUARD_AMOUNT_THRESHOLD", "50000"))
DUPLICATE_WINDOW_DAYS: int = int(os.getenv("GUARD_DUPLICATE_WINDOW_DAYS", "30"))


@dataclass
class GuardResult:
    check: str          # name of the check
    passed: bool        # True = no issue, False = violation / warning
    severity: str       # "block" | "warn"
    message: str        # human-readable description


def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _norm_id(value: str) -> str:
    """Match the normalisation in agent/tools.py — SQLite '=' is case-sensitive."""
    return (value or "").strip().upper()


# ── Individual checks ─────────────────────────────────────────────────────────

def check_amount_threshold(amount: float) -> GuardResult:
    """
    Flag POs that exceed the single-approval threshold ($50k default).
    These require an approver with 'finance' or 'admin' role.
    """
    if amount > AMOUNT_THRESHOLD:
        return GuardResult(
            check="amount_threshold",
            passed=False,
            severity="warn",
            message=(
                f"⚠️  PO amount ${amount:,.2f} exceeds the ${AMOUNT_THRESHOLD:,.0f} threshold. "
                "Finance sign-off is required before approval."
            ),
        )
    return GuardResult(
        check="amount_threshold",
        passed=True,
        severity="warn",
        message=f"Amount ${amount:,.2f} is within the approved threshold.",
    )


def check_vendor_status(vendor_id: int) -> GuardResult:
    """
    Block approval if the vendor is blacklisted; warn if they are inactive.
    """
    with _get_db() as conn:
        vendor = conn.execute(
            "SELECT name, status FROM vendors WHERE id = ?", (vendor_id,)
        ).fetchone()

    if not vendor:
        return GuardResult(
            check="vendor_status",
            passed=False,
            severity="block",
            message="🚫 Vendor not found. Cannot approve PO for unknown vendor.",
        )

    if vendor["status"] == "blacklisted":
        return GuardResult(
            check="vendor_status",
            passed=False,
            severity="block",
            message=(
                f"🚫 BLOCKED: Vendor '{vendor['name']}' is BLACKLISTED. "
                "This PO cannot be approved."
            ),
        )

    if vendor["status"] == "inactive":
        return GuardResult(
            check="vendor_status",
            passed=False,
            severity="warn",
            message=(
                f"⚠️  Vendor '{vendor['name']}' is currently INACTIVE. "
                "Verify this is intentional before approving."
            ),
        )

    return GuardResult(
        check="vendor_status",
        passed=True,
        severity="warn",
        message=f"Vendor '{vendor['name']}' is active — no issues.",
    )


def check_duplicate_po(
    vendor_id: int, amount: float, exclude_po_id: int | None = None
) -> GuardResult:
    """
    Warn if a PO of similar amount (±20%) exists for this vendor
    within the last N days (default 30). Helps detect duplicate submissions.

    exclude_po_id omits the PO under review so it cannot match itself.
    """
    lower = amount * 0.80
    upper = amount * 1.20

    with _get_db() as conn:
        duplicate = conn.execute(
            """
            SELECT po_number, amount, created_at
            FROM purchase_orders
            WHERE vendor_id = ?
              AND amount BETWEEN ? AND ?
              AND status NOT IN ('rejected', 'closed')
              AND created_at >= datetime('now', ? || ' days')
              AND id IS NOT ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (vendor_id, lower, upper, f"-{DUPLICATE_WINDOW_DAYS}", exclude_po_id),
        ).fetchone()

    if duplicate:
        return GuardResult(
            check="duplicate_po",
            passed=False,
            severity="warn",
            message=(
                f"⚠️  Possible duplicate: PO {duplicate['po_number']} "
                f"(${duplicate['amount']:,.2f}) was submitted for this vendor "
                f"on {duplicate['created_at'][:10]} — within the last "
                f"{DUPLICATE_WINDOW_DAYS} days."
            ),
        )

    return GuardResult(
        check="duplicate_po",
        passed=True,
        severity="warn",
        message="No duplicate POs detected in the review window.",
    )


def check_self_approval(requested_by: str | None, approved_by: str) -> GuardResult:
    """
    Block a requester from approving their own PO (segregation of duties).

    This is the control most procurement fraud depends on, so it blocks rather
    than warns.
    """
    requester = (requested_by or "").strip().lower()
    approver = (approved_by or "").strip().lower()

    if requester and approver and requester == approver:
        return GuardResult(
            check="self_approval",
            passed=False,
            severity="block",
            message=(
                f"🚫 BLOCKED: '{approved_by}' raised this PO and cannot also "
                "approve it. Segregation of duties requires a second approver."
            ),
        )

    return GuardResult(
        check="self_approval",
        passed=True,
        severity="block",
        message="Requester and approver are different people.",
    )


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_all_guards(po_number: str, approved_by: str | None = None) -> list[GuardResult]:
    """
    Run all guard checks for a given PO number.
    Returns a list of GuardResult objects — callers should check for failures.

    approved_by enables the segregation-of-duties check; without it that check
    is skipped rather than passed, so a missing approver can't look like a clean
    result.

    Example:
        results = run_all_guards("PO-123456", approved_by="manager@company.com")
        blocks  = [r for r in results if not r.passed and r.severity == "block"]
        warns   = [r for r in results if not r.passed and r.severity == "warn"]
    """
    with _get_db() as conn:
        po = conn.execute(
            """
            SELECT id, amount, vendor_id, requested_by
            FROM purchase_orders WHERE po_number = ?
            """,
            (_norm_id(po_number),),
        ).fetchone()

    if not po:
        return [
            GuardResult(
                check="po_lookup",
                passed=False,
                severity="block",
                message=f"🚫 PO '{po_number}' not found. Cannot run guard checks.",
            )
        ]

    results = [
        check_amount_threshold(po["amount"]),
        check_vendor_status(po["vendor_id"]),
        check_duplicate_po(po["vendor_id"], po["amount"], exclude_po_id=po["id"]),
    ]

    if approved_by:
        results.append(check_self_approval(po["requested_by"], approved_by))

    for r in results:
        if not r.passed:
            log.warning("Guard [%s] %s: %s", r.severity.upper(), r.check, r.message)

    return results


def format_guard_summary(results: list[GuardResult]) -> str:
    """
    Format guard results into a human-readable summary for the agent to include
    in its response.
    """
    failures = [r for r in results if not r.passed]
    if not failures:
        return "✅ All compliance checks passed."

    blocks = [r for r in failures if r.severity == "block"]
    warns  = [r for r in failures if r.severity == "warn"]

    lines = ["**Compliance Check Results:**"]
    # Messages already carry their own severity marker, so don't re-prefix.
    for r in blocks:
        lines.append(f"- {r.message}")
    for r in warns:
        lines.append(f"- {r.message}")

    if blocks:
        lines.append("\n**This PO cannot be approved until blocking issues are resolved.**")
    else:
        lines.append("\nNo blocking issues — approval may proceed with awareness of warnings.")

    return "\n".join(lines)
