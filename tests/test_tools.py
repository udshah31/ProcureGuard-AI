"""
tests/test_tools.py
───────────────────
Tests for the domain tools' handling of user-supplied identifiers.

PO and invoice numbers arrive from an LLM parsing free-form user text, so they
turn up lowercased, padded, or both. SQLite's '=' is case-sensitive, which
turned any of those into a spurious "not found".
"""

import pytest

from agent.tools import (
    approve_purchase_order,
    check_invoice_status,
    create_purchase_order,
    flag_suspicious_invoice,
    reject_purchase_order,
)


@pytest.fixture
def invoice(db, make_vendor, make_po):
    vendor_id = make_vendor("Acme Corp")
    po_id = make_po("PO-100001", vendor_id, amount=1_000.0)
    db.execute(
        """
        INSERT INTO invoices (invoice_number, po_id, vendor_id, amount, due_date, status)
        VALUES ('INV-2025-001', ?, ?, 1000.0, '2025-12-01', 'pending')
        """,
        (po_id, vendor_id),
    )
    db.commit()


@pytest.mark.parametrize("typed", ["PO-100002", "po-100002", "  po-100002  "])
def test_approve_accepts_any_casing(db, make_vendor, make_po, typed):
    vendor_id = make_vendor("Acme Corp")
    make_po("PO-100002", vendor_id, amount=1_000.0, status="pending")

    result = approve_purchase_order.invoke(
        {"po_number": typed, "approved_by": "manager@company.com"}
    )

    assert "not found" not in result.lower()
    status = db.execute(
        "SELECT status FROM purchase_orders WHERE po_number = 'PO-100002'"
    ).fetchone()["status"]
    assert status == "approved"


@pytest.mark.parametrize("typed", ["PO-100003", "po-100003"])
def test_reject_accepts_any_casing(db, make_vendor, make_po, typed):
    vendor_id = make_vendor("Acme Corp")
    make_po("PO-100003", vendor_id, status="pending")

    result = reject_purchase_order.invoke(
        {"po_number": typed, "rejected_by": "ops@company.com", "reason": "budget cut"}
    )

    assert "not found" not in result.lower()
    status = db.execute(
        "SELECT status FROM purchase_orders WHERE po_number = 'PO-100003'"
    ).fetchone()["status"]
    assert status == "rejected"


@pytest.mark.parametrize("typed", ["INV-2025-001", "inv-2025-001"])
def test_invoice_lookup_accepts_any_casing(invoice, typed):
    result = check_invoice_status.invoke({"invoice_number": typed})
    assert "not found" not in result.lower()


@pytest.mark.parametrize("typed", ["INV-2025-001", "inv-2025-001"])
def test_flagging_an_invoice_accepts_any_casing(db, invoice, typed):
    result = flag_suspicious_invoice.invoke(
        {"invoice_number": typed, "flagged_by": "finance@company.com", "reason": "overbilled"}
    )

    assert "not found" not in result.lower()
    status = db.execute(
        "SELECT status FROM invoices WHERE invoice_number = 'INV-2025-001'"
    ).fetchone()["status"]
    assert status == "disputed"


def test_genuinely_missing_po_still_reports_not_found(db):
    result = approve_purchase_order.invoke(
        {"po_number": "PO-000000", "approved_by": "manager@company.com"}
    )
    assert "not found" in result.lower()


def test_created_po_is_audited_against_its_own_id(db, make_vendor):
    """Regression: the audit row used to store the vendor's id, so the
    compliance trail pointed at the wrong entity."""
    make_vendor("Acme Corp")

    create_purchase_order.invoke(
        {
            "vendor_name": "Acme Corp",
            "description": "Office chairs",
            "amount": 2_500.0,
            "requested_by": "hr@company.com",
        }
    )

    audit = db.execute(
        "SELECT entity_id FROM audit_log WHERE entity_type = 'purchase_order' AND action = 'created'"
    ).fetchone()
    po_id = db.execute("SELECT id FROM purchase_orders").fetchone()["id"]

    assert audit["entity_id"] == po_id
