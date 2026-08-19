"""
tests/test_models.py
──────────────────────
Validation tests for api/models.py's pydantic schemas — the request-side
constraints (required fields, min_length) and default_factory behavior on
response models.
"""

import pytest
from pydantic import ValidationError

from api.models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    InvoiceOut,
    PurchaseOrderOut,
    VendorOut,
)


# ── ChatRequest ────────────────────────────────────────────────────────────────

def test_chat_request_accepts_valid_input():
    req = ChatRequest(session_id="s1", message="hello")
    assert req.session_id == "s1"
    assert req.message == "hello"


def test_chat_request_rejects_empty_message():
    with pytest.raises(ValidationError):
        ChatRequest(session_id="s1", message="")


def test_chat_request_requires_session_id():
    with pytest.raises(ValidationError):
        ChatRequest(message="hello")


def test_chat_request_requires_message():
    with pytest.raises(ValidationError):
        ChatRequest(session_id="s1")


def test_chat_request_has_no_role_field():
    """role was removed — it's resolved server-side from the API key now,
    not trusted from the client. A stray role in the payload is just an
    unused extra field, not a way to set it (pydantic ignores extras by
    default)."""
    req = ChatRequest(session_id="s1", message="hi", role="admin")
    assert not hasattr(req, "role")


# ── ChatResponse ───────────────────────────────────────────────────────────────

def test_chat_response_defaults_lists_to_empty():
    resp = ChatResponse(session_id="s1", reply="hi")
    assert resp.risk_flags == []
    assert resp.tool_calls_made == []


def test_chat_response_default_lists_are_independent_per_instance():
    """default_factory=list must not share a single mutable default across
    instances — mutating one response's list shouldn't leak into another's."""
    a = ChatResponse(session_id="a", reply="x")
    b = ChatResponse(session_id="b", reply="y")
    a.risk_flags.append("flag")
    assert b.risk_flags == []


# ── VendorOut / PurchaseOrderOut / InvoiceOut ───────────────────────────────────

def test_vendor_out_allows_null_optional_fields():
    v = VendorOut(
        id=1, name="Acme", contact=None, email=None, phone=None, address=None,
        status="active", created_at="2025-01-01", updated_at="2025-01-01",
    )
    assert v.contact is None


def test_vendor_out_requires_name_and_status():
    with pytest.raises(ValidationError):
        VendorOut(
            id=1, contact=None, email=None, phone=None, address=None,
            status="active", created_at="2025-01-01", updated_at="2025-01-01",
        )


def test_purchase_order_out_requires_numeric_amount():
    with pytest.raises(ValidationError):
        PurchaseOrderOut(
            id=1, po_number="PO-1", vendor_id=1, vendor_name="Acme",
            description=None, amount="not-a-number", currency="USD",
            status="pending", requested_by=None, approved_by=None,
            created_at="2025-01-01", updated_at="2025-01-01",
        )


def test_invoice_out_allows_null_paid_date():
    inv = InvoiceOut(
        id=1, invoice_number="INV-1", po_id=1, po_number="PO-1", vendor_id=1,
        vendor_name="Acme", amount=100.0, currency="USD", due_date="2025-01-01",
        paid_date=None, status="pending", created_at="2025-01-01", updated_at="2025-01-01",
    )
    assert inv.paid_date is None


# ── HealthResponse ─────────────────────────────────────────────────────────────

def test_health_response_defaults_status_to_ok():
    h = HealthResponse(model_name="demo", db_path="data/procurement.db", active_sessions=0)
    assert h.status == "ok"
