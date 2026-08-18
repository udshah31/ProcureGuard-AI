"""
tests/test_routers.py
──────────────────────
HTTP-level tests for the read-only REST routers: vendors, purchase_orders,
invoices. Each router keeps its own module-level DB_PATH (read from the
DB_PATH env var at import time), so the fixture patches those directly in
addition to setting DB_PATH — reloading server_demo alone re-reads env for
server_demo's own constant, but not for already-imported router modules.

Fixture data is inserted directly rather than relying on db/seed.py, so
these tests don't break if the seed dataset changes shape.
"""

import sqlite3

import pytest

from db.init_db import init_db

AUTH_HEADERS = {"X-API-Key": "test-key"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    db_path = str(tmp_path / "routers_test.db")
    monkeypatch.setenv("DB_PATH", db_path)
    monkeypatch.setenv("API_KEYS", "test-key:admin")

    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO vendors (name, contact, email, phone, address, status) "
            "VALUES ('Acme Corp', 'Jane Doe', 'jane@acme.com', '555-1000', '1 Main St', 'active')"
        )
        conn.execute(
            "INSERT INTO vendors (name, status) VALUES ('TechSupply Inc', 'active')"
        )
        conn.execute(
            "INSERT INTO vendors (name, status) VALUES ('ShadyDeals LLC', 'blacklisted')"
        )
        acme_id = conn.execute("SELECT id FROM vendors WHERE name='Acme Corp'").fetchone()[0]
        tech_id = conn.execute("SELECT id FROM vendors WHERE name='TechSupply Inc'").fetchone()[0]
        shady_id = conn.execute("SELECT id FROM vendors WHERE name='ShadyDeals LLC'").fetchone()[0]

        conn.execute(
            "INSERT INTO purchase_orders "
            "(po_number, vendor_id, description, amount, status, requested_by, approved_by) "
            "VALUES ('PO-100001', ?, 'Office supplies', 12500.0, 'approved', "
            "'john@company.com', 'manager@company.com')",
            (acme_id,),
        )
        conn.execute(
            "INSERT INTO purchase_orders "
            "(po_number, vendor_id, description, amount, status, requested_by) "
            "VALUES ('PO-100002', ?, 'Server hardware', 87500.0, 'pending', 'it@company.com')",
            (tech_id,),
        )
        conn.execute(
            "INSERT INTO purchase_orders "
            "(po_number, vendor_id, description, amount, status, requested_by) "
            "VALUES ('PO-100003', ?, 'Consulting retainer', 5000.0, 'pending', 'ops@company.com')",
            (shady_id,),
        )
        po1_id = conn.execute("SELECT id FROM purchase_orders WHERE po_number='PO-100001'").fetchone()[0]
        po2_id = conn.execute("SELECT id FROM purchase_orders WHERE po_number='PO-100002'").fetchone()[0]
        po3_id = conn.execute("SELECT id FROM purchase_orders WHERE po_number='PO-100003'").fetchone()[0]

        conn.execute(
            "INSERT INTO invoices (invoice_number, po_id, vendor_id, amount, status, due_date, paid_date) "
            "VALUES ('INV-2025-001', ?, ?, 12500.0, 'paid', '2025-01-01', '2025-01-02')",
            (po1_id, acme_id),
        )
        conn.execute(
            "INSERT INTO invoices (invoice_number, po_id, vendor_id, amount, status, due_date) "
            "VALUES ('INV-2025-002', ?, ?, 87500.0, 'pending', '2020-01-01')",
            (po2_id, tech_id),
        )
        conn.execute(
            "INSERT INTO invoices (invoice_number, po_id, vendor_id, amount, status, due_date) "
            "VALUES ('INV-2025-003', ?, ?, 5000.0, 'disputed', '2099-01-01')",
            (po3_id, shady_id),
        )
        conn.commit()

    import importlib

    import api.routers.invoices as invoices_router
    import api.routers.purchase_orders as po_router
    import api.routers.vendors as vendors_router
    import api.server_demo as server_demo

    importlib.reload(server_demo)
    monkeypatch.setattr(vendors_router, "DB_PATH", db_path)
    monkeypatch.setattr(po_router, "DB_PATH", db_path)
    monkeypatch.setattr(invoices_router, "DB_PATH", db_path)

    with TestClient(server_demo.app) as c:
        yield c


# ── /vendors ───────────────────────────────────────────────────────────────────

def test_list_vendors_returns_all(client):
    resp = client.get("/api/v1/vendors", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    names = {v["name"] for v in resp.json()}
    assert names == {"Acme Corp", "TechSupply Inc", "ShadyDeals LLC"}


def test_list_vendors_filters_by_status(client):
    resp = client.get("/api/v1/vendors", params={"status": "blacklisted"}, headers=AUTH_HEADERS)
    assert resp.status_code == 200
    vendors = resp.json()
    assert len(vendors) == 1
    assert vendors[0]["name"] == "ShadyDeals LLC"


def test_list_vendors_rejects_invalid_status(client):
    resp = client.get("/api/v1/vendors", params={"status": "not-a-status"}, headers=AUTH_HEADERS)
    assert resp.status_code == 422


def test_list_vendors_filters_by_partial_name(client):
    resp = client.get("/api/v1/vendors", params={"name": "acme"}, headers=AUTH_HEADERS)
    assert resp.status_code == 200
    vendors = resp.json()
    assert len(vendors) == 1
    assert vendors[0]["name"] == "Acme Corp"


def test_get_vendor_by_id(client):
    listed = client.get("/api/v1/vendors", params={"name": "Acme"}, headers=AUTH_HEADERS).json()
    vendor_id = listed[0]["id"]

    resp = client.get(f"/api/v1/vendors/{vendor_id}", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    assert resp.json()["name"] == "Acme Corp"
    assert resp.json()["email"] == "jane@acme.com"


def test_get_vendor_not_found(client):
    resp = client.get("/api/v1/vendors/999999", headers=AUTH_HEADERS)
    assert resp.status_code == 404


def test_vendors_requires_auth(client):
    resp = client.get("/api/v1/vendors")
    assert resp.status_code == 401


# ── /purchase-orders ───────────────────────────────────────────────────────────

def test_list_purchase_orders_returns_all(client):
    resp = client.get("/api/v1/purchase-orders", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    assert {po["po_number"] for po in resp.json()} == {"PO-100001", "PO-100002", "PO-100003"}


def test_list_purchase_orders_includes_joined_vendor_name(client):
    resp = client.get("/api/v1/purchase-orders", params={"status": "approved"}, headers=AUTH_HEADERS)
    pos = resp.json()
    assert len(pos) == 1
    assert pos[0]["po_number"] == "PO-100001"
    assert pos[0]["vendor_name"] == "Acme Corp"
    assert pos[0]["approved_by"] == "manager@company.com"


def test_list_purchase_orders_rejects_invalid_status(client):
    resp = client.get("/api/v1/purchase-orders", params={"status": "bogus"}, headers=AUTH_HEADERS)
    assert resp.status_code == 422


def test_list_purchase_orders_filters_by_vendor_id(client):
    vendors = client.get("/api/v1/vendors", params={"name": "TechSupply"}, headers=AUTH_HEADERS).json()
    vendor_id = vendors[0]["id"]

    resp = client.get(
        "/api/v1/purchase-orders", params={"vendor_id": vendor_id}, headers=AUTH_HEADERS
    )
    pos = resp.json()
    assert len(pos) == 1
    assert pos[0]["po_number"] == "PO-100002"


def test_list_purchase_orders_filters_by_amount_range(client):
    resp = client.get(
        "/api/v1/purchase-orders",
        params={"min_amount": 10000, "max_amount": 50000},
        headers=AUTH_HEADERS,
    )
    pos = resp.json()
    assert len(pos) == 1
    assert pos[0]["po_number"] == "PO-100001"


def test_get_purchase_order_by_number(client):
    resp = client.get("/api/v1/purchase-orders/PO-100002", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["amount"] == 87500.0
    assert body["status"] == "pending"


def test_get_purchase_order_not_found(client):
    resp = client.get("/api/v1/purchase-orders/PO-000000", headers=AUTH_HEADERS)
    assert resp.status_code == 404


def test_purchase_orders_requires_auth(client):
    resp = client.get("/api/v1/purchase-orders")
    assert resp.status_code == 401


# ── /invoices ──────────────────────────────────────────────────────────────────

def test_list_invoices_returns_all(client):
    resp = client.get("/api/v1/invoices", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    assert len(resp.json()) == 3


def test_list_invoices_filters_by_status(client):
    resp = client.get("/api/v1/invoices", params={"status": "disputed"}, headers=AUTH_HEADERS)
    invoices = resp.json()
    assert len(invoices) == 1
    assert invoices[0]["invoice_number"] == "INV-2025-003"


def test_list_invoices_rejects_invalid_status(client):
    resp = client.get("/api/v1/invoices", params={"status": "not-real"}, headers=AUTH_HEADERS)
    assert resp.status_code == 422


def test_list_invoices_filters_by_po_id(client):
    po = client.get("/api/v1/purchase-orders/PO-100001", headers=AUTH_HEADERS).json()
    resp = client.get("/api/v1/invoices", params={"po_id": po["id"]}, headers=AUTH_HEADERS)
    invoices = resp.json()
    assert len(invoices) == 1
    assert invoices[0]["invoice_number"] == "INV-2025-001"


def test_list_invoices_filters_by_vendor_id(client):
    vendor = client.get("/api/v1/vendors", params={"name": "Acme"}, headers=AUTH_HEADERS).json()[0]
    resp = client.get(
        "/api/v1/invoices", params={"vendor_id": vendor["id"]}, headers=AUTH_HEADERS
    )
    invoices = resp.json()
    assert len(invoices) == 1
    assert invoices[0]["invoice_number"] == "INV-2025-001"


def test_list_invoices_overdue_only_excludes_paid_and_future(client):
    """INV-2025-002 is unpaid and past due -> overdue. INV-2025-001 is paid
    (excluded even though its due_date is in the past). INV-2025-003's due
    date is in the future (excluded)."""
    resp = client.get("/api/v1/invoices", params={"overdue_only": True}, headers=AUTH_HEADERS)
    invoices = resp.json()
    assert {i["invoice_number"] for i in invoices} == {"INV-2025-002"}


def test_get_invoice_by_number(client):
    resp = client.get("/api/v1/invoices/INV-2025-003", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "disputed"
    assert body["vendor_name"] == "ShadyDeals LLC"
    assert body["po_number"] == "PO-100003"


def test_get_invoice_not_found(client):
    resp = client.get("/api/v1/invoices/INV-9999-999", headers=AUTH_HEADERS)
    assert resp.status_code == 404


def test_invoices_requires_auth(client):
    resp = client.get("/api/v1/invoices")
    assert resp.status_code == 401
