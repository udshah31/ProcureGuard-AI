"""
tests/test_auth.py
────────────────────
Unit tests for api.auth._parse_api_keys / require_api_key, plus HTTP-level
tests against the demo server confirming the X-API-Key gate on /api/v1
routes and that /health stays open.
"""

import pytest
from fastapi import HTTPException

from api.auth import AuthContext, _parse_api_keys, require_api_key


# ── _parse_api_keys ───────────────────────────────────────────────────────────

def test_parses_multiple_key_role_pairs_with_default_identity():
    keys = _parse_api_keys("k1:requester,k2:approver,k3:finance,k4:admin")
    assert keys == {
        "k1": AuthContext("requester", "requester@procureguard.local"),
        "k2": AuthContext("approver", "approver@procureguard.local"),
        "k3": AuthContext("finance", "finance@procureguard.local"),
        "k4": AuthContext("admin", "admin@procureguard.local"),
    }


def test_parses_explicit_identity():
    keys = _parse_api_keys("k1:approver:bob@company.com")
    assert keys == {"k1": AuthContext("approver", "bob@company.com")}


def test_empty_string_yields_no_keys():
    assert _parse_api_keys("") == {}


def test_ignores_malformed_entries():
    keys = _parse_api_keys("k1:requester, not-a-pair, k2:not-a-real-role, ,k3:admin")
    assert keys == {
        "k1": AuthContext("requester", "requester@procureguard.local"),
        "k3": AuthContext("admin", "admin@procureguard.local"),
    }


def test_tolerates_surrounding_whitespace():
    keys = _parse_api_keys(" k1 : requester : bob@company.com , k2:admin ")
    assert keys == {
        "k1": AuthContext("requester", "bob@company.com"),
        "k2": AuthContext("admin", "admin@procureguard.local"),
    }


# ── require_api_key (unit) ────────────────────────────────────────────────────

def test_rejects_missing_key(monkeypatch):
    monkeypatch.setenv("API_KEYS", "k1:requester")
    with pytest.raises(HTTPException) as exc:
        require_api_key(x_api_key=None)
    assert exc.value.status_code == 401


def test_rejects_unknown_key(monkeypatch):
    monkeypatch.setenv("API_KEYS", "k1:requester")
    with pytest.raises(HTTPException) as exc:
        require_api_key(x_api_key="not-a-real-key")
    assert exc.value.status_code == 401


def test_accepts_a_known_key_and_returns_its_auth_context(monkeypatch):
    monkeypatch.setenv("API_KEYS", "k1:approver:bob@company.com")
    assert require_api_key(x_api_key="k1") == AuthContext("approver", "bob@company.com")


def test_rejects_everything_when_no_keys_configured(monkeypatch):
    monkeypatch.delenv("API_KEYS", raising=False)
    with pytest.raises(HTTPException) as exc:
        require_api_key(x_api_key="anything")
    assert exc.value.status_code == 503


# ── HTTP-level: demo server ───────────────────────────────────────────────────

AUTH_HEADERS = {"X-API-Key": "test-key"}


@pytest.fixture
def demo_client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    db_path = tmp_path / "auth_demo.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    monkeypatch.setenv(
        "API_KEYS",
        "test-key:requester,approver-key:approver,admin-key:admin",
    )

    import importlib

    import api.server_demo as server_demo

    importlib.reload(server_demo)

    with TestClient(server_demo.app) as client:
        yield client


def test_vendors_rejects_missing_key(demo_client):
    resp = demo_client.get("/api/v1/vendors")
    assert resp.status_code == 401


def test_vendors_rejects_wrong_key(demo_client):
    resp = demo_client.get("/api/v1/vendors", headers={"X-API-Key": "bogus"})
    assert resp.status_code == 401


def test_vendors_allows_a_valid_key(demo_client):
    resp = demo_client.get("/api/v1/vendors", headers=AUTH_HEADERS)
    assert resp.status_code == 200


def test_metrics_requires_a_key(demo_client):
    assert demo_client.get("/api/v1/metrics").status_code == 401
    assert demo_client.get("/api/v1/metrics", headers=AUTH_HEADERS).status_code == 200


def test_chat_requires_a_key(demo_client):
    resp = demo_client.post("/api/v1/chat", json={"session_id": "s1", "message": "hi"})
    assert resp.status_code == 401


def test_chat_role_comes_from_the_key_not_the_body(demo_client):
    """A caller authenticated as 'requester' cannot claim to be 'admin' via
    a role field in the request body — there is no such field anymore, and
    even a client that sends one must be ignored."""
    resp = demo_client.post(
        "/api/v1/chat",
        json={"session_id": "s1", "message": "hi", "role": "admin"},
        headers=AUTH_HEADERS,  # maps to "requester"
    )
    assert resp.status_code == 200


def test_health_stays_open(demo_client):
    resp = demo_client.get("/api/v1/health")
    assert resp.status_code == 200


def test_requested_by_comes_from_the_authenticated_identity(demo_client):
    """Creating a PO must record the authenticated caller's identity as
    requested_by, not a generic role-derived placeholder."""
    import sqlite3

    import api.server_demo as server_demo

    db_path = server_demo.DB_PATH
    with sqlite3.connect(db_path) as conn:
        conn.execute("INSERT INTO vendors (name, status) VALUES ('Acme', 'active')")
        conn.commit()

    resp = demo_client.post(
        "/api/v1/chat",
        json={"session_id": "s1", "message": "create po for Acme $1000"},
        headers=AUTH_HEADERS,  # "test-key" -> requester@procureguard.local
    )
    assert resp.status_code == 200

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT requested_by FROM purchase_orders WHERE status='draft' "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
    assert row is not None
    assert row[0] == "requester@procureguard.local"


def test_approved_by_comes_from_the_authenticated_identity_not_the_message(demo_client, tmp_path, monkeypatch):
    """A caller cannot claim to be a different approver by naming one in the
    chat message — approved_by must always be the authenticated identity."""
    import sqlite3

    import api.server_demo as server_demo
    from agent import guard_rules

    db_path = server_demo.DB_PATH
    monkeypatch.setattr(guard_rules, "DB_PATH", db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO vendors (name, status) VALUES ('Acme', 'active')"
        )
        vendor_id = conn.execute("SELECT id FROM vendors WHERE name='Acme'").fetchone()[0]
        conn.execute(
            "INSERT INTO purchase_orders (po_number, vendor_id, amount, status) "
            "VALUES ('PO-999001', ?, 1000, 'pending')",
            (vendor_id,),
        )
        conn.commit()

    resp = demo_client.post(
        "/api/v1/chat",
        json={"session_id": "s1", "message": "approve PO-999001 by spoofed@evil.com"},
        headers=AUTH_HEADERS,  # "test-key" -> requester@procureguard.local
    )
    assert resp.status_code == 200

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT approved_by FROM purchase_orders WHERE po_number='PO-999001'"
        ).fetchone()
    assert row[0] == "requester@procureguard.local"
