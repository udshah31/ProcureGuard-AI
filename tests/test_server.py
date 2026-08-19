"""
tests/test_server.py
──────────────────────
HTTP-level tests for api/server.py (the real, LLM-backed app) — as opposed
to api/server_demo.py, which the rest of the HTTP-level test suite exercises
because it needs no LLM key.

The fixture forces LLM_PROVIDER=groq with no GROQ_API_KEY set, so build_llm()
deterministically raises RuntimeError and app.state.graph is None — the same
"LLM unavailable" path server.py's lifespan already handles by letting the
rest of the API serve. That keeps these tests hermetic (no real provider
key, no network call, no dependency on what's installed locally) while still
exercising server.py's own lifespan, routing, and auth wiring rather than
server_demo's.
"""

import sqlite3

import pytest

AUTH_HEADERS = {"X-API-Key": "test-key"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    db_path = str(tmp_path / "server_test.db")
    monkeypatch.setenv("DB_PATH", db_path)
    monkeypatch.setenv("API_KEYS", "test-key:admin")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "")  # empty, not absent — delenv would let load_dotenv() refill it from .env on reload
    monkeypatch.setenv("GOOGLE_API_KEY", "")

    import importlib

    import api.routers.invoices as invoices_router
    import api.routers.purchase_orders as po_router
    import api.routers.vendors as vendors_router
    import api.server as server

    importlib.reload(server)
    monkeypatch.setattr(vendors_router, "DB_PATH", db_path)
    monkeypatch.setattr(po_router, "DB_PATH", db_path)
    monkeypatch.setattr(invoices_router, "DB_PATH", db_path)

    with TestClient(server.app) as c:
        yield c, db_path


def test_health_reports_no_llm_when_key_is_missing(client):
    c, _ = client
    resp = c.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["active_sessions"] == 0


def test_chat_returns_503_when_llm_is_unconfigured(client):
    c, _ = client
    resp = c.post(
        "/api/v1/chat",
        json={"session_id": "s1", "message": "hello"},
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 503
    assert "No LLM configured" in resp.json()["detail"]


def test_chat_requires_auth_even_before_the_llm_check(client):
    c, _ = client
    resp = c.post("/api/v1/chat", json={"session_id": "s1", "message": "hello"})
    assert resp.status_code == 401


def test_lifespan_auto_seeds_an_empty_database(client):
    """server.py's lifespan runs its own init_db + seed — this must work
    independently of server_demo's equivalent path."""
    c, _ = client
    resp = c.get("/api/v1/vendors", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    assert len(resp.json()) > 0


def test_vendors_requires_auth(client):
    c, _ = client
    resp = c.get("/api/v1/vendors")
    assert resp.status_code == 401


def test_purchase_orders_requires_auth(client):
    c, _ = client
    resp = c.get("/api/v1/purchase-orders")
    assert resp.status_code == 401


def test_invoices_requires_auth(client):
    c, _ = client
    resp = c.get("/api/v1/invoices")
    assert resp.status_code == 401


def test_metrics_requires_auth(client):
    c, _ = client
    resp = c.get("/api/v1/metrics")
    assert resp.status_code == 401


def test_health_stays_open(client):
    c, _ = client
    resp = c.get("/api/v1/health")
    assert resp.status_code == 200


def test_response_carries_a_request_id(client):
    c, _ = client
    resp = c.get("/api/v1/vendors", headers=AUTH_HEADERS)
    assert resp.headers["X-Request-ID"]


def test_root_serves_the_ui(client):
    c, _ = client
    resp = c.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_does_not_reseed_a_nonempty_database(client, tmp_path, monkeypatch):
    """Startup only seeds when vendors is empty — a database that already
    has data must be left alone."""
    import importlib

    from db.init_db import init_db

    db_path = str(tmp_path / "preseeded.db")
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("INSERT INTO vendors (name, status) VALUES ('Only Vendor', 'active')")
        conn.commit()

    monkeypatch.setenv("DB_PATH", db_path)
    monkeypatch.setenv("API_KEYS", "test-key:admin")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "")  # empty, not absent — delenv would let load_dotenv() refill it from .env on reload
    monkeypatch.setenv("GOOGLE_API_KEY", "")

    import api.routers.vendors as vendors_router
    import api.server as server

    importlib.reload(server)
    monkeypatch.setattr(vendors_router, "DB_PATH", db_path)

    from fastapi.testclient import TestClient

    with TestClient(server.app) as c2:
        resp = c2.get("/api/v1/vendors", headers=AUTH_HEADERS)
        vendors = resp.json()

    assert len(vendors) == 1
    assert vendors[0]["name"] == "Only Vendor"
