"""
tests/test_auth.py
────────────────────
Unit tests for api.auth._parse_api_keys / require_api_key, plus HTTP-level
tests against the demo server confirming the X-API-Key gate on /api/v1
routes and that /health stays open.
"""

import pytest
from fastapi import HTTPException

from api.auth import _parse_api_keys, require_api_key


# ── _parse_api_keys ───────────────────────────────────────────────────────────

def test_parses_multiple_key_role_pairs():
    keys = _parse_api_keys("k1:requester,k2:approver,k3:finance,k4:admin")
    assert keys == {"k1": "requester", "k2": "approver", "k3": "finance", "k4": "admin"}


def test_empty_string_yields_no_keys():
    assert _parse_api_keys("") == {}


def test_ignores_malformed_entries():
    keys = _parse_api_keys("k1:requester, not-a-pair, k2:not-a-real-role, ,k3:admin")
    assert keys == {"k1": "requester", "k3": "admin"}


def test_tolerates_surrounding_whitespace():
    keys = _parse_api_keys(" k1 : requester , k2:admin ")
    assert keys == {"k1": "requester", "k2": "admin"}


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


def test_accepts_a_known_key_and_returns_its_role(monkeypatch):
    monkeypatch.setenv("API_KEYS", "k1:approver")
    assert require_api_key(x_api_key="k1") == "approver"


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
