"""
tests/test_observability.py
──────────────────────────────
Metrics unit tests, plus HTTP-level tests against the demo server confirming
X-Request-ID propagation and the /metrics endpoint reflect real traffic.
"""

import logging

import pytest

from observability import Metrics, RequestIdFilter, Timer, request_id_var


def test_increment_accumulates_per_key():
    m = Metrics()
    m.increment("a")
    m.increment("a")
    m.increment("b", by=5)

    snap = m.snapshot()
    assert snap["counters"] == {"a": 2, "b": 5}


def test_record_latency_computes_avg_and_max():
    m = Metrics()
    m.record_latency("op", 0.1)
    m.record_latency("op", 0.3)

    snap = m.snapshot()
    timer = snap["timers"]["op"]
    assert timer["count"] == 2
    assert timer["avg_ms"] == pytest.approx(200.0, abs=0.1)
    assert timer["max_ms"] == pytest.approx(300.0, abs=0.1)


def test_reset_clears_everything():
    m = Metrics()
    m.increment("a")
    m.record_latency("op", 0.1)
    m.reset()

    snap = m.snapshot()
    assert snap["counters"] == {}
    assert snap["timers"] == {}


def test_snapshot_is_independent_of_further_mutation():
    m = Metrics()
    m.increment("a")
    snap = m.snapshot()
    m.increment("a")

    assert snap["counters"]["a"] == 1


def test_timer_context_manager_measures_elapsed():
    with Timer() as t:
        pass
    assert t.elapsed >= 0.0


def test_request_id_filter_defaults_to_dash_outside_a_request():
    token = request_id_var.set("-")
    try:
        record = logging.LogRecord("x", logging.INFO, __file__, 1, "msg", None, None)
        assert RequestIdFilter().filter(record)
        assert record.request_id == "-"
    finally:
        request_id_var.reset(token)


def test_request_id_filter_picks_up_context_value():
    token = request_id_var.set("abc123")
    try:
        record = logging.LogRecord("x", logging.INFO, __file__, 1, "msg", None, None)
        RequestIdFilter().filter(record)
        assert record.request_id == "abc123"
    finally:
        request_id_var.reset(token)


# ── HTTP-level: demo server ───────────────────────────────────────────────────


AUTH_HEADERS = {"X-API-Key": "test-key"}


@pytest.fixture
def demo_client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    db_path = tmp_path / "observability_demo.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    monkeypatch.setenv("API_KEYS", "test-key:admin")

    import importlib

    import api.server_demo as server_demo
    from observability import metrics as global_metrics

    importlib.reload(server_demo)
    global_metrics.reset()

    with TestClient(server_demo.app) as client:
        yield client


def test_response_carries_a_request_id(demo_client):
    resp = demo_client.get("/api/v1/vendors", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    assert resp.headers["X-Request-ID"]


def test_incoming_request_id_is_reused(demo_client):
    resp = demo_client.get(
        "/api/v1/vendors", headers={**AUTH_HEADERS, "X-Request-ID": "caller-supplied-id"}
    )
    assert resp.headers["X-Request-ID"] == "caller-supplied-id"


def test_metrics_endpoint_reflects_traffic(demo_client):
    demo_client.get("/api/v1/vendors", headers=AUTH_HEADERS)
    demo_client.get("/api/v1/vendors", headers=AUTH_HEADERS)

    resp = demo_client.get("/api/v1/metrics", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    body = resp.json()

    # Two prior /vendors calls, plus this /metrics call itself is not yet counted
    # (it's recorded after the response is built).
    assert body["counters"]["http_requests_total:GET:/api/v1/vendors:200"] == 2
    assert "http_latency:GET:/api/v1/vendors" in body["timers"]


def test_chat_endpoint_increments_chat_counters(demo_client):
    demo_client.post(
        "/api/v1/chat",
        json={"session_id": "s1", "message": "hello"},
        headers=AUTH_HEADERS,
    )
    resp = demo_client.get("/api/v1/metrics", headers=AUTH_HEADERS)
    assert resp.json()["counters"]["chat_requests_total"] == 1


# ── Metric cardinality ───────────────────────────────────────────────────────

def test_path_params_collapse_into_one_series(demo_client):
    """Keying on the concrete path mints a series per id — 900 requests once
    produced 900 counters, which both leaks memory and destroys aggregation.

    Status code stays a separate dimension: it's bounded, and splitting 200s
    from 404s is the whole point of the counter.
    """
    for i in range(25):
        demo_client.get(f"/api/v1/vendors/{i}", headers=AUTH_HEADERS)

    counters = demo_client.get("/api/v1/metrics", headers=AUTH_HEADERS).json()["counters"]
    vendor_keys = [k for k in counters if "/vendors/" in k]
    templates = {k.rsplit(":", 1)[0] for k in vendor_keys}

    assert templates == {"http_requests_total:GET:/api/v1/vendors/{vendor_id}"}
    assert sum(counters[k] for k in vendor_keys) == 25


def test_route_label_keeps_the_router_prefix(demo_client):
    """scope['route'].path is router-relative, so a naive label would read
    '/vendors/{vendor_id}' and could collide with another router's route."""
    demo_client.get("/api/v1/vendors/1", headers=AUTH_HEADERS)

    counters = demo_client.get("/api/v1/metrics", headers=AUTH_HEADERS).json()["counters"]
    assert any(k.startswith("http_requests_total:GET:/api/v1/vendors/") for k in counters)


def test_unmatched_paths_share_a_single_bucket(demo_client):
    """404 probing must not be a remote memory-growth vector."""
    for i in range(25):
        demo_client.get(f"/definitely-not-a-route-{i}")

    counters = demo_client.get("/api/v1/metrics", headers=AUTH_HEADERS).json()["counters"]
    unmatched = {k: v for k, v in counters.items() if "<unmatched>" in k}

    assert len(unmatched) == 1
    assert sum(unmatched.values()) == 25


def test_collection_and_item_routes_are_separate_series(demo_client):
    demo_client.get("/api/v1/vendors", headers=AUTH_HEADERS)
    demo_client.get("/api/v1/vendors/1", headers=AUTH_HEADERS)

    counters = demo_client.get("/api/v1/metrics", headers=AUTH_HEADERS).json()["counters"]

    assert any(k.endswith(":GET:/api/v1/vendors:200") for k in counters)
    assert any("/api/v1/vendors/{vendor_id}" in k for k in counters)
