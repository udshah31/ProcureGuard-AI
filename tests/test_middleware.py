"""
tests/test_middleware.py
───────────────────────────
Unit tests for api/middleware.py's route_label (pure function — a fake
duck-typed Request is enough, no need to spin up Starlette internals) plus
an HTTP-level test that RequestContextMiddleware still tags the response
with X-Request-ID even when the downstream handler raises.
"""

import pytest

from api.middleware import route_label, _UNMATCHED


class FakeURL:
    def __init__(self, path: str) -> None:
        self.path = path


class FakeRequest:
    """Duck-typed stand-in for starlette.requests.Request — route_label only
    reads .scope and .url.path, so a real Request (and its ASGI machinery)
    isn't needed to exercise it."""

    def __init__(self, path: str, route_path: str | None, path_params: dict | None = None):
        self.url = FakeURL(path)
        route = None
        if route_path is not None:
            route = type("FakeRoute", (), {"path": route_path})()
        self.scope = {"route": route, "path_params": path_params or {}}


def test_unmatched_route_returns_sentinel():
    req = FakeRequest("/nope", route_path=None)
    assert route_label(req) == _UNMATCHED


def test_static_route_returns_prefixed_path():
    req = FakeRequest("/api/v1/health", route_path="/health")
    assert route_label(req) == "/api/v1/health"


def test_route_with_path_param_renders_the_template():
    req = FakeRequest("/api/v1/vendors/42", route_path="/vendors/{vendor_id}", path_params={"vendor_id": "42"})
    assert route_label(req) == "/api/v1/vendors/{vendor_id}"


def test_multiple_path_params_all_render():
    req = FakeRequest(
        "/api/v1/orgs/acme/vendors/42",
        route_path="/orgs/{org}/vendors/{vendor_id}",
        path_params={"org": "acme", "vendor_id": "42"},
    )
    assert route_label(req) == "/api/v1/orgs/{org}/vendors/{vendor_id}"


def test_prefix_is_whatever_precedes_the_rendered_tail():
    """The router-relative route.path doesn't carry the router's mount
    prefix, so route_label recovers it from wherever the rendered tail
    lands inside the actual request path — this must work regardless of
    what that prefix is, not just '/api/v1'."""
    req = FakeRequest("/some/other/prefix/vendors/7", route_path="/vendors/{vendor_id}", path_params={"vendor_id": "7"})
    assert route_label(req) == "/some/other/prefix/vendors/{vendor_id}"


def test_rendered_tail_not_found_in_url_falls_back_to_route_path():
    """Defensive path: if the rendered template unexpectedly isn't a
    suffix of the actual URL, fall back to the raw route path rather than
    raising or returning a nonsensical slice."""
    req = FakeRequest("/api/v1/vendors/42", route_path="/vendors/{other_id}", path_params={"vendor_id": "42"})
    assert route_label(req) == "/vendors/{other_id}"


# ── HTTP-level: X-Request-ID survives a handler exception ─────────────────────

@pytest.fixture
def crashing_app():
    from fastapi import FastAPI

    from api.middleware import RequestContextMiddleware

    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)

    @app.get("/boom")
    def boom():
        raise RuntimeError("kaboom")

    @app.get("/ok")
    def ok():
        return {"ok": True}

    return app


def test_ok_response_carries_request_id(crashing_app):
    from fastapi.testclient import TestClient

    with TestClient(crashing_app, raise_server_exceptions=False) as client:
        resp = client.get("/ok")
        assert resp.status_code == 200
        assert resp.headers["X-Request-ID"]


def test_unhandled_exception_still_completes_the_middleware(crashing_app):
    """The middleware's finally-block (metrics, logging, contextvar reset)
    must run even when the handler raises — a crash there shouldn't also
    corrupt every subsequent request's request_id."""
    from fastapi.testclient import TestClient

    with TestClient(crashing_app, raise_server_exceptions=False) as client:
        resp = client.get("/boom")
        assert resp.status_code == 500

        # A follow-up request must get its own, different request_id — proof
        # the contextvar was reset rather than left dangling from the crash.
        resp2 = client.get("/ok", headers={"X-Request-ID": "caller-id"})
        assert resp2.headers["X-Request-ID"] == "caller-id"
