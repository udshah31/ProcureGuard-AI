"""
tests/test_rate_limit.py
──────────────────────────
RateLimiter unit tests (deterministic — driven by a fake clock, no real
sleeping) plus an HTTP-level test against the demo server's /chat endpoint,
which needs no LLM key and so can run in CI like the rest of the suite.
"""

import pytest

from api.rate_limit import RateLimiter


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_allows_up_to_the_limit_within_the_window():
    clock = FakeClock()
    limiter = RateLimiter(max_requests=3, window_seconds=60, clock=clock)

    for _ in range(3):
        allowed, retry_after = limiter.check("client-a")
        assert allowed
        assert retry_after == 0.0


def test_blocks_the_request_after_the_limit():
    clock = FakeClock()
    limiter = RateLimiter(max_requests=3, window_seconds=60, clock=clock)

    for _ in range(3):
        assert limiter.check("client-a")[0]

    allowed, retry_after = limiter.check("client-a")
    assert not allowed
    assert retry_after > 0


def test_blocked_request_is_not_recorded():
    """A rejected attempt must not itself consume a slot in the window."""
    clock = FakeClock()
    limiter = RateLimiter(max_requests=1, window_seconds=60, clock=clock)

    assert limiter.check("client-a")[0]
    for _ in range(5):
        assert not limiter.check("client-a")[0]

    clock.advance(61)
    assert limiter.check("client-a")[0]


def test_window_slides_and_frees_capacity():
    clock = FakeClock()
    limiter = RateLimiter(max_requests=2, window_seconds=10, clock=clock)

    assert limiter.check("client-a")[0]
    clock.advance(5)
    assert limiter.check("client-a")[0]

    # Both hits (t=0, t=5) still inside the 10s window at t=6 — third is blocked.
    clock.advance(1)
    assert not limiter.check("client-a")[0]

    # t=11: the t=0 hit has aged out, freeing one slot.
    clock.advance(5)
    allowed, _ = limiter.check("client-a")
    assert allowed


def test_keys_are_tracked_independently():
    clock = FakeClock()
    limiter = RateLimiter(max_requests=1, window_seconds=60, clock=clock)

    assert limiter.check("client-a")[0]
    assert not limiter.check("client-a")[0]
    assert limiter.check("client-b")[0], "a different key must not share client-a's window"


def test_reset_clears_a_keys_window():
    clock = FakeClock()
    limiter = RateLimiter(max_requests=1, window_seconds=60, clock=clock)

    assert limiter.check("client-a")[0]
    assert not limiter.check("client-a")[0]

    limiter.reset("client-a")
    assert limiter.check("client-a")[0]


@pytest.mark.parametrize("bad_kwargs", [{"max_requests": 0}, {"max_requests": -1}])
def test_rejects_non_positive_max_requests(bad_kwargs):
    with pytest.raises(ValueError):
        RateLimiter(window_seconds=60, **bad_kwargs)


@pytest.mark.parametrize("bad_kwargs", [{"window_seconds": 0}, {"window_seconds": -1}])
def test_rejects_non_positive_window(bad_kwargs):
    with pytest.raises(ValueError):
        RateLimiter(max_requests=5, **bad_kwargs)


# ── Key eviction ─────────────────────────────────────────────────────────────

def test_expired_keys_are_evicted():
    """Without a sweep the key map grows for the life of the process — one
    entry per address ever seen. 50k keys held ~38 MB before this."""
    clock = FakeClock()
    limiter = RateLimiter(max_requests=5, window_seconds=60, clock=clock)

    for i in range(1_000):
        limiter.check(f"client-{i}")
    assert limiter.tracked_keys() == 1_000

    clock.advance(61)
    limiter.check("someone-new")

    assert limiter.tracked_keys() == 1


def test_active_keys_survive_the_sweep():
    clock = FakeClock()
    limiter = RateLimiter(max_requests=5, window_seconds=60, clock=clock)

    limiter.check("stale-client")
    clock.advance(50)
    limiter.check("active-client")

    clock.advance(11)          # stale-client aged out, active-client has not
    limiter.check("trigger-sweep")

    assert limiter.tracked_keys() == 2
    assert "stale-client" not in limiter._hits


def test_sweep_does_not_reset_an_active_window():
    """Eviction must not hand a throttled client a fresh allowance."""
    clock = FakeClock()
    limiter = RateLimiter(max_requests=2, window_seconds=60, clock=clock)

    assert limiter.check("client-a")[0]
    clock.advance(30)
    assert limiter.check("client-a")[0]
    assert not limiter.check("client-a")[0]

    clock.advance(31)          # triggers a sweep; client-a's 2nd hit is still live
    allowed, _ = limiter.check("client-a")
    assert allowed             # first hit aged out, so exactly one slot freed
    assert not limiter.check("client-a")[0]


# ── HTTP-level: demo server's /chat, which needs no LLM key ──────────────────


@pytest.fixture
def demo_client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    db_path = tmp_path / "rate_limit_demo.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    monkeypatch.setenv("CHAT_RATE_LIMIT", "3")
    monkeypatch.setenv("CHAT_RATE_WINDOW_SECONDS", "60")

    import importlib

    import api.server_demo as server_demo

    importlib.reload(server_demo)

    with TestClient(server_demo.app) as client:
        yield client


def test_chat_endpoint_enforces_rate_limit(demo_client):
    for _ in range(3):
        resp = demo_client.post(
            "/api/v1/chat",
            json={"session_id": "s1", "message": "hello", "role": "requester"},
        )
        assert resp.status_code == 200

    resp = demo_client.post(
        "/api/v1/chat",
        json={"session_id": "s1", "message": "hello", "role": "requester"},
    )
    assert resp.status_code == 429
    assert "Retry-After" in resp.headers


def test_chat_rate_limit_is_keyed_per_client_not_per_session(demo_client):
    """Rotating session_id must not bypass the limit — the same TestClient IP
    should still get throttled after 3 requests regardless of session_id."""
    for i in range(3):
        resp = demo_client.post(
            "/api/v1/chat",
            json={"session_id": f"s{i}", "message": "hello", "role": "requester"},
        )
        assert resp.status_code == 200

    resp = demo_client.post(
        "/api/v1/chat",
        json={"session_id": "s-new", "message": "hello", "role": "requester"},
    )
    assert resp.status_code == 429
