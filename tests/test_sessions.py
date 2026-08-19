"""
tests/test_sessions.py
────────────────────────
Unit tests for api/sessions.py's SessionStore — an in-memory, TTL-evicting,
thread-safe map from session_id to AgentState.
"""

from api.sessions import SessionStore


def test_get_or_create_makes_a_fresh_state():
    store = SessionStore()
    state = store.get_or_create("s1", role="approver", identity="bob@company.com")
    assert state == {
        "messages": [],
        "role": "approver",
        "identity": "bob@company.com",
        "context": {},
        "risk_flags": [],
    }


def test_get_or_create_defaults_role_and_identity():
    store = SessionStore()
    state = store.get_or_create("s1")
    assert state["role"] == "requester"
    assert state["identity"] == ""


def test_get_or_create_resumes_an_existing_session():
    store = SessionStore()
    first = store.get_or_create("s1", role="approver")
    first["messages"] = ["some message"]

    second = store.get_or_create("s1", role="admin")  # role ignored on resume
    assert second is first
    assert second["messages"] == ["some message"]
    assert second["role"] == "approver"


def test_update_persists_new_state():
    store = SessionStore()
    store.get_or_create("s1")
    store.update("s1", {"messages": ["hi"], "role": "approver", "identity": "", "context": {}, "risk_flags": []})

    resumed = store.get_or_create("s1")
    assert resumed["messages"] == ["hi"]


def test_update_on_unknown_session_is_a_noop():
    store = SessionStore()
    store.update("never-created", {"messages": ["x"]})
    assert store.active_count() == 0


def test_delete_removes_the_session():
    store = SessionStore()
    store.get_or_create("s1")
    store.delete("s1")
    assert store.active_count() == 0


def test_delete_unknown_session_does_not_raise():
    store = SessionStore()
    store.delete("never-created")  # no exception


def test_active_count_reflects_number_of_sessions():
    store = SessionStore()
    store.get_or_create("s1")
    store.get_or_create("s2")
    assert store.active_count() == 2


def test_list_sessions_returns_all_ids():
    store = SessionStore()
    store.get_or_create("s1")
    store.get_or_create("s2")
    assert set(store.list_sessions()) == {"s1", "s2"}


# ── TTL eviction ─────────────────────────────────────────────────────────────

class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_expired_sessions_are_evicted_on_access(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr("api.sessions.time.monotonic", clock)

    store = SessionStore(ttl=60)
    store.get_or_create("s1")

    clock.advance(61)
    assert store.active_count() == 0


def test_recently_accessed_session_survives_the_sweep(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr("api.sessions.time.monotonic", clock)

    store = SessionStore(ttl=60)
    store.get_or_create("s1")

    clock.advance(30)
    store.get_or_create("s1")  # refresh TTL

    clock.advance(40)  # 70s since creation, but only 40s since refresh
    assert store.active_count() == 1


def test_get_or_create_on_expired_session_creates_a_new_one(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr("api.sessions.time.monotonic", clock)

    store = SessionStore(ttl=60)
    first = store.get_or_create("s1", role="approver")
    first["messages"] = ["old"]

    clock.advance(61)
    second = store.get_or_create("s1", role="approver")
    assert second["messages"] == []
