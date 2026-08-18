"""
api/sessions.py
───────────────
In-memory session store for multi-turn agent conversations.

Each session maps a session_id → AgentState dict.
Sessions expire after SESSION_TTL_SECONDS of inactivity (default 30 min).
Thread-safe via threading.Lock (safe for single-worker uvicorn).

Usage:
    store = SessionStore()
    state = store.get_or_create("user-abc", role="approver")
    store.update("user-abc", new_state)
    store.delete("user-abc")
"""

import logging
import threading
import time
from typing import Optional

log = logging.getLogger(__name__)

SESSION_TTL_SECONDS: int = 30 * 60  # 30 minutes


class SessionStore:
    def __init__(self, ttl: int = SESSION_TTL_SECONDS) -> None:
        self._ttl = ttl
        self._lock = threading.Lock()
        # { session_id: {"state": AgentState, "last_access": float} }
        self._store: dict[str, dict] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def get_or_create(self, session_id: str, role: str = "requester", identity: str = "") -> dict:
        """
        Return the existing AgentState for session_id, or create a fresh one.
        Refreshes the TTL on access.
        """
        with self._lock:
            self._evict_expired()
            if session_id in self._store:
                entry = self._store[session_id]
                entry["last_access"] = time.monotonic()
                log.debug("Session '%s' resumed (%d messages).",
                          session_id, len(entry["state"]["messages"]))
                return entry["state"]

            # New session
            state: dict = {
                "messages": [],
                "role": role,
                "identity": identity,
                "context": {},
                "risk_flags": [],
            }
            self._store[session_id] = {
                "state": state,
                "last_access": time.monotonic(),
            }
            log.info("Session '%s' created (role=%s).", session_id, role)
            return state

    def update(self, session_id: str, state: dict) -> None:
        """Persist the updated AgentState after an agent invocation."""
        with self._lock:
            if session_id in self._store:
                self._store[session_id]["state"] = state
                self._store[session_id]["last_access"] = time.monotonic()

    def delete(self, session_id: str) -> None:
        """Explicitly remove a session."""
        with self._lock:
            self._store.pop(session_id, None)
            log.info("Session '%s' deleted.", session_id)

    def active_count(self) -> int:
        """Return the number of currently active (non-expired) sessions."""
        with self._lock:
            self._evict_expired()
            return len(self._store)

    def list_sessions(self) -> list[str]:
        """Return all active session IDs."""
        with self._lock:
            self._evict_expired()
            return list(self._store.keys())

    # ── Internal ──────────────────────────────────────────────────────────────

    def _evict_expired(self) -> None:
        """Remove sessions that have exceeded TTL. Must be called under lock."""
        now = time.monotonic()
        expired = [
            sid for sid, entry in self._store.items()
            if (now - entry["last_access"]) > self._ttl
        ]
        for sid in expired:
            del self._store[sid]
            log.info("Session '%s' expired and evicted.", sid)
