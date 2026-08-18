"""
api/rate_limit.py
──────────────────
Sliding-window rate limiter for the chat endpoint.

/api/v1/chat is the only endpoint that calls out to a paid LLM, so it's the
one worth protecting from a client hammering it. There's no real auth yet, so
the only identity signal available at the transport layer is the client's
address — session_id and role are self-reported in the request body and
trivially rotated, so they're not usable as a limiter key on their own.

In-memory only, same tradeoff SessionStore already makes: correct for a
single-worker deployment, and each additional uvicorn worker would track its
own independent window. A multi-worker deployment needs a shared store
(Redis, etc.) to enforce one global limit.
"""

import threading
import time
from collections import deque


class RateLimiter:
    """Fixed-size sliding window: at most `max_requests` per `window_seconds`, per key."""

    def __init__(
        self,
        max_requests: int,
        window_seconds: float,
        clock=time.monotonic,
    ) -> None:
        if max_requests < 1:
            raise ValueError("max_requests must be >= 1")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        self._max = max_requests
        self._window = window_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._hits: dict[str, deque] = {}
        self._last_sweep = clock()

    def _sweep(self, now: float) -> None:
        """
        Drop keys whose windows have fully aged out. Caller must hold the lock.

        Without this the key map grows for the lifetime of the process: one
        entry per address ever seen, never reclaimed. A client rotating source
        addresses could exhaust memory through the very component meant to
        stop it.
        """
        cutoff = now - self._window
        stale = [k for k, w in self._hits.items() if not w or w[-1] <= cutoff]
        for key in stale:
            del self._hits[key]
        self._last_sweep = now

    def check(self, key: str) -> tuple[bool, float]:
        """
        Record a request attempt for `key`.

        Returns (allowed, retry_after_seconds). If allowed is False, the
        request is NOT recorded — the caller should reject it, and
        retry_after_seconds says how long until the oldest hit in the window
        ages out and a slot frees up.
        """
        now = self._clock()
        with self._lock:
            if now - self._last_sweep >= self._window:
                self._sweep(now)

            window = self._hits.setdefault(key, deque())
            cutoff = now - self._window
            while window and window[0] <= cutoff:
                window.popleft()

            if len(window) >= self._max:
                retry_after = window[0] + self._window - now
                return False, max(retry_after, 0.0)

            window.append(now)
            return True, 0.0

    def tracked_keys(self) -> int:
        """Number of keys currently held. Exposed for tests and monitoring."""
        with self._lock:
            return len(self._hits)

    def reset(self, key: str) -> None:
        """Clear a key's window. Mainly useful for tests."""
        with self._lock:
            self._hits.pop(key, None)
