"""
observability.py
─────────────────
Lightweight request correlation + in-memory metrics — no new dependency, no
external collector. Lives at the repo root (not under api/ or agent/) because
both layers need it: main.py's graph nodes record guard/tool outcomes, and
api/ records HTTP-level counters and stamps log lines with a request ID.

    configure_logging()  — call once at process startup; idempotent.
    request_id_var        — ContextVar carrying the current request's ID,
                             propagated automatically into thread-pooled sync
                             endpoint handlers by Starlette/anyio.
    metrics                — process-wide counters + latency stats singleton.

Everything here is in-memory and single-process, the same tradeoff
api/sessions.py and api/rate_limit.py already make. A multi-worker deployment
needs a real metrics backend (Prometheus, etc.) to get one global view.
"""

import logging
import os
import threading
import time
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    """Attaches the current request ID (or '-' outside a request) to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


_LOG_FORMAT = "%(asctime)s | %(levelname)s | req=%(request_id)s | %(name)s | %(message)s"


def configure_logging(level: str | None = None) -> None:
    """
    Configure root logging with request-ID-aware formatting.

    Safe to call from multiple entry points (main.py, api/server.py,
    api/server_demo.py all import each other in various combinations) —
    only the first call installs handlers; later calls just make sure the
    request-ID filter is present.
    """
    level = level or os.getenv("LOG_LEVEL", "INFO")
    root = logging.getLogger()

    if not root.handlers:
        logging.basicConfig(level=level, format=_LOG_FORMAT)
    else:
        root.setLevel(level)

    for handler in root.handlers:
        if not any(isinstance(f, RequestIdFilter) for f in handler.filters):
            handler.addFilter(RequestIdFilter())


class Metrics:
    """Thread-safe counters and simple latency stats (count/avg/max per name)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {}
        self._timers: dict[str, dict[str, float]] = {}

    def increment(self, name: str, by: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + by

    def record_latency(self, name: str, seconds: float) -> None:
        with self._lock:
            t = self._timers.setdefault(name, {"count": 0, "total": 0.0, "max": 0.0})
            t["count"] += 1
            t["total"] += seconds
            t["max"] = max(t["max"], seconds)

    def snapshot(self) -> dict:
        """A JSON-serialisable point-in-time view of all counters and timers."""
        with self._lock:
            counters = dict(self._counters)
            timers = {
                name: {
                    "count": t["count"],
                    "avg_ms": round((t["total"] / t["count"]) * 1000, 2) if t["count"] else 0.0,
                    "max_ms": round(t["max"] * 1000, 2),
                }
                for name, t in self._timers.items()
            }
        return {"counters": counters, "timers": timers}

    def reset(self) -> None:
        """Clear all state. Mainly useful for tests."""
        with self._lock:
            self._counters.clear()
            self._timers.clear()


metrics = Metrics()


class Timer:
    """Context manager: `with Timer() as t: ...` then `t.elapsed` in seconds."""

    def __enter__(self) -> "Timer":
        self._start = time.monotonic()
        self.elapsed = 0.0
        return self

    def __exit__(self, *exc_info) -> None:
        self.elapsed = time.monotonic() - self._start
