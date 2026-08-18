"""
api/middleware.py
──────────────────
RequestContextMiddleware — stamps every request with a correlation ID
(reused from an incoming X-Request-ID header if the caller sent one, so
requests can be traced across a reverse proxy), times it, logs one summary
line, and records HTTP-level counters/latency in observability.metrics.

Runs first in the middleware stack so every downstream log line — including
ones written deep inside guard_node/tool_node during graph.invoke — carries
the same request_id via observability.request_id_var.
"""

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from observability import metrics, request_id_var

log = logging.getLogger(__name__)

_UNMATCHED = "<unmatched>"


def route_label(request: Request) -> str:
    """
    A bounded, prefix-qualified label for the matched route.

    Metrics must be keyed on the route template, not the concrete path — keying
    on the path mints a new series per vendor id / PO number, so counters grow
    without bound and stop aggregating.

    scope["route"] is the innermost matched route, whose .path is relative to
    its router ("/vendors/{vendor_id}"), so the "/api/v1" prefix is missing and
    two routers could collide. Rendering the template with the request's actual
    path params gives the concrete tail, and whatever precedes it in the URL is
    the prefix.
    """
    route_path = getattr(request.scope.get("route"), "path", None)
    if not route_path:
        return _UNMATCHED

    rendered = route_path
    for name, value in (request.scope.get("path_params") or {}).items():
        rendered = rendered.replace("{" + name + "}", str(value))

    url_path = request.url.path
    if rendered and url_path.endswith(rendered):
        return url_path[: -len(rendered)] + route_path
    return route_path


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        token = request_id_var.set(req_id)
        start = time.monotonic()
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = req_id
            return response
        finally:
            elapsed = time.monotonic() - start
            route = route_label(request)
            metrics.increment(f"http_requests_total:{request.method}:{route}:{status_code}")
            metrics.record_latency(f"http_latency:{request.method}:{route}", elapsed)
            log.info(
                "%s %s -> %d (%.1fms)",
                request.method, request.url.path, status_code, elapsed * 1000,
            )
            request_id_var.reset(token)
