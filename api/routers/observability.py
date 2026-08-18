"""
api/routers/observability.py
──────────────────────────────
GET /api/v1/metrics — snapshot of in-memory counters and latency stats.

Not authenticated (nothing else in this app is either) — fine for a demo
project, worth gating behind an internal network / auth check before this
ever fronts production traffic, since it exposes request volume and error
rates by route.
"""

from fastapi import APIRouter

from observability import metrics

router = APIRouter()


@router.get("/metrics", summary="Runtime counters and latency stats")
def get_metrics() -> dict:
    return metrics.snapshot()
