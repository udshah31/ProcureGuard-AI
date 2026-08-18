"""
api/auth.py
────────────
Minimal API-key auth. Each key maps to a role (requester | approver |
finance | admin) via the API_KEYS env var:

    API_KEYS="key1:requester,key2:approver,key3:finance,key4:admin"

Callers send the key in the X-API-Key header. The resolved role is the
authoritative source of truth for the caller's role — client-supplied
role fields in request bodies (e.g. ChatRequest.role) are no longer
trusted; see api/routers/chat.py.
"""

import logging
import os

from fastapi import Header, HTTPException

log = logging.getLogger(__name__)

VALID_ROLES = {"requester", "approver", "finance", "admin"}


def _parse_api_keys(raw: str) -> dict[str, str]:
    keys: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            log.warning("Ignoring malformed API_KEYS entry (expected key:role).")
            continue
        key, _, role = pair.partition(":")
        key, role = key.strip(), role.strip()
        if not key or role not in VALID_ROLES:
            log.warning("Ignoring malformed API_KEYS entry (expected key:role).")
            continue
        keys[key] = role
    return keys


def _api_keys() -> dict[str, str]:
    # Parsed fresh per call (cheap) rather than cached at import time, so it
    # always reflects the current API_KEYS env var — tests set/change it per
    # case, and a module-level constant would freeze whatever value was set
    # at first import.
    return _parse_api_keys(os.getenv("API_KEYS", ""))


def require_api_key(x_api_key: str | None = Header(default=None)) -> str:
    """FastAPI dependency: validates X-API-Key and returns the caller's role."""
    keys = _api_keys()
    if not keys:
        raise HTTPException(
            status_code=503,
            detail="Server has no API_KEYS configured — set API_KEYS in .env.",
        )
    if not x_api_key or x_api_key not in keys:
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")
    return keys[x_api_key]
