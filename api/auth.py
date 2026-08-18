"""
api/auth.py
────────────
Minimal API-key auth. Each key maps to a role, and optionally an identity
(the string recorded as approved_by/actor on writes), via the API_KEYS env
var:

    API_KEYS="key1:requester:alice@company.com,key2:approver:bob@company.com"

Identity may be omitted (key:role) and defaults to "<role>@procureguard.local".

Callers send the key in the X-API-Key header. The resolved AuthContext is
the authoritative source of truth for the caller's role and identity —
client-supplied role/approved_by fields are no longer trusted; see
api/routers/chat.py and main.py's guard_node.
"""

import logging
import os
from dataclasses import dataclass

from fastapi import Header, HTTPException

log = logging.getLogger(__name__)

VALID_ROLES = {"requester", "approver", "finance", "admin"}


@dataclass(frozen=True)
class AuthContext:
    role: str
    identity: str


def _parse_api_keys(raw: str) -> dict[str, AuthContext]:
    keys: dict[str, AuthContext] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":", 2)
        if len(parts) < 2:
            log.warning("Ignoring malformed API_KEYS entry (expected key:role[:identity]).")
            continue
        key, role = parts[0].strip(), parts[1].strip()
        identity = parts[2].strip() if len(parts) == 3 else ""
        if not key or role not in VALID_ROLES:
            log.warning("Ignoring malformed API_KEYS entry (expected key:role[:identity]).")
            continue
        keys[key] = AuthContext(role=role, identity=identity or f"{role}@procureguard.local")
    return keys


def _api_keys() -> dict[str, AuthContext]:
    # Parsed fresh per call (cheap) rather than cached at import time, so it
    # always reflects the current API_KEYS env var — tests set/change it per
    # case, and a module-level constant would freeze whatever value was set
    # at first import.
    return _parse_api_keys(os.getenv("API_KEYS", ""))


def require_api_key(x_api_key: str | None = Header(default=None)) -> AuthContext:
    """FastAPI dependency: validates X-API-Key and returns the caller's AuthContext."""
    keys = _api_keys()
    if not keys:
        raise HTTPException(
            status_code=503,
            detail="Server has no API_KEYS configured — set API_KEYS in .env.",
        )
    if not x_api_key or x_api_key not in keys:
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")
    return keys[x_api_key]
