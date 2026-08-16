"""
api/routers/vendors.py
──────────────────────
GET /api/v1/vendors          — List all vendors (optional ?status= filter)
GET /api/v1/vendors/{id}     — Get a single vendor by ID
"""

import logging
import os
import sqlite3
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from api.models import VendorOut

log = logging.getLogger(__name__)
router = APIRouter()

DB_PATH: str = os.getenv("DB_PATH", "data/procurement.db")


def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_vendor(row: sqlite3.Row) -> VendorOut:
    return VendorOut(
        id=row["id"],
        name=row["name"],
        contact=row["contact"],
        email=row["email"],
        phone=row["phone"],
        address=row["address"],
        status=row["status"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


@router.get(
    "/vendors",
    response_model=list[VendorOut],
    summary="List all vendors",
)
def list_vendors(
    status: Optional[str] = Query(
        default=None,
        description="Filter by status: active | inactive | blacklisted",
        examples=["active"],
    ),
    name: Optional[str] = Query(
        default=None,
        description="Partial name search (case-insensitive).",
        examples=["Acme"],
    ),
) -> list[VendorOut]:
    """
    Returns all registered vendors.

    - Filter by `status` (active, inactive, blacklisted).
    - Search by partial `name`.
    """
    query = "SELECT * FROM vendors WHERE 1=1"
    params: list = []

    if status:
        valid = {"active", "inactive", "blacklisted"}
        if status not in valid:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid status '{status}'. Must be one of: {sorted(valid)}",
            )
        query += " AND status = ?"
        params.append(status)

    if name:
        query += " AND name LIKE ?"
        params.append(f"%{name}%")

    query += " ORDER BY name"

    with _get_db() as conn:
        rows = conn.execute(query, params).fetchall()

    return [_row_to_vendor(r) for r in rows]


@router.get(
    "/vendors/{vendor_id}",
    response_model=VendorOut,
    summary="Get a vendor by ID",
)
def get_vendor(vendor_id: int) -> VendorOut:
    """Returns a single vendor by their numeric ID."""
    with _get_db() as conn:
        row = conn.execute(
            "SELECT * FROM vendors WHERE id = ?", (vendor_id,)
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail=f"Vendor {vendor_id} not found.")

    return _row_to_vendor(row)
