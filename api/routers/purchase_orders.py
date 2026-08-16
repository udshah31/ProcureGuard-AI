"""
api/routers/purchase_orders.py
───────────────────────────────
GET /api/v1/purchase-orders                  — List POs (optional filters)
GET /api/v1/purchase-orders/{po_number}      — Get a single PO by PO number
"""

import logging
import os
import sqlite3
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from api.models import PurchaseOrderOut

log = logging.getLogger(__name__)
router = APIRouter()

DB_PATH: str = os.getenv("DB_PATH", "data/procurement.db")

VALID_PO_STATUSES = {"draft", "pending", "approved", "rejected", "closed"}


def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_po(row: sqlite3.Row) -> PurchaseOrderOut:
    return PurchaseOrderOut(
        id=row["id"],
        po_number=row["po_number"],
        vendor_id=row["vendor_id"],
        vendor_name=row["vendor_name"],
        description=row["description"],
        amount=row["amount"],
        currency=row["currency"],
        status=row["status"],
        requested_by=row["requested_by"],
        approved_by=row["approved_by"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


_BASE_QUERY = """
    SELECT
        po.id, po.po_number, po.vendor_id, v.name AS vendor_name,
        po.description, po.amount, po.currency, po.status,
        po.requested_by, po.approved_by, po.created_at, po.updated_at
    FROM purchase_orders po
    JOIN vendors v ON po.vendor_id = v.id
    WHERE 1=1
"""


@router.get(
    "/purchase-orders",
    response_model=list[PurchaseOrderOut],
    summary="List purchase orders",
)
def list_purchase_orders(
    status: Optional[str] = Query(
        default=None,
        description="Filter by status: draft | pending | approved | rejected | closed",
        examples=["pending"],
    ),
    vendor_id: Optional[int] = Query(
        default=None,
        description="Filter by vendor ID.",
    ),
    min_amount: Optional[float] = Query(
        default=None,
        description="Filter POs with amount >= this value.",
        examples=[50000],
    ),
    max_amount: Optional[float] = Query(
        default=None,
        description="Filter POs with amount <= this value.",
    ),
) -> list[PurchaseOrderOut]:
    """
    Returns purchase orders with optional filters.

    - `status`: draft, pending, approved, rejected, closed
    - `vendor_id`: filter by vendor
    - `min_amount` / `max_amount`: range filter on PO amount
    """
    query = _BASE_QUERY
    params: list = []

    if status:
        if status not in VALID_PO_STATUSES:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid status '{status}'. Must be one of: {sorted(VALID_PO_STATUSES)}",
            )
        query += " AND po.status = ?"
        params.append(status)

    if vendor_id is not None:
        query += " AND po.vendor_id = ?"
        params.append(vendor_id)

    if min_amount is not None:
        query += " AND po.amount >= ?"
        params.append(min_amount)

    if max_amount is not None:
        query += " AND po.amount <= ?"
        params.append(max_amount)

    query += " ORDER BY po.created_at DESC"

    with _get_db() as conn:
        rows = conn.execute(query, params).fetchall()

    return [_row_to_po(r) for r in rows]


@router.get(
    "/purchase-orders/{po_number}",
    response_model=PurchaseOrderOut,
    summary="Get a purchase order by PO number",
)
def get_purchase_order(po_number: str) -> PurchaseOrderOut:
    """Returns a single purchase order by its PO number (e.g. `PO-100002`)."""
    with _get_db() as conn:
        row = conn.execute(
            _BASE_QUERY + " AND po.po_number = ?", (po_number,)
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail=f"Purchase order '{po_number}' not found.")

    return _row_to_po(row)
