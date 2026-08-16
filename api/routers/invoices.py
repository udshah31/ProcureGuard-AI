"""
api/routers/invoices.py
───────────────────────
GET /api/v1/invoices                         — List invoices (optional filters)
GET /api/v1/invoices/{invoice_number}        — Get a single invoice
"""

import logging
import os
import sqlite3
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from api.models import InvoiceOut

log = logging.getLogger(__name__)
router = APIRouter()

DB_PATH: str = os.getenv("DB_PATH", "data/procurement.db")

VALID_INV_STATUSES = {"pending", "approved", "paid", "disputed", "cancelled"}


def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_invoice(row: sqlite3.Row) -> InvoiceOut:
    return InvoiceOut(
        id=row["id"],
        invoice_number=row["invoice_number"],
        po_id=row["po_id"],
        po_number=row["po_number"],
        vendor_id=row["vendor_id"],
        vendor_name=row["vendor_name"],
        amount=row["amount"],
        currency=row["currency"],
        due_date=row["due_date"],
        paid_date=row["paid_date"],
        status=row["status"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


_BASE_QUERY = """
    SELECT
        i.id, i.invoice_number, i.po_id, po.po_number,
        i.vendor_id, v.name AS vendor_name,
        i.amount, i.currency, i.due_date, i.paid_date,
        i.status, i.created_at, i.updated_at
    FROM invoices i
    JOIN purchase_orders po ON i.po_id = po.id
    JOIN vendors v ON i.vendor_id = v.id
    WHERE 1=1
"""


@router.get(
    "/invoices",
    response_model=list[InvoiceOut],
    summary="List invoices",
)
def list_invoices(
    status: Optional[str] = Query(
        default=None,
        description="Filter by status: pending | approved | paid | disputed | cancelled",
        examples=["disputed"],
    ),
    vendor_id: Optional[int] = Query(
        default=None,
        description="Filter by vendor ID.",
    ),
    po_id: Optional[int] = Query(
        default=None,
        description="Filter by purchase order ID.",
    ),
    overdue_only: bool = Query(
        default=False,
        description="If true, return only invoices past their due date and not yet paid.",
    ),
) -> list[InvoiceOut]:
    """
    Returns invoices with optional filters.

    - `status`: pending, approved, paid, disputed, cancelled
    - `vendor_id`: filter by vendor
    - `po_id`: filter by purchase order
    - `overdue_only`: returns unpaid invoices past their due date
    """
    query = _BASE_QUERY
    params: list = []

    if status:
        if status not in VALID_INV_STATUSES:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid status '{status}'. Must be one of: {sorted(VALID_INV_STATUSES)}",
            )
        query += " AND i.status = ?"
        params.append(status)

    if vendor_id is not None:
        query += " AND i.vendor_id = ?"
        params.append(vendor_id)

    if po_id is not None:
        query += " AND i.po_id = ?"
        params.append(po_id)

    if overdue_only:
        query += (
            " AND i.due_date < date('now')"
            " AND i.status NOT IN ('paid', 'cancelled')"
        )

    query += " ORDER BY i.created_at DESC"

    with _get_db() as conn:
        rows = conn.execute(query, params).fetchall()

    return [_row_to_invoice(r) for r in rows]


@router.get(
    "/invoices/{invoice_number}",
    response_model=InvoiceOut,
    summary="Get an invoice by invoice number",
)
def get_invoice(invoice_number: str) -> InvoiceOut:
    """Returns a single invoice by its invoice number (e.g. `INV-2025-003`)."""
    with _get_db() as conn:
        row = conn.execute(
            _BASE_QUERY + " AND i.invoice_number = ?", (invoice_number,)
        ).fetchone()

    if not row:
        raise HTTPException(
            status_code=404, detail=f"Invoice '{invoice_number}' not found."
        )

    return _row_to_invoice(row)
