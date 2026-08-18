"""
api/models.py
─────────────
Pydantic v2 request and response schemas for all ProcureGuard API endpoints.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════════════════════
# /chat
# ══════════════════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    session_id: str = Field(
        ...,
        description="Unique session identifier for multi-turn conversation.",
        examples=["user-abc-123"],
    )
    message: str = Field(
        ...,
        min_length=1,
        description="The user's message to the procurement agent.",
        examples=["Look up vendor Acme Corp"],
    )


class ChatResponse(BaseModel):
    session_id: str
    reply: str = Field(description="The agent's text response.")
    risk_flags: list[str] = Field(
        default_factory=list,
        description="Any compliance warnings raised during this turn.",
    )
    tool_calls_made: list[str] = Field(
        default_factory=list,
        description="Names of tools the agent invoked to produce this reply.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# /vendors
# ══════════════════════════════════════════════════════════════════════════════

class VendorOut(BaseModel):
    id: int
    name: str
    contact: str | None
    email: str | None
    phone: str | None
    address: str | None
    status: str
    created_at: str
    updated_at: str


# ══════════════════════════════════════════════════════════════════════════════
# /purchase-orders
# ══════════════════════════════════════════════════════════════════════════════

class PurchaseOrderOut(BaseModel):
    id: int
    po_number: str
    vendor_id: int
    vendor_name: str
    description: str | None
    amount: float
    currency: str
    status: str
    requested_by: str | None
    approved_by: str | None
    created_at: str
    updated_at: str


# ══════════════════════════════════════════════════════════════════════════════
# /invoices
# ══════════════════════════════════════════════════════════════════════════════

class InvoiceOut(BaseModel):
    id: int
    invoice_number: str
    po_id: int
    po_number: str
    vendor_id: int
    vendor_name: str
    amount: float
    currency: str
    due_date: str | None
    paid_date: str | None
    status: str
    created_at: str
    updated_at: str


# ══════════════════════════════════════════════════════════════════════════════
# /health
# ══════════════════════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    status: str = "ok"
    model_name: str
    db_path: str
    active_sessions: int
