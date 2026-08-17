"""
agent/tools.py
──────────────
All 8 procurement tools for the ProcureGuard agent.

Tools:
    Existing (migrated from main.py):
        lookup_vendor           – search vendor registry by name
        create_purchase_order   – create a draft PO
        check_invoice_status    – look up an invoice by number

    New:
        approve_purchase_order  – approve a pending PO (writes audit_log)
        reject_purchase_order   – reject a PO with reason (writes audit_log)
        add_vendor              – register a new vendor
        flag_suspicious_invoice – mark an invoice as disputed
        vendor_risk_score       – compute heuristic risk score (0–100)
"""

import json
import logging
import os
import random
import sqlite3
import string
from datetime import datetime

from langchain_core.tools import tool

log = logging.getLogger(__name__)

DB_PATH: str = os.getenv("DB_PATH", "data/procurement.db")

PO_NUMBER_ATTEMPTS = 5


# ── DB helper ─────────────────────────────────────────────────────────────────

def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def _norm_id(value: str) -> str:
    """Canonicalise a PO/invoice number. SQLite '=' is case-sensitive, so an
    un-normalised 'po-100002' reads as a missing record rather than a match."""
    return (value or "").strip().upper()


def _write_audit(
    conn: sqlite3.Connection,
    entity_type: str,
    entity_id: int,
    action: str,
    actor: str,
    details: dict,
) -> None:
    """Insert a row into audit_log inside an existing connection."""
    conn.execute(
        """
        INSERT INTO audit_log (entity_type, entity_id, action, actor, details)
        VALUES (?, ?, ?, ?, ?)
        """,
        (entity_type, entity_id, action, actor, json.dumps(details)),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Existing Tools (migrated)
# ══════════════════════════════════════════════════════════════════════════════

@tool
def lookup_vendor(name: str) -> str:
    """
    Look up a vendor by name (partial match).
    Returns vendor details including status, or a 'not found' message.
    """
    with _get_db() as conn:
        row = conn.execute(
            """
            SELECT id, name, contact, email, phone, status
            FROM vendors
            WHERE name LIKE ?
            """,
            (f"%{name}%",),
        ).fetchone()

    if row:
        return (
            f"Vendor found — ID: {row['id']} | Name: {row['name']} | "
            f"Contact: {row['contact']} | Email: {row['email']} | "
            f"Phone: {row['phone']} | Status: {row['status'].upper()}"
        )
    return f"No vendor found matching '{name}'."


@tool
def create_purchase_order(
    vendor_name: str,
    description: str,
    amount: float,
    requested_by: str,
) -> str:
    """
    Create a new draft purchase order for the named vendor.
    ONLY use these exact parameters:
      - vendor_name: str  (must match an existing vendor name)
      - description: str  (what is being purchased)
      - amount: float     (total value in USD)
      - requested_by: str (email of the requester)
    The PO starts in 'draft' status and must be approved separately.
    Returns the generated PO number on success.
    Do NOT pass any other parameters.
    """
    po_number = "PO-" + "".join(random.choices(string.digits, k=6))

    with _get_db() as conn:
        vendor = conn.execute(
            "SELECT id FROM vendors WHERE name LIKE ?", (f"%{vendor_name}%",)
        ).fetchone()

        if not vendor:
            return (
                f"Cannot create PO: vendor '{vendor_name}' not found. "
                "Register them first with the add_vendor tool."
            )

        # po_number is random against a UNIQUE column, so collisions are a
        # question of when, not if. Retry rather than surfacing IntegrityError.
        for attempt in range(PO_NUMBER_ATTEMPTS):
            try:
                cursor = conn.execute(
                    """
                    INSERT INTO purchase_orders
                        (po_number, vendor_id, description, amount, requested_by, status)
                    VALUES (?, ?, ?, ?, ?, 'draft')
                    """,
                    (po_number, vendor["id"], description, amount, requested_by),
                )
                break
            except sqlite3.IntegrityError:
                if attempt == PO_NUMBER_ATTEMPTS - 1:
                    return "Could not allocate a unique PO number. Please retry."
                po_number = "PO-" + "".join(random.choices(string.digits, k=6))

        _write_audit(conn, "purchase_order", cursor.lastrowid, "created", requested_by,
                     {"po_number": po_number, "amount": amount, "description": description})
        conn.commit()

    log.info("Created PO %s for vendor '%s' (amount: %.2f)", po_number, vendor_name, amount)
    return (
        f"✅ Purchase order created. PO Number: {po_number} | "
        f"Vendor: {vendor_name} | Amount: ${amount:,.2f} | Status: DRAFT"
    )


@tool
def check_invoice_status(invoice_number: str) -> str:
    """
    Check the current status of an invoice by its invoice number.
    Returns amount, currency, due date, and status.
    """
    invoice_number = _norm_id(invoice_number)
    with _get_db() as conn:
        row = conn.execute(
            """
            SELECT i.invoice_number, i.amount, i.currency, i.status,
                   i.due_date, i.paid_date, v.name AS vendor_name
            FROM invoices i
            JOIN vendors v ON i.vendor_id = v.id
            WHERE i.invoice_number = ?
            """,
            (invoice_number,),
        ).fetchone()

    if row:
        paid_info = f" | Paid: {row['paid_date']}" if row["paid_date"] else ""
        return (
            f"Invoice {row['invoice_number']} [{row['status'].upper()}] — "
            f"Vendor: {row['vendor_name']} | "
            f"Amount: {row['currency']} {row['amount']:,.2f} | "
            f"Due: {row['due_date']}{paid_info}"
        )
    return f"Invoice '{invoice_number}' not found."


# ══════════════════════════════════════════════════════════════════════════════
# New Tools
# ══════════════════════════════════════════════════════════════════════════════

@tool
def approve_purchase_order(po_number: str, approved_by: str) -> str:
    """
    Approve a purchase order, moving its status from 'pending' to 'approved'.
    Records the approver name and writes to the audit log.
    NOTE: Guard rules are checked by the agent before calling this tool.
    """
    po_number = _norm_id(po_number)
    with _get_db() as conn:
        po = conn.execute(
            "SELECT id, status, amount, vendor_id FROM purchase_orders WHERE po_number = ?",
            (po_number,),
        ).fetchone()

        if not po:
            return f"PO '{po_number}' not found."
        if po["status"] not in ("draft", "pending", "approved"):
            return (
                f"PO '{po_number}' cannot be approved — current status: "
                f"{po['status'].upper()}."
            )

        # Fold the status check into the UPDATE itself so it's atomic with the
        # write — a prior SELECT-then-UPDATE let two concurrent approvals both
        # read 'pending' before either committed, silently double-approving.
        cursor = conn.execute(
            """
            UPDATE purchase_orders
            SET status = 'approved', approved_by = ?, updated_at = datetime('now')
            WHERE po_number = ? AND status IN ('draft', 'pending')
            """,
            (approved_by, po_number),
        )
        if cursor.rowcount == 0:
            return f"PO '{po_number}' is already approved."

        _write_audit(
            conn, "purchase_order", po["id"], "approved", approved_by,
            {"po_number": po_number, "amount": po["amount"]},
        )
        conn.commit()

    log.info("PO %s approved by %s", po_number, approved_by)
    return f"✅ PO {po_number} has been APPROVED by {approved_by}."


@tool
def reject_purchase_order(po_number: str, rejected_by: str, reason: str) -> str:
    """
    Reject a purchase order with a mandatory reason.
    Status moves to 'rejected' and the reason is logged to audit_log.
    """
    po_number = _norm_id(po_number)
    with _get_db() as conn:
        po = conn.execute(
            "SELECT id, status FROM purchase_orders WHERE po_number = ?",
            (po_number,),
        ).fetchone()

        if not po:
            return f"PO '{po_number}' not found."
        if po["status"] == "closed":
            return f"PO '{po_number}' is closed and cannot be rejected."

        cursor = conn.execute(
            """
            UPDATE purchase_orders
            SET status = 'rejected', updated_at = datetime('now')
            WHERE po_number = ? AND status != 'rejected'
            """,
            (po_number,),
        )
        if cursor.rowcount == 0:
            return f"PO '{po_number}' is already rejected."

        _write_audit(
            conn, "purchase_order", po["id"], "rejected", rejected_by,
            {"po_number": po_number, "reason": reason},
        )
        conn.commit()

    log.info("PO %s rejected by %s. Reason: %s", po_number, rejected_by, reason)
    return f"❌ PO {po_number} has been REJECTED by {rejected_by}. Reason: {reason}"


@tool
def add_vendor(
    name: str,
    contact: str,
    email: str,
    phone: str = "",
    address: str = "",
) -> str:
    """
    Register a new vendor in the system.
    Vendor starts with 'active' status. Returns the new vendor ID.
    """
    with _get_db() as conn:
        # ON CONFLICT keeps the existence check atomic with the insert; a prior
        # SELECT-then-INSERT let two concurrent registrations both see "absent"
        # and the loser got a raw IntegrityError instead of this message.
        cursor = conn.execute(
            """
            INSERT INTO vendors (name, contact, email, phone, address, status)
            VALUES (?, ?, ?, ?, ?, 'active')
            ON CONFLICT(name) DO NOTHING
            """,
            (name, contact, email, phone, address),
        )
        if cursor.rowcount == 0:
            existing = conn.execute(
                "SELECT id FROM vendors WHERE name = ?", (name,)
            ).fetchone()
            return f"Vendor '{name}' already exists (ID: {existing['id']})."

        vendor_id = cursor.lastrowid
        _write_audit(conn, "vendor", vendor_id, "created", "agent",
                     {"name": name, "email": email})
        conn.commit()

    log.info("New vendor registered: %s (ID: %d)", name, vendor_id)
    return f"✅ Vendor '{name}' registered successfully. Vendor ID: {vendor_id}"


@tool
def flag_suspicious_invoice(
    invoice_number: str,
    flagged_by: str,
    reason: str,
) -> str:
    """
    Flag an invoice as suspicious/disputed.
    Sets invoice status to 'disputed' and records the reason in the audit log.
    Use when an invoice amount doesn't match the PO, or appears fraudulent.
    """
    invoice_number = _norm_id(invoice_number)
    with _get_db() as conn:
        inv = conn.execute(
            "SELECT id, status, amount FROM invoices WHERE invoice_number = ?",
            (invoice_number,),
        ).fetchone()

        if not inv:
            return f"Invoice '{invoice_number}' not found."
        if inv["status"] == "paid":
            return (
                f"⚠️  Invoice '{invoice_number}' is already PAID. "
                "Flagging for review — contact finance to initiate a reversal."
            )

        cursor = conn.execute(
            """
            UPDATE invoices
            SET status = 'disputed', updated_at = datetime('now')
            WHERE invoice_number = ? AND status != 'disputed'
            """,
            (invoice_number,),
        )
        if cursor.rowcount == 0:
            return f"Invoice '{invoice_number}' is already marked as disputed."

        _write_audit(
            conn, "invoice", inv["id"], "flagged_suspicious", flagged_by,
            {"invoice_number": invoice_number, "reason": reason, "amount": inv["amount"]},
        )
        conn.commit()

    log.warning("Invoice %s flagged as suspicious by %s: %s", invoice_number, flagged_by, reason)
    return (
        f"🚨 Invoice {invoice_number} has been flagged as DISPUTED. "
        f"Flagged by: {flagged_by} | Reason: {reason}"
    )


@tool
def vendor_risk_score(vendor_name: str) -> str:
    """
    Compute a heuristic risk score (0–100) for a vendor based on their history.
    Higher score = higher risk. Factors: dispute rate, PO volume, avg amount, status.
    Returns a score, risk tier (LOW/MEDIUM/HIGH/CRITICAL), and breakdown.
    """
    with _get_db() as conn:
        vendor = conn.execute(
            "SELECT id, name, status FROM vendors WHERE name LIKE ?",
            (f"%{vendor_name}%",),
        ).fetchone()

        if not vendor:
            return f"Vendor '{vendor_name}' not found."

        vid = vendor["id"]

        po_stats = conn.execute(
            """
            SELECT
                COUNT(*) AS total_pos,
                AVG(amount) AS avg_amount,
                SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END) AS rejected_pos
            FROM purchase_orders WHERE vendor_id = ?
            """,
            (vid,),
        ).fetchone()

        inv_stats = conn.execute(
            """
            SELECT
                COUNT(*) AS total_invoices,
                SUM(CASE WHEN status = 'disputed' THEN 1 ELSE 0 END) AS disputed_invoices
            FROM invoices WHERE vendor_id = ?
            """,
            (vid,),
        ).fetchone()

    score = 0
    breakdown = []

    # Factor 1: Vendor status
    if vendor["status"] == "blacklisted":
        score += 60
        breakdown.append("Blacklisted vendor (+60)")
    elif vendor["status"] == "inactive":
        score += 20
        breakdown.append("Inactive vendor (+20)")

    # Factor 2: PO rejection rate
    total_pos = po_stats["total_pos"] or 0
    rejected_pos = po_stats["rejected_pos"] or 0
    if total_pos > 0:
        rejection_rate = rejected_pos / total_pos
        pts = int(rejection_rate * 25)
        score += pts
        breakdown.append(f"PO rejection rate {rejection_rate:.0%} (+{pts})")

    # Factor 3: Invoice dispute rate
    total_inv = inv_stats["total_invoices"] or 0
    disputed_inv = inv_stats["disputed_invoices"] or 0
    if total_inv > 0:
        dispute_rate = disputed_inv / total_inv
        pts = int(dispute_rate * 30)
        score += pts
        breakdown.append(f"Invoice dispute rate {dispute_rate:.0%} (+{pts})")

    # Factor 4: Average PO amount (high-value vendors = higher inherent risk)
    avg_amount = po_stats["avg_amount"] or 0
    if avg_amount > 100_000:
        score += 10
        breakdown.append("Avg PO > $100k (+10)")
    elif avg_amount > 50_000:
        score += 5
        breakdown.append("Avg PO > $50k (+5)")

    score = min(score, 100)  # cap at 100

    # Tier
    if score >= 75:
        tier = "🔴 CRITICAL"
    elif score >= 50:
        tier = "🟠 HIGH"
    elif score >= 25:
        tier = "🟡 MEDIUM"
    else:
        tier = "🟢 LOW"

    breakdown_str = " | ".join(breakdown) if breakdown else "No risk factors detected"
    return (
        f"Vendor Risk Score — {vendor['name']}\n"
        f"Score: {score}/100 | Risk Tier: {tier}\n"
        f"Breakdown: {breakdown_str}\n"
        f"POs: {total_pos} total, {rejected_pos} rejected | "
        f"Invoices: {total_inv} total, {disputed_inv} disputed"
    )


# ── Tool registry ─────────────────────────────────────────────────────────────
TOOLS = [
    lookup_vendor,
    create_purchase_order,
    check_invoice_status,
    approve_purchase_order,
    reject_purchase_order,
    add_vendor,
    flag_suspicious_invoice,
    vendor_risk_score,
]

TOOL_MAP: dict = {t.name: t for t in TOOLS}
