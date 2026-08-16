"""
db/seed.py
──────────
Seeds the SQLite database with realistic test data for development and testing.

Data seeded:
    5 vendors  (active × 3, inactive × 1, blacklisted × 1)
    6 purchase orders (draft, pending, approved, rejected, closed)
    3 invoices (pending, paid, disputed)

Idempotent — skips inserts if data already exists (checks by name/number).

Usage:
    python db/seed.py
"""

import logging
import os
import sqlite3
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

DB_PATH: str = os.getenv("DB_PATH", "data/procurement.db")


def seed(db_path: str = DB_PATH) -> None:
    path = Path(db_path)
    if not path.exists():
        log.error("Database not found at %s — run db/init_db.py first.", path)
        return

    with sqlite3.connect(str(path)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON;")
        _seed_vendors(conn)
        _seed_purchase_orders(conn)
        _seed_invoices(conn)
        conn.commit()

    log.info("Seeding complete.")


# ── Vendors ───────────────────────────────────────────────────────────────────

VENDORS = [
    {
        "name": "Acme Corp",
        "contact": "Alice Johnson",
        "email": "alice@acmecorp.com",
        "phone": "+1-555-0101",
        "address": "123 Main St, Springfield, IL",
        "status": "active",
    },
    {
        "name": "TechSupply Inc",
        "contact": "Bob Martinez",
        "email": "bob@techsupply.com",
        "phone": "+1-555-0202",
        "address": "456 Tech Ave, Austin, TX",
        "status": "active",
    },
    {
        "name": "Global Logistics Ltd",
        "contact": "Carol White",
        "email": "carol@globallogistics.com",
        "phone": "+1-555-0303",
        "address": "789 Harbor Blvd, Los Angeles, CA",
        "status": "active",
    },
    {
        "name": "OldParts Co",
        "contact": "Dave Brown",
        "email": "dave@oldparts.com",
        "phone": "+1-555-0404",
        "address": "321 Rust Lane, Detroit, MI",
        "status": "inactive",
    },
    {
        "name": "ShadyDeals LLC",
        "contact": "Eve Black",
        "email": "eve@shadydeals.net",
        "phone": "+1-555-0505",
        "address": "999 Dark Alley, Las Vegas, NV",
        "status": "blacklisted",
    },
]


def _seed_vendors(conn: sqlite3.Connection) -> None:
    inserted = 0
    for v in VENDORS:
        existing = conn.execute(
            "SELECT id FROM vendors WHERE name = ?", (v["name"],)
        ).fetchone()
        if existing:
            log.debug("Vendor '%s' already exists, skipping.", v["name"])
            continue
        conn.execute(
            """
            INSERT INTO vendors (name, contact, email, phone, address, status)
            VALUES (:name, :contact, :email, :phone, :address, :status)
            """,
            v,
        )
        inserted += 1
    log.info("Vendors: %d inserted, %d already existed.", inserted, len(VENDORS) - inserted)


# ── Purchase Orders ───────────────────────────────────────────────────────────

PURCHASE_ORDERS = [
    {
        "po_number": "PO-100001",
        "vendor_name": "Acme Corp",
        "description": "Office supplies Q3 2025",
        "amount": 12500.00,
        "requested_by": "john.doe@company.com",
        "status": "approved",
        "approved_by": "manager@company.com",
    },
    {
        "po_number": "PO-100002",
        "vendor_name": "TechSupply Inc",
        "description": "Server hardware upgrade — 10x Dell PowerEdge R750",
        "amount": 87500.00,
        "requested_by": "it.dept@company.com",
        "status": "pending",
        "approved_by": None,
    },
    {
        "po_number": "PO-100003",
        "vendor_name": "Global Logistics Ltd",
        "description": "Freight services contract — Q4 2025",
        "amount": 34000.00,
        "requested_by": "ops@company.com",
        "status": "draft",
        "approved_by": None,
    },
    {
        "po_number": "PO-100004",
        "vendor_name": "Acme Corp",
        "description": "Emergency ergonomic chairs procurement",
        "amount": 8200.00,
        "requested_by": "hr@company.com",
        "status": "rejected",
        "approved_by": None,
    },
    {
        "po_number": "PO-100005",
        "vendor_name": "TechSupply Inc",
        "description": "Software licences — Microsoft 365 annual",
        "amount": 45000.00,
        "requested_by": "it.dept@company.com",
        "status": "closed",
        "approved_by": "manager@company.com",
    },
    {
        "po_number": "PO-100006",
        "vendor_name": "Global Logistics Ltd",
        "description": "Warehouse pallet racking installation",
        "amount": 21000.00,
        "requested_by": "warehouse@company.com",
        "status": "approved",
        "approved_by": "manager@company.com",
    },
]


def _seed_purchase_orders(conn: sqlite3.Connection) -> None:
    inserted = 0
    for po in PURCHASE_ORDERS:
        existing = conn.execute(
            "SELECT id FROM purchase_orders WHERE po_number = ?", (po["po_number"],)
        ).fetchone()
        if existing:
            log.debug("PO '%s' already exists, skipping.", po["po_number"])
            continue

        vendor = conn.execute(
            "SELECT id FROM vendors WHERE name = ?", (po["vendor_name"],)
        ).fetchone()
        if not vendor:
            log.warning("Vendor '%s' not found — skipping PO %s.", po["vendor_name"], po["po_number"])
            continue

        conn.execute(
            """
            INSERT INTO purchase_orders
                (po_number, vendor_id, description, amount, requested_by, status, approved_by)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                po["po_number"], vendor["id"], po["description"],
                po["amount"], po["requested_by"], po["status"], po["approved_by"],
            ),
        )
        inserted += 1
    log.info("Purchase orders: %d inserted, %d already existed.", inserted, len(PURCHASE_ORDERS) - inserted)


# ── Invoices ──────────────────────────────────────────────────────────────────

INVOICES = [
    {
        "invoice_number": "INV-2025-001",
        "po_number": "PO-100001",
        "vendor_name": "Acme Corp",
        "amount": 12500.00,
        "currency": "USD",
        "due_date": "2025-10-15",
        "paid_date": "2025-10-12",
        "status": "paid",
    },
    {
        "invoice_number": "INV-2025-002",
        "po_number": "PO-100006",
        "vendor_name": "Global Logistics Ltd",
        "amount": 21000.00,
        "currency": "USD",
        "due_date": "2025-11-30",
        "paid_date": None,
        "status": "pending",
    },
    {
        "invoice_number": "INV-2025-003",
        "po_number": "PO-100005",
        "vendor_name": "TechSupply Inc",
        "amount": 52000.00,   # deliberate mismatch — $7k over PO amount
        "currency": "USD",
        "due_date": "2025-09-01",
        "paid_date": None,
        "status": "disputed",
    },
]


def _seed_invoices(conn: sqlite3.Connection) -> None:
    inserted = 0
    for inv in INVOICES:
        existing = conn.execute(
            "SELECT id FROM invoices WHERE invoice_number = ?", (inv["invoice_number"],)
        ).fetchone()
        if existing:
            log.debug("Invoice '%s' already exists, skipping.", inv["invoice_number"])
            continue

        po = conn.execute(
            "SELECT id FROM purchase_orders WHERE po_number = ?", (inv["po_number"],)
        ).fetchone()
        vendor = conn.execute(
            "SELECT id FROM vendors WHERE name = ?", (inv["vendor_name"],)
        ).fetchone()

        if not po or not vendor:
            log.warning("PO or vendor not found for invoice %s — skipping.", inv["invoice_number"])
            continue

        conn.execute(
            """
            INSERT INTO invoices
                (invoice_number, po_id, vendor_id, amount, currency,
                 due_date, paid_date, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                inv["invoice_number"], po["id"], vendor["id"],
                inv["amount"], inv["currency"],
                inv["due_date"], inv["paid_date"], inv["status"],
            ),
        )
        inserted += 1
    log.info("Invoices: %d inserted, %d already existed.", inserted, len(INVOICES) - inserted)


if __name__ == "__main__":
    seed()
