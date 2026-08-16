"""
db/init_db.py
─────────────
Initialises the SQLite database for the Procurement Agent.

Run standalone:
    python db/init_db.py

Tables created (all idempotent via IF NOT EXISTS):
    vendors          – supplier registry
    purchase_orders  – PO lifecycle tracking
    invoices         – invoice records linked to POs
    audit_log        – immutable event trail
"""

import sqlite3
import logging
import os
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
# Allow override via environment variable so tests can point to an in-memory DB
DB_PATH: str = os.getenv("DB_PATH", "data/procurement.db")

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── DDL Statements ────────────────────────────────────────────────────────────
DDL_STATEMENTS: list[str] = [
    # Vendor / Supplier registry
    """
    CREATE TABLE IF NOT EXISTS vendors (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT    NOT NULL UNIQUE,
        contact     TEXT,
        email       TEXT,
        phone       TEXT,
        address     TEXT,
        status      TEXT    NOT NULL DEFAULT 'active'
                    CHECK(status IN ('active', 'inactive', 'blacklisted')),
        created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
        updated_at  TEXT    NOT NULL DEFAULT (datetime('now'))
    )
    """,

    # Purchase orders
    """
    CREATE TABLE IF NOT EXISTS purchase_orders (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        po_number    TEXT    NOT NULL UNIQUE,
        vendor_id    INTEGER NOT NULL REFERENCES vendors(id),
        description  TEXT,
        amount       REAL    NOT NULL DEFAULT 0.0,
        currency     TEXT    NOT NULL DEFAULT 'USD',
        status       TEXT    NOT NULL DEFAULT 'draft'
                     CHECK(status IN ('draft', 'pending', 'approved', 'rejected', 'closed')),
        requested_by TEXT,
        approved_by  TEXT,
        created_at   TEXT    NOT NULL DEFAULT (datetime('now')),
        updated_at   TEXT    NOT NULL DEFAULT (datetime('now'))
    )
    """,

    # Invoices linked to purchase orders
    """
    CREATE TABLE IF NOT EXISTS invoices (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        invoice_number  TEXT    NOT NULL UNIQUE,
        po_id           INTEGER NOT NULL REFERENCES purchase_orders(id),
        vendor_id       INTEGER NOT NULL REFERENCES vendors(id),
        amount          REAL    NOT NULL DEFAULT 0.0,
        currency        TEXT    NOT NULL DEFAULT 'USD',
        due_date        TEXT,
        paid_date       TEXT,
        status          TEXT    NOT NULL DEFAULT 'pending'
                        CHECK(status IN ('pending', 'approved', 'paid', 'disputed', 'cancelled')),
        created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
        updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
    )
    """,

    # Immutable audit / event trail
    """
    CREATE TABLE IF NOT EXISTS audit_log (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        entity_type TEXT    NOT NULL,    -- e.g. 'purchase_order', 'invoice', 'vendor'
        entity_id   INTEGER NOT NULL,
        action      TEXT    NOT NULL,    -- e.g. 'created', 'status_changed', 'approved'
        actor       TEXT,                -- user or agent identifier
        details     TEXT,                -- JSON blob with change details
        timestamp   TEXT    NOT NULL DEFAULT (datetime('now'))
    )
    """,
]


def init_db(db_path: str = DB_PATH) -> None:
    """Create the database file and all required tables."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Connecting to database: %s", path.resolve())
    with sqlite3.connect(str(path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")   # better concurrency
        conn.execute("PRAGMA foreign_keys=ON;")     # enforce FK constraints
        cursor = conn.cursor()

        for ddl in DDL_STATEMENTS:
            cursor.execute(ddl)

        conn.commit()

    tables = [row[0] for row in
              sqlite3.connect(str(path))
              .execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
              .fetchall()]
    log.info("Database ready. Tables: %s", tables)


if __name__ == "__main__":
    init_db()
