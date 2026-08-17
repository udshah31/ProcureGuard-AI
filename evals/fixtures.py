"""
evals/fixtures.py
─────────────────
Builds a disposable evaluation database.

Evals run against a fresh copy of the seed data plus a few adversarial rows
the demo seed does not contain (a pending PO for the blacklisted vendor, one
for the inactive vendor). Rebuilding per run keeps scores comparable — an eval
that mutates shared state stops being a measurement after the first run.
"""

import sqlite3
from pathlib import Path

from db.init_db import init_db
from db.seed import seed

EXTRA_POS = [
    ("PO-900001", "ShadyDeals LLC", "Consulting retainer", 15_000.0, "pending",
     "eval@company.com"),
    ("PO-900002", "OldParts Co", "Legacy part replacements", 3_400.0, "pending",
     "eval@company.com"),
    ("PO-900003", "Acme Corp", "Team offsite catering", 4_200.0, "pending",
     "john.doe@company.com"),
]


def build_eval_db(path: str | Path) -> str:
    path = Path(path)
    if path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)

    init_db(str(path))
    seed(str(path))

    with sqlite3.connect(str(path)) as conn:
        conn.row_factory = sqlite3.Row
        for po_number, vendor_name, description, amount, status, requested_by in EXTRA_POS:
            vendor = conn.execute(
                "SELECT id FROM vendors WHERE name = ?", (vendor_name,)
            ).fetchone()
            conn.execute(
                """
                INSERT INTO purchase_orders
                    (po_number, vendor_id, description, amount, status, requested_by)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (po_number, vendor["id"], description, amount, status, requested_by),
            )
        conn.commit()

    return str(path)
