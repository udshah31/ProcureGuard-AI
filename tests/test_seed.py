"""
tests/test_seed.py
──────────────────────
Unit tests for db/seed.py — idempotency, foreign-key linking, and the
guard-relevant fixture data (a blacklisted vendor, a self-approval case)
that the evals and guard-rule tests depend on existing.

Assertions are driven by seed.py's own VENDORS/PURCHASE_ORDERS/INVOICES
lists rather than hardcoded counts, so they track the dataset rather than
duplicating it.
"""

import sqlite3

import pytest

from db.init_db import init_db
from db.seed import INVOICES, PURCHASE_ORDERS, VENDORS, seed


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "t.db"
    init_db(str(path))
    return str(path)


def test_seeds_all_vendors(db_path):
    seed(db_path)
    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM vendors").fetchone()[0]
    assert count == len(VENDORS)


def test_seeds_all_purchase_orders(db_path):
    seed(db_path)
    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM purchase_orders").fetchone()[0]
    assert count == len(PURCHASE_ORDERS)


def test_seeds_all_invoices(db_path):
    seed(db_path)
    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM invoices").fetchone()[0]
    assert count == len(INVOICES)


def test_is_idempotent(db_path):
    seed(db_path)
    seed(db_path)  # second run must not duplicate rows

    with sqlite3.connect(db_path) as conn:
        vendors = conn.execute("SELECT COUNT(*) FROM vendors").fetchone()[0]
        pos = conn.execute("SELECT COUNT(*) FROM purchase_orders").fetchone()[0]
        invoices = conn.execute("SELECT COUNT(*) FROM invoices").fetchone()[0]

    assert vendors == len(VENDORS)
    assert pos == len(PURCHASE_ORDERS)
    assert invoices == len(INVOICES)


def test_purchase_orders_link_to_the_correct_vendor(db_path):
    seed(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        for po in PURCHASE_ORDERS:
            row = conn.execute(
                """
                SELECT v.name FROM purchase_orders po
                JOIN vendors v ON po.vendor_id = v.id
                WHERE po.po_number = ?
                """,
                (po["po_number"],),
            ).fetchone()
            assert row["name"] == po["vendor_name"]


def test_invoices_link_to_the_correct_po_and_vendor(db_path):
    seed(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        for inv in INVOICES:
            row = conn.execute(
                """
                SELECT po.po_number, v.name AS vendor_name
                FROM invoices i
                JOIN purchase_orders po ON i.po_id = po.id
                JOIN vendors v ON i.vendor_id = v.id
                WHERE i.invoice_number = ?
                """,
                (inv["invoice_number"],),
            ).fetchone()
            assert row["po_number"] == inv["po_number"]
            assert row["vendor_name"] == inv["vendor_name"]


def test_does_nothing_if_database_does_not_exist(tmp_path):
    """seed() must not create the database file itself — that's init_db's
    job. Calling it against a missing path should be a no-op, not a crash."""
    missing_path = str(tmp_path / "does-not-exist.db")
    seed(missing_path)  # must not raise
    import os
    assert not os.path.exists(missing_path)


# ── Guard-relevant fixture data ─────────────────────────────────────────────

def test_seeds_at_least_one_blacklisted_vendor():
    assert any(v["status"] == "blacklisted" for v in VENDORS)


def test_seeds_at_least_one_inactive_vendor():
    assert any(v["status"] == "inactive" for v in VENDORS)


def test_seeds_a_po_over_the_fifty_k_guard_threshold():
    assert any(po["amount"] > 50_000 for po in PURCHASE_ORDERS)
