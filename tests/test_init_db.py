"""
tests/test_init_db.py
────────────────────────
Unit tests for db/init_db.py — schema creation, idempotency, and the CHECK
constraints that keep status columns honest.
"""

import sqlite3

import pytest

from db.init_db import init_db


def test_creates_all_expected_tables(tmp_path):
    db_path = tmp_path / "t.db"
    init_db(str(db_path))

    with sqlite3.connect(str(db_path)) as conn:
        tables = {
            row[0] for row in
            conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
    assert {"vendors", "purchase_orders", "invoices", "audit_log"} <= tables


def test_is_idempotent(tmp_path):
    db_path = tmp_path / "t.db"
    init_db(str(db_path))
    init_db(str(db_path))  # must not raise or duplicate schema

    with sqlite3.connect(str(db_path)) as conn:
        tables = [
            row[0] for row in
            conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        ]
    assert len(tables) == len(set(tables))


def test_creates_parent_directories(tmp_path):
    db_path = tmp_path / "nested" / "dir" / "t.db"
    init_db(str(db_path))
    assert db_path.exists()


def test_foreign_keys_are_enforced(tmp_path):
    db_path = tmp_path / "t.db"
    init_db(str(db_path))

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("PRAGMA foreign_keys=ON;")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO purchase_orders (po_number, vendor_id) VALUES ('PO-1', 999)"
            )


def test_vendor_status_check_constraint_rejects_invalid_values(tmp_path):
    db_path = tmp_path / "t.db"
    init_db(str(db_path))

    with sqlite3.connect(str(db_path)) as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO vendors (name, status) VALUES ('X', 'not-a-real-status')"
            )


def test_purchase_order_status_check_constraint_rejects_invalid_values(tmp_path):
    db_path = tmp_path / "t.db"
    init_db(str(db_path))

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("INSERT INTO vendors (name) VALUES ('X')")
        vendor_id = conn.execute("SELECT id FROM vendors WHERE name='X'").fetchone()[0]
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO purchase_orders (po_number, vendor_id, status) VALUES (?, ?, ?)",
                ("PO-1", vendor_id, "not-a-real-status"),
            )


def test_vendor_name_is_unique(tmp_path):
    db_path = tmp_path / "t.db"
    init_db(str(db_path))

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("INSERT INTO vendors (name) VALUES ('Dup Corp')")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO vendors (name) VALUES ('Dup Corp')")


def test_vendor_status_defaults_to_active(tmp_path):
    db_path = tmp_path / "t.db"
    init_db(str(db_path))

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("INSERT INTO vendors (name) VALUES ('X')")
        status = conn.execute("SELECT status FROM vendors WHERE name='X'").fetchone()[0]
    assert status == "active"
