"""
tests/conftest.py
─────────────────
Shared fixtures. Every test runs against a throwaway SQLite file so the
seeded demo database in data/ is never touched.

agent.tools and agent.guard_rules both read DB_PATH into a module-level
constant at import time, so the fixture patches the attribute on each module
rather than the environment variable.
"""

import sqlite3

import pytest

from agent import guard_rules, tools
from db.init_db import init_db


@pytest.fixture
def db(tmp_path, monkeypatch):
    """An empty, schema-initialised database wired into both agent modules."""
    path = tmp_path / "test_procurement.db"
    init_db(str(path))

    monkeypatch.setattr(tools, "DB_PATH", str(path))
    monkeypatch.setattr(guard_rules, "DB_PATH", str(path))

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


@pytest.fixture
def make_vendor(db):
    def _make(name: str, status: str = "active") -> int:
        cur = db.execute(
            "INSERT INTO vendors (name, status) VALUES (?, ?)", (name, status)
        )
        db.commit()
        return cur.lastrowid

    return _make


@pytest.fixture
def make_po(db):
    def _make(
        po_number: str,
        vendor_id: int,
        amount: float = 1_000.0,
        status: str = "pending",
        created_at: str | None = None,
    ) -> int:
        if created_at is None:
            cur = db.execute(
                """
                INSERT INTO purchase_orders (po_number, vendor_id, amount, status)
                VALUES (?, ?, ?, ?)
                """,
                (po_number, vendor_id, amount, status),
            )
        else:
            cur = db.execute(
                """
                INSERT INTO purchase_orders
                    (po_number, vendor_id, amount, status, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (po_number, vendor_id, amount, status, created_at),
            )
        db.commit()
        return cur.lastrowid

    return _make
