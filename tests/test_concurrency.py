"""
tests/test_concurrency.py
──────────────────────────
Concurrent write-and-audit tools used to have a check-then-act race: they read
a row's status, decided what to do in Python, then wrote — with nothing
stopping two overlapping calls from both reading the pre-write status. Two
simultaneous approvals of the same PO could both "succeed", both write an
audit_log row, and leave the final approver attributed by write-order luck
rather than by who actually got there first.

The fix folds the status check into the UPDATE's WHERE clause and branches on
rowcount, making the check-and-write atomic. These tests fire real concurrent
threads (not sequential calls) at the same row and assert exactly one write
wins.
"""

import sqlite3
import threading

import pytest

from agent.tools import (
    add_vendor,
    approve_purchase_order,
    create_purchase_order,
    flag_suspicious_invoice,
    reject_purchase_order,
)

N_THREADS = 20


def _run_concurrently(fn, n=N_THREADS):
    results = []
    lock = threading.Lock()

    def worker(i):
        r = fn(i)
        with lock:
            results.append(r)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def test_concurrent_approvals_of_same_po_only_succeed_once(db, make_vendor, make_po):
    vendor_id = make_vendor("Acme Corp")
    make_po("PO-RACE001", vendor_id, amount=1_000.0, status="pending")

    results = _run_concurrently(
        lambda i: approve_purchase_order.invoke(
            {"po_number": "PO-RACE001", "approved_by": f"user{i}@company.com"}
        )
    )

    succeeded = [r for r in results if "has been APPROVED" in r]
    already = [r for r in results if "already approved" in r]
    assert len(succeeded) == 1, f"expected exactly one winner, got: {results}"
    assert len(already) == N_THREADS - 1

    row = db.execute(
        "SELECT status, approved_by FROM purchase_orders WHERE po_number = 'PO-RACE001'"
    ).fetchone()
    assert row["status"] == "approved"
    assert f"has been APPROVED by {row['approved_by']}" in succeeded[0]

    audit_rows = db.execute(
        "SELECT COUNT(*) AS n FROM audit_log WHERE action = 'approved'"
    ).fetchone()["n"]
    assert audit_rows == 1, "concurrent approvals must not write duplicate audit rows"


def test_concurrent_rejections_of_same_po_only_succeed_once(db, make_vendor, make_po):
    vendor_id = make_vendor("Acme Corp")
    make_po("PO-RACE002", vendor_id, amount=1_000.0, status="pending")

    results = _run_concurrently(
        lambda i: reject_purchase_order.invoke(
            {
                "po_number": "PO-RACE002",
                "rejected_by": f"user{i}@company.com",
                "reason": "duplicate submission",
            }
        )
    )

    succeeded = [r for r in results if "has been REJECTED" in r]
    assert len(succeeded) == 1, f"expected exactly one winner, got: {results}"

    audit_rows = db.execute(
        "SELECT COUNT(*) AS n FROM audit_log WHERE action = 'rejected'"
    ).fetchone()["n"]
    assert audit_rows == 1


def test_concurrent_flags_of_same_invoice_only_succeed_once(db, make_vendor, make_po):
    vendor_id = make_vendor("Acme Corp")
    po_id = make_po("PO-RACE003", vendor_id, amount=1_000.0)
    db.execute(
        """
        INSERT INTO invoices (invoice_number, po_id, vendor_id, amount, due_date, status)
        VALUES ('INV-RACE003', ?, ?, 1000.0, '2025-12-01', 'pending')
        """,
        (po_id, vendor_id),
    )
    db.commit()

    results = _run_concurrently(
        lambda i: flag_suspicious_invoice.invoke(
            {
                "invoice_number": "INV-RACE003",
                "flagged_by": f"user{i}@company.com",
                "reason": "amount mismatch",
            }
        )
    )

    succeeded = [r for r in results if "has been flagged as DISPUTED" in r]
    assert len(succeeded) == 1, f"expected exactly one winner, got: {results}"

    audit_rows = db.execute(
        "SELECT COUNT(*) AS n FROM audit_log WHERE action = 'flagged_suspicious'"
    ).fetchone()["n"]
    assert audit_rows == 1


def test_concurrent_registrations_of_same_vendor_only_succeed_once(db):
    """SELECT-then-INSERT on a UNIQUE name had the same race: both callers saw
    'absent' and the loser got a raw IntegrityError instead of a clean message.

    Caveat: unlike the approve/reject/flag cases above, this one rarely trips
    the old code under natural threading — SQLite's write lock serialises the
    fast path, so the second SELECT usually already sees the committed row.
    Forcing the window open with a barrier reproduces it every time (19/20
    threads raise IntegrityError). Treat this as a behavioural regression test
    for the one-winner contract, not as proof the race is gone.
    """
    results = _run_concurrently(
        lambda i: add_vendor.invoke(
            {
                "name": "Contested Supplies Ltd",
                "contact": f"Rep {i}",
                "email": f"rep{i}@contested.com",
            }
        )
    )

    registered = [r for r in results if "registered successfully" in r]
    already = [r for r in results if "already exists" in r]

    assert len(registered) == 1, f"expected exactly one winner, got: {results}"
    assert len(already) == N_THREADS - 1
    assert not [r for r in results if "IntegrityError" in r]

    count = db.execute(
        "SELECT COUNT(*) AS n FROM vendors WHERE name = 'Contested Supplies Ltd'"
    ).fetchone()["n"]
    assert count == 1


def test_concurrent_po_creation_allocates_unique_numbers(db, make_vendor):
    """PO numbers are random against a UNIQUE column; a collision must retry,
    not surface as an IntegrityError."""
    make_vendor("Acme Corp")

    results = _run_concurrently(
        lambda i: create_purchase_order.invoke(
            {
                "vendor_name": "Acme Corp",
                "description": f"Bulk order {i}",
                "amount": 100.0 + i,
                "requested_by": f"user{i}@company.com",
            }
        )
    )

    created = [r for r in results if "Purchase order created" in r]
    assert len(created) == N_THREADS, f"expected all to succeed, got: {results}"

    rows = db.execute("SELECT po_number FROM purchase_orders").fetchall()
    numbers = [r["po_number"] for r in rows]
    assert len(numbers) == N_THREADS
    assert len(set(numbers)) == N_THREADS, "PO numbers must be unique"


def test_po_number_collision_is_retried(db, make_vendor, monkeypatch):
    """Force a collision deterministically rather than hoping for a 1-in-a-million."""
    from agent import tools

    make_vendor("Acme Corp")
    args = {
        "vendor_name": "Acme Corp",
        "description": "Chairs",
        "amount": 100.0,
        "requested_by": "hr@company.com",
    }

    # First two allocations collide, the third is distinct.
    sequence = iter(["111111", "111111", "222222"])
    monkeypatch.setattr(
        tools.random, "choices", lambda *a, **kw: list(next(sequence))
    )

    assert "Purchase order created" in create_purchase_order.invoke(args)
    assert "Purchase order created" in create_purchase_order.invoke(args)

    numbers = {r["po_number"] for r in db.execute("SELECT po_number FROM purchase_orders")}
    assert numbers == {"PO-111111", "PO-222222"}
