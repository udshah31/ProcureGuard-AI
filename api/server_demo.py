"""
api/server_demo.py
──────────────────
ProcureGuard AI — DEMO MODE server.

Identical to server.py but replaces the HuggingFace LLM + LangGraph
with a lightweight mock that:
  - Calls the real SQLite tools directly
  - Returns realistic canned responses based on keyword matching
  - Enforces real guard rules
  - Requires ZERO heavy dependencies (no torch, no transformers, no langchain)

Run:
    uvicorn api.server_demo:app --reload --port 8000
    open http://localhost:8000
"""

import json
import logging
import os
import re
import sqlite3
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from agent.guard_rules import format_guard_summary, run_all_guards
from api.auth import AuthContext, require_api_key
from api.middleware import RequestContextMiddleware
from api.rate_limit import RateLimiter
from api.sessions import SessionStore
from api.models import ChatRequest, ChatResponse, HealthResponse
from api.routers import vendors, purchase_orders, invoices, observability as observability_router
from observability import configure_logging, metrics

load_dotenv()
configure_logging()
log = logging.getLogger(__name__)

DB_PATH: str = os.getenv("DB_PATH", "data/procurement.db")

CHAT_RATE_LIMIT:  int = int(os.getenv("CHAT_RATE_LIMIT", "20"))
CHAT_RATE_WINDOW: int = int(os.getenv("CHAT_RATE_WINDOW_SECONDS", "60"))


# ══════════════════════════════════════════════════════════════════════════════
# Mock Agent — real tools, smart keyword routing, no LLM needed
# ══════════════════════════════════════════════════════════════════════════════

def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def _write_audit(conn, entity_type, entity_id, action, actor, details):
    conn.execute(
        "INSERT INTO audit_log (entity_type, entity_id, action, actor, details) VALUES (?,?,?,?,?)",
        (entity_type, entity_id, action, actor, json.dumps(details)),
    )


def mock_agent(message: str, role: str, identity: str) -> tuple[str, list[str], list[str]]:
    """
    Returns (reply, tool_calls_made, risk_flags).
    Routes to real DB operations based on message keywords.
    """
    msg   = message.lower()
    tools = []
    flags = []

    # ── lookup vendor ───────────────────────────────────────────────────────
    if any(k in msg for k in ["look up", "lookup", "find vendor", "search vendor", "vendor info"]):
        # Extract name: words after 'vendor', 'up', 'find', 'for'
        m = re.search(r'(?:look\s+up|lookup|find|search)\s+(?:vendor\s+)?([\w\s&]+?)(?:\s*$|,|\.)', msg)
        name = m.group(1).strip() if m else ""
        # Remove trailing common words
        name = re.sub(r'\b(vendor|up|find|for)\b', '', name).strip()
        if not name:
            words = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', message)
            name = " ".join(words) if words else "Acme"
        tools.append("lookup_vendor")
        with _db() as conn:
            row = conn.execute(
                "SELECT id,name,contact,email,phone,status FROM vendors WHERE name LIKE ?",
                (f"%{name}%",),
            ).fetchone()
        if row:
            status_warn = " ⚠️ This vendor is **blacklisted**." if row["status"] == "blacklisted" else ""
            return (
                f"**Vendor found** — ID: {row['id']}\n"
                f"• Name: {row['name']}\n"
                f"• Contact: {row['contact'] or '—'}\n"
                f"• Email: {row['email'] or '—'}\n"
                f"• Phone: {row['phone'] or '—'}\n"
                f"• Status: **{row['status'].upper()}**{status_warn}",
                tools, flags,
            )
        return f"No vendor found matching '{name}'.", tools, flags

    # ── risk score ──────────────────────────────────────────────────────────
    if any(k in msg for k in ["risk score", "risk", "score for"]):
        # Extract vendor name: words after 'for' or 'score' keyword
        m = re.search(r'(?:for|score\s+for|risk\s+score\s+for)\s+([\w\s&]+?)(?:\s*$|,|\.)', msg)
        name = m.group(1).strip() if m else ""
        if not name:
            words = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', message)
            name = " ".join(words) if words else "TechSupply"
        tools.append("vendor_risk_score")
        with _db() as conn:
            v = conn.execute("SELECT id,name,status FROM vendors WHERE name LIKE ?", (f"%{name}%",)).fetchone()
        if not v:
            return f"Vendor '{name}' not found.", tools, flags
        score = 0; breakdown = []
        if v["status"] == "blacklisted": score += 60; breakdown.append("Blacklisted vendor (+60)")
        elif v["status"] == "inactive":  score += 20; breakdown.append("Inactive vendor (+20)")
        with _db() as conn:
            ps = conn.execute(
                "SELECT COUNT(*) t, SUM(CASE WHEN status='rejected' THEN 1 ELSE 0 END) r, AVG(amount) a FROM purchase_orders WHERE vendor_id=?", (v["id"],)
            ).fetchone()
            ivs = conn.execute(
                "SELECT COUNT(*) t, SUM(CASE WHEN status='disputed' THEN 1 ELSE 0 END) d FROM invoices WHERE vendor_id=?", (v["id"],)
            ).fetchone()
        if ps["t"]:
            rr = (ps["r"] or 0) / ps["t"]; pts = int(rr*25); score+=pts
            if pts: breakdown.append(f"PO rejection rate {rr:.0%} (+{pts})")
        if ivs["t"]:
            dr = (ivs["d"] or 0) / ivs["t"]; pts = int(dr*30); score+=pts
            if pts: breakdown.append(f"Invoice dispute rate {dr:.0%} (+{pts})")
        if (ps["a"] or 0) > 100000: score+=10; breakdown.append("Avg PO >$100k (+10)")
        elif (ps["a"] or 0) > 50000: score+=5; breakdown.append("Avg PO >$50k (+5)")
        score = min(score, 100)
        tier = "🔴 CRITICAL" if score>=75 else "🟠 HIGH" if score>=50 else "🟡 MEDIUM" if score>=25 else "🟢 LOW"
        return (
            f"**Vendor Risk Score — {v['name']}**\n"
            f"• Score: **{score}/100** | Tier: {tier}\n"
            f"• {' | '.join(breakdown) or 'No risk factors detected'}\n"
            f"• POs: {ps['t']} total, {ps['r'] or 0} rejected | Invoices: {ivs['t']} total, {ivs['d'] or 0} disputed",
            tools, flags,
        )

    # ── approve PO ──────────────────────────────────────────────────────────
    if "approve" in msg and ("po-" in msg or "purchase order" in msg):
        po_match = re.search(r"po-\d+", msg)
        po_num = po_match.group(0).upper() if po_match else None
        # approved_by is always the authenticated caller's identity — never
        # taken from the message text, or a caller could name anyone as the
        # approver and walk past the self-approval guard.
        approved_by = identity
        tools.append("approve_purchase_order")

        if not po_num:
            return "Please specify a PO number (e.g. PO-100002).", tools, flags

        with _db() as conn:
            po = conn.execute(
                "SELECT id,status,amount,vendor_id FROM purchase_orders WHERE po_number=?", (po_num,)
            ).fetchone()

        if not po:
            return f"PO '{po_num}' not found.", tools, flags

        # Same guard layer the real agent uses (agent.guard_rules is pure
        # stdlib, so demo mode still needs no langchain). Previously inlined,
        # which let demo mode drift behind the real compliance rules.
        results = run_all_guards(po_num, approved_by=approved_by)
        failures = [r for r in results if not r.passed]
        flags = [r.message for r in failures]
        blocked = any(r.severity == "block" for r in failures)

        if blocked:
            return (
                f"🚫 **Approval BLOCKED** for {po_num}\n\n"
                + format_guard_summary(results),
                tools, flags,
            )

        if po["status"] not in ("draft", "pending"):
            return f"PO '{po_num}' cannot be approved — current status: **{po['status'].upper()}**.", tools, flags

        with _db() as conn:
            conn.execute(
                "UPDATE purchase_orders SET status='approved', approved_by=?, updated_at=datetime('now') WHERE po_number=?",
                (approved_by, po_num),
            )
            _write_audit(conn, "purchase_order", po["id"], "approved", approved_by,
                         {"po_number": po_num, "amount": po["amount"]})
            conn.commit()

        warn_text = ("\n\n**Compliance Warnings:**\n" + "\n".join(f"• {f}" for f in flags)) if flags else ""
        return (
            f"✅ **PO {po_num} approved** by {approved_by}.{warn_text}",
            tools, flags,
        )


    # ── reject PO ───────────────────────────────────────────────────────────
    if "reject" in msg and ("po-" in msg or "purchase order" in msg):
        po_match = re.search(r"po-\d+", msg)
        po_num   = po_match.group(0).upper() if po_match else None
        tools.append("reject_purchase_order")
        if not po_num:
            return "Please specify a PO number (e.g. PO-100003).", tools, flags
        with _db() as conn:
            po = conn.execute("SELECT id,status FROM purchase_orders WHERE po_number=?", (po_num,)).fetchone()
        if not po:
            return f"PO '{po_num}' not found.", tools, flags
        if po["status"] in ("rejected","closed"):
            return f"PO '{po_num}' is already {po['status']}.", tools, flags
        reason_m = re.search(r"reason[:\s]+(.+)$", msg)
        reason   = reason_m.group(1).strip() if reason_m else "Rejected via ProcureGuard AI"
        with _db() as conn:
            conn.execute(
                "UPDATE purchase_orders SET status='rejected', updated_at=datetime('now') WHERE po_number=?",
                (po_num,),
            )
            _write_audit(conn, "purchase_order", po["id"], "rejected", role, {"reason": reason})
            conn.commit()
        return f"❌ **PO {po_num} rejected**. Reason: {reason}", tools, flags

    # ── add vendor ──────────────────────────────────────────────────────────
    if any(k in msg for k in ["add vendor", "register vendor", "new vendor", "create vendor"]):
        tools.append("add_vendor")
        words  = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", message)
        name   = " ".join(words[:3]) if words else "New Vendor Inc"
        email_m = re.search(r"[\w.]+@[\w.]+", message)
        email   = email_m.group(0) if email_m else "contact@newvendor.com"
        with _db() as conn:
            ex = conn.execute("SELECT id FROM vendors WHERE name=?", (name,)).fetchone()
            if ex:
                return f"Vendor '{name}' already exists (ID: {ex['id']}).", tools, flags
            cur = conn.execute(
                "INSERT INTO vendors (name,contact,email,status) VALUES (?,?,?,'active')",
                (name, role+"@company.com", email),
            )
            conn.commit()
        return f"✅ **Vendor '{name}' registered** successfully. ID: {cur.lastrowid}", tools, flags

    # ── create PO ───────────────────────────────────────────────────────────
    if any(k in msg for k in ["create po", "create purchase", "new po", "new purchase order", "raise po"]):
        tools.append("create_purchase_order")
        import random, string
        po_num = "PO-" + "".join(random.choices(string.digits, k=6))
        amt_m  = re.search(r"\$?([\d,]+(?:\.\d+)?)", message)
        amount = float(amt_m.group(1).replace(",","")) if amt_m else 5000.0
        words  = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", message)
        vname  = " ".join(words[:2]) if words else "Acme Corp"
        with _db() as conn:
            v = conn.execute("SELECT id FROM vendors WHERE name LIKE ?", (f"%{vname}%",)).fetchone()
            if not v:
                return f"Vendor '{vname}' not found. Register them first.", tools, flags
            conn.execute(
                "INSERT INTO purchase_orders (po_number,vendor_id,description,amount,requested_by,status) VALUES (?,?,?,?,?,'draft')",
                (po_num, v["id"], message[:80], amount, role+"@company.com"),
            )
            conn.commit()
        flags_out = []
        if amount > 50000:
            flags_out.append(f"Amount ${amount:,.0f} exceeds $50k threshold — finance sign-off required")
        return (
            f"✅ **Purchase Order created**\n• PO Number: **{po_num}**\n• Amount: ${amount:,.2f}\n• Status: DRAFT",
            tools, flags_out,
        )

    # ── check invoice ───────────────────────────────────────────────────────
    if any(k in msg for k in ["invoice", "inv-"]):
        inv_m = re.search(r"inv-[\d-]+", msg)
        inv_n = inv_m.group(0).upper() if inv_m else None
        tools.append("check_invoice_status")
        if not inv_n:
            # list all invoices
            with _db() as conn:
                rows = conn.execute(
                    "SELECT i.invoice_number,i.status,i.amount,v.name FROM invoices i JOIN vendors v ON i.vendor_id=v.id"
                ).fetchall()
            lines = [f"• {r['invoice_number']} — {r['name']} — ${r['amount']:,.2f} [{r['status'].upper()}]" for r in rows]
            return "**All invoices:**\n" + "\n".join(lines), tools, flags
        with _db() as conn:
            row = conn.execute(
                "SELECT i.*,v.name vname,po.po_number FROM invoices i JOIN vendors v ON i.vendor_id=v.id JOIN purchase_orders po ON i.po_id=po.id WHERE i.invoice_number=?",
                (inv_n,),
            ).fetchone()
        if not row:
            return f"Invoice '{inv_n}' not found.", tools, flags
        paid = f" | Paid: {row['paid_date']}" if row["paid_date"] else ""
        return (
            f"**Invoice {row['invoice_number']}** [{row['status'].upper()}]\n"
            f"• Vendor: {row['vname']}\n• PO: {row['po_number']}\n"
            f"• Amount: {row['currency']} {row['amount']:,.2f}\n"
            f"• Due: {row['due_date'] or '—'}{paid}",
            tools, flags,
        )

    # ── flag invoice ────────────────────────────────────────────────────────
    if "flag" in msg and ("invoice" in msg or "inv-" in msg):
        inv_m  = re.search(r"inv-[\d-]+", msg)
        inv_n  = inv_m.group(0).upper() if inv_m else None
        tools.append("flag_suspicious_invoice")
        if not inv_n:
            return "Please specify an invoice number (e.g. INV-2025-003).", tools, flags
        reason_m = re.search(r"(?:reason[:\s]+|,\s*)(.+)$", msg)
        reason   = reason_m.group(1).strip() if reason_m else "Flagged as suspicious"
        with _db() as conn:
            inv = conn.execute("SELECT id,status,amount FROM invoices WHERE invoice_number=?", (inv_n,)).fetchone()
        if not inv:
            return f"Invoice '{inv_n}' not found.", tools, flags
        if inv["status"] == "disputed":
            return f"Invoice '{inv_n}' is already marked as disputed.", tools, flags
        with _db() as conn:
            conn.execute(
                "UPDATE invoices SET status='disputed', updated_at=datetime('now') WHERE invoice_number=?",
                (inv_n,),
            )
            _write_audit(conn, "invoice", inv["id"], "flagged_suspicious", role,
                         {"invoice_number": inv_n, "reason": reason})
            conn.commit()
        return (
            f"🚨 **Invoice {inv_n} flagged as DISPUTED**\n• Flagged by: {role}\n• Reason: {reason}",
            tools, flags,
        )

    # ── list vendors ─────────────────────────────────────────────────────────
    if "vendor" in msg and any(k in msg for k in ["list", "show", "all", "active", "blacklisted"]):
        tools.append("lookup_vendor")
        status_f = "blacklisted" if "blacklisted" in msg else ("inactive" if "inactive" in msg else None)
        with _db() as conn:
            if status_f:
                rows = conn.execute("SELECT name,status FROM vendors WHERE status=?", (status_f,)).fetchall()
            else:
                rows = conn.execute("SELECT name,status FROM vendors ORDER BY name").fetchall()
        lines = [f"• {r['name']} [{r['status'].upper()}]" for r in rows]
        return f"**Vendors ({len(rows)}):**\n" + "\n".join(lines), tools, flags

    # ── help / greeting ──────────────────────────────────────────────────────
    if any(k in msg for k in ["hello", "hi", "help", "what can", "capabilities"]):
        return (
            "👋 **Welcome to ProcureGuard AI!**\n\n"
            "I can help you:\n"
            "• 🏢 **Vendor management** — look up, register, risk-score vendors\n"
            "• 📋 **Purchase orders** — create, approve (with guard rules), reject\n"
            "• 🧾 **Invoices** — check status, flag suspicious invoices\n\n"
            "Try asking:\n"
            "• *\"Look up vendor Acme Corp\"*\n"
            "• *\"Approve PO-100002\"* (will trigger guard — amount > $50k)\n"
            "• *\"Risk score for ShadyDeals LLC\"* (blacklisted!)\n"
            "• *\"Flag invoice INV-2025-003 as suspicious\"*",
            [], [],
        )

    # ── fallback ─────────────────────────────────────────────────────────────
    return (
        "I'm your procurement assistant. I can help with:\n"
        "• Looking up vendors and checking their risk score\n"
        "• Creating, approving, or rejecting purchase orders\n"
        "• Checking invoice status or flagging suspicious invoices\n\n"
        "What would you like to do?",
        [], [],
    )


# ══════════════════════════════════════════════════════════════════════════════
# Lifespan
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("ProcureGuard DEMO API starting up…")
    from db.init_db import init_db
    init_db(DB_PATH)
    with sqlite3.connect(DB_PATH) as conn:
        count = conn.execute("SELECT COUNT(*) FROM vendors").fetchone()[0]
    if count == 0:
        from db.seed import seed
        seed(DB_PATH)
        log.info("DB seeded.")

    app.state.session_store     = SessionStore()
    app.state.chat_rate_limiter = RateLimiter(CHAT_RATE_LIMIT, CHAT_RATE_WINDOW)
    app.state.model_name        = "demo-mode (no LLM)"
    app.state.db_path           = DB_PATH
    log.info("Demo mode ready — no LLM loaded.")
    yield
    log.info("Shutting down.")


# ══════════════════════════════════════════════════════════════════════════════
# App
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="ProcureGuard AI (Demo)",
    description="Demo mode — real DB + tools, mock LLM.",
    version="0.1.0-demo",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
app.add_middleware(RequestContextMiddleware)

PREFIX = "/api/v1"
app.include_router(vendors.router,           prefix=PREFIX, tags=["Vendors"],
                    dependencies=[Depends(require_api_key)])
app.include_router(purchase_orders.router,   prefix=PREFIX, tags=["Purchase Orders"],
                    dependencies=[Depends(require_api_key)])
app.include_router(invoices.router,          prefix=PREFIX, tags=["Invoices"],
                    dependencies=[Depends(require_api_key)])
app.include_router(observability_router.router, prefix=PREFIX, tags=["Observability"],
                    dependencies=[Depends(require_api_key)])


# ── Chat endpoint (inline — no langchain needed) ──────────────────────────────
from fastapi import APIRouter
_chat = APIRouter()

@_chat.post("/chat", response_model=ChatResponse)
def chat(request: Request, body: ChatRequest, auth: AuthContext = Depends(require_api_key)) -> ChatResponse:
    client_key = request.client.host if request.client else "unknown"
    allowed, retry_after = request.app.state.chat_rate_limiter.check(client_key)
    if not allowed:
        metrics.increment("chat_rate_limited_total")
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please slow down and try again shortly.",
            headers={"Retry-After": str(int(retry_after) + 1)},
        )

    metrics.increment("chat_requests_total")
    store = request.app.state.session_store
    store.get_or_create(body.session_id, role=auth.role, identity=auth.identity)
    reply, tool_calls, risk_flags = mock_agent(body.message, auth.role, auth.identity)
    store.update(body.session_id, {"messages": [], "role": auth.role, "identity": auth.identity,
                                    "context": {}, "risk_flags": risk_flags})
    log.info("Chat [%s] tools=%s → %d chars", body.session_id, tool_calls, len(reply))
    return ChatResponse(session_id=body.session_id, reply=reply,
                        risk_flags=risk_flags, tool_calls_made=tool_calls)

@_chat.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    return HealthResponse(
        status="ok", model_name=request.app.state.model_name,
        db_path=request.app.state.db_path,
        active_sessions=request.app.state.session_store.active_count(),
    )

app.include_router(_chat, prefix=PREFIX, tags=["Chat"])

# ── Static UI ─────────────────────────────────────────────────────────────────
app.mount("/ui", StaticFiles(directory="ui"), name="ui")

@app.get("/", include_in_schema=False)
def root():
    return FileResponse("ui/index.html")
