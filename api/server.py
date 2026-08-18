"""
api/server.py
─────────────
ProcureGuard AI — FastAPI application (Groq LLM backend).

Startup (lifespan):
  1. Initialises SQLite DB (idempotent)
  2. Auto-seeds if vendors table is empty
  3. Initialises Groq LLM ONCE — stored in app.state
  4. Compiles the LangGraph ONCE — stored in app.state

All routers mounted under /api/v1.

Required env vars:
  GROQ_API_KEY   — get free at https://console.groq.com
  GROQ_MODEL     — default: llama-3.1-8b-instant

Run:
    uvicorn api.server:app --reload --port 8000
"""

import logging
import os
import sqlite3
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.rate_limit import RateLimiter
from api.sessions import SessionStore
from api.routers import chat, vendors, purchase_orders, invoices

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger(__name__)

DB_PATH:        str = os.getenv("DB_PATH",    "data/procurement.db")
MAX_NEW_TOKENS: int = int(os.getenv("MAX_NEW_TOKENS", "512"))

CHAT_RATE_LIMIT:  int = int(os.getenv("CHAT_RATE_LIMIT", "20"))
CHAT_RATE_WINDOW: int = int(os.getenv("CHAT_RATE_WINDOW_SECONDS", "60"))


# ══════════════════════════════════════════════════════════════════════════════
# Lifespan — runs once on startup / shutdown
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────────
    log.info("ProcureGuard API starting up…")

    # 1. Ensure DB schema
    from db.init_db import init_db
    init_db(DB_PATH)

    # 2. Auto-seed if empty
    with sqlite3.connect(DB_PATH) as conn:
        count = conn.execute("SELECT COUNT(*) FROM vendors").fetchone()[0]
    if count == 0:
        log.info("Empty DB — running seeder.")
        from db.seed import seed
        seed(DB_PATH)

    # 3. Init the configured LLM provider (groq | gemini | ollama)
    from agent.llm import build_llm, describe as describe_llm

    model_name = describe_llm()
    try:
        llm = build_llm()
        log.info("LLM ready: %s", model_name)
    except RuntimeError as exc:
        # Let the rest of the API serve; only /chat depends on the model.
        log.warning("LLM unavailable — chat will fail: %s", exc)
        llm = None

    # 4. Build and compile LangGraph (once)
    from main import build_graph
    graph = build_graph(llm) if llm else None
    log.info("LangGraph compiled." if graph else "LangGraph not compiled — no LLM.")

    # 5. Session store
    session_store = SessionStore()

    # 6. Rate limiter for /chat (the only endpoint that calls the LLM)
    chat_rate_limiter = RateLimiter(CHAT_RATE_LIMIT, CHAT_RATE_WINDOW)
    log.info("Chat rate limit: %d requests / %ds per client.", CHAT_RATE_LIMIT, CHAT_RATE_WINDOW)

    # Attach to app.state for router access
    app.state.graph             = graph
    app.state.session_store     = session_store
    app.state.chat_rate_limiter = chat_rate_limiter
    app.state.model_name        = model_name
    app.state.db_path           = DB_PATH

    yield  # ← server runs here

    # ── Shutdown ──────────────────────────────────────────────────────────────
    log.info("ProcureGuard API shutting down. Active sessions: %d",
             session_store.active_count())


# ══════════════════════════════════════════════════════════════════════════════
# App
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="ProcureGuard AI",
    description=(
        "AI-powered procurement agent with compliance guard rules. "
        "Manage vendors, purchase orders, and invoices through a conversational interface."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS (open for dev — tighten for production) ──────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
PREFIX = "/api/v1"
app.include_router(chat.router,           prefix=PREFIX, tags=["Chat"])
app.include_router(vendors.router,        prefix=PREFIX, tags=["Vendors"])
app.include_router(purchase_orders.router,prefix=PREFIX, tags=["Purchase Orders"])
app.include_router(invoices.router,       prefix=PREFIX, tags=["Invoices"])


# ── Static UI files ──────────────────────────────────────────────────────────
app.mount("/ui", StaticFiles(directory="ui"), name="ui")

@app.get("/", include_in_schema=False)
def root():
    return FileResponse("ui/index.html")
