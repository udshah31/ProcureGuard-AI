<div align="center">

# 🛡️ ProcureGuard AI

**An AI-powered procurement management agent with real-time compliance guard rules**

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent-purple)](https://langchain-ai.github.io/langgraph/)
[![Groq](https://img.shields.io/badge/LLM-Groq%20Llama%203.1-orange)](https://groq.com)
[![SQLite](https://img.shields.io/badge/DB-SQLite-lightblue?logo=sqlite)](https://sqlite.org)

</div>

---

## 🎥 What it does

ProcureGuard AI is a full-stack procurement agent that lets you manage vendors, purchase orders, and invoices through a **conversational chat interface**. It uses a real AI language model (Llama 3.1 via Groq) to understand natural language and enforce compliance rules automatically.

```
User: "Approve PO-100002 for cfo@company.com"
Agent: ✅ PO approved
       ⚠️ Warning: Amount $87,500 exceeds $50k threshold — finance sign-off required.
       ⚙️ Tools used: approve_purchase_order
```

---

## 🏗️ Architecture

```
Browser Dashboard (ui/index.html)
    └── FastAPI REST API (api/)
         ├── POST /api/v1/chat          ← LangGraph agent loop
         ├── GET  /api/v1/vendors       ← SQLite data endpoints
         ├── GET  /api/v1/purchase-orders
         └── GET  /api/v1/invoices

LangGraph Agent
    ├── LLM: Groq Llama 3.1 (API, no local downloads)
    ├── Tools: 8 domain tools (lookup_vendor, approve_po, flag_invoice, …)
    └── Guard Node: compliance checks before any write operation
         ├── Blacklist check
         ├── Approval threshold ($50k)
         └── Duplicate PO detection
```

---

## ✨ Features

| Feature | Detail |
|---|---|
| 💬 **Conversational UI** | Premium dark-mode 3-column chat dashboard |
| 🤖 **Real LLM** | Groq Llama 3.1 — fast, free, no local downloads |
| 🔗 **LangGraph agent** | Multi-step tool use with memory across turns |
| 🛡️ **Guard rules** | Blacklist, threshold, duplicate PO checks before approval |
| 📊 **Live stats sidebar** | Vendor/PO/Invoice counts with auto-refresh every 15s |
| 🏢 **Vendor management** | Risk scoring, status tracking (active/inactive/blacklisted) |
| 📋 **Purchase orders** | Full lifecycle: draft → pending → approved/rejected |
| 🧾 **Invoice tracking** | Dispute flagging with audit trail |
| 🔐 **Role-based** | requester / approver / finance / admin roles |
| 💾 **SQLite** | Zero-config, fully seeded with realistic test data |

---

## 🚀 Quick Start

### 1. Clone & set up

```bash
git clone https://github.com/udaysah/ProcureGuard-AI.git
cd ProcureGuard-AI

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get a free Groq API key

1. Go to [console.groq.com](https://console.groq.com) — sign up free
2. Click **API Keys → Create API Key**
3. Copy your key

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and paste your Groq key:
# GROQ_API_KEY=gsk_...
```

### 4. Run

```bash
uvicorn api.server:app --reload --port 8000
open http://localhost:8000
```

> **No GPU needed. No large downloads.** Groq runs the model in the cloud.

---

## 🧪 Demo Mode (no API key needed)

Want to try it without any API key? Run the demo server:

```bash
uvicorn api.server_demo:app --reload --port 8000
open http://localhost:8000
```

Demo mode uses real SQLite tools with keyword routing instead of an LLM. All vendor/PO/invoice operations work — the agent just doesn't understand free-form language.

---

## 📁 Project Structure

```
procurement_agent/
├── agent/
│   ├── state.py          # AgentState — messages, role, context, risk_flags
│   ├── tools.py          # 8 domain tools (lookup, approve, flag, …)
│   └── guard_rules.py    # Compliance checks (blacklist, threshold, duplicate)
├── api/
│   ├── server.py         # FastAPI app + Groq LLM lifespan
│   ├── server_demo.py    # Zero-dep demo server
│   ├── sessions.py       # Thread-safe session store (30-min TTL)
│   ├── models.py         # Pydantic schemas
│   └── routers/
│       ├── chat.py       # POST /chat, GET /health
│       ├── vendors.py    # GET /vendors
│       ├── purchase_orders.py
│       └── invoices.py
├── db/
│   ├── init_db.py        # Schema (vendors, purchase_orders, invoices, audit_log)
│   └── seed.py           # Realistic test data
├── ui/
│   └── index.html        # Full dashboard (vanilla HTML/CSS/JS)
├── main.py               # LangGraph graph builder + CLI
├── .env.example
└── requirements.txt
```

---

## 🌐 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/chat` | Send a message to the agent |
| `GET` | `/api/v1/health` | Server health + model info |
| `GET` | `/api/v1/vendors` | List vendors (`?status=blacklisted`) |
| `GET` | `/api/v1/vendors/{id}` | Single vendor |
| `GET` | `/api/v1/purchase-orders` | List POs (`?status=pending&min_amount=50000`) |
| `GET` | `/api/v1/purchase-orders/{po_number}` | Single PO |
| `GET` | `/api/v1/invoices` | List invoices (`?status=disputed&overdue_only=true`) |
| `GET` | `/api/v1/invoices/{invoice_number}` | Single invoice |
| `GET` | `/docs` | Swagger UI |

---

## 🔧 Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | **Required** — get free at console.groq.com |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model to use |
| `DB_PATH` | `data/procurement.db` | SQLite database path |
| `LOG_LEVEL` | `INFO` | Logging level |
| `MAX_NEW_TOKENS` | `512` | Max tokens per agent response |

---

## 🛡️ Guard Rules

Every write operation (approve PO, etc.) passes through the compliance guard:

| Rule | Trigger | Severity |
|---|---|---|
| Blacklist check | Vendor is blacklisted | 🚫 **Block** |
| Approval threshold | PO amount > $50,000 | ⚠️ Warn |
| Duplicate detection | Similar PO from same vendor in last 30 days | ⚠️ Warn |

---

## 📄 License

MIT — free to use and modify.
