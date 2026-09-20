# Multi-Agent Customer Support System

> A production-grade agentic AI backend demonstrating real-world LangGraph patterns: multi-agent orchestration, FAISS RAG with debugging observability, structured outputs, SQLite-persisted human-in-the-loop approval, and full API audit trails.

**Stack:** Python · FastAPI · LangGraph · LangChain · FAISS · SentenceTransformers · SQLAlchemy · SQLite · Pydantic v2 · Rich

---

## Architecture

```
[POST /api/chat]
       │
  ┌────▼────┐   Structured Output   ┌─────────────────────────┐
  │  Triage │ ─────────────────────▶│  billing_specialist      │
  │  Agent  │   Pydantic Schema     │  tools: order, tickets   │
  └────┬────┘                       └───────────┬─────────────┘
       │                                        │
       │  intent=technical         ┌────────────▼────────────┐
       ├─────────────────────────▶ │  technical_specialist   │
       │                           │  tools: RAG, tickets    │
       │  intent=general           └────────────┬────────────┘
       ├─────────────────────────▶ [general_specialist]       │
       │                                        │
       │  intent=escalation                     │ next_action=escalation
       │        ┌───────────────────────────────┘
       ▼        ▼
  ┌────────────────────┐
  │  [HITL INTERRUPT]  │  ◀── interrupt_before=["escalation"]
  │  SqliteSaver       │       Graph pauses, state persisted to disk
  └────────┬───────────┘
           │  POST /api/approve (supervisor decision)
           ▼
  ┌────────────────────┐
  │  Escalation Agent  │  Creates ticket, notifies supervisor
  └────────────────────┘
```

---

## Production Patterns Implemented

### 1. Structured Output (Triage)
Instead of prompting the LLM to return JSON and parsing it with regex (which silently breaks in production), the Triage Agent uses LangChain's `.with_structured_output(TriageResult)`.

This uses the model's native tool/function-calling API to force schema-valid responses. Invalid values raise a `ValidationError` immediately — no silent failures.

```python
class TriageResult(BaseModel):
    intent: str      # "billing" | "technical" | "general" | "escalation"
    email: str       # extracted from conversation
    confidence: float
    reason: str

structured_llm = llm.with_structured_output(TriageResult)
result = structured_llm.invoke(messages)
```

### 2. Partitioned Specialist Agents (ReAct)
Three domain-scoped ReAct agents replace one general-purpose agent:

| Agent | Tools | Why Separate? |
|---|---|---|
| `billing_specialist` | `get_order_status`, `get_user_tickets`, `send_email` | Avoids tool confusion on payment queries |
| `technical_specialist` | `search_knowledge_base`, `get_user_tickets` | Keeps RAG tool usage focused on tech docs |
| `general_specialist` | `search_knowledge_base`, `send_email` | FAQ-only scope, fewer hallucination risks |

All three agents are **pre-compiled at module load** (not per-request), eliminating repeated graph-compilation overhead.

### 3. FAISS RAG with Retrieval Debugging
The `search_knowledge_base` tool uses FAISS + SentenceTransformers for vector similarity search. Every search returns similarity scores that surface in the API response as `debug_info.retrieval_sources`.

**The RAG Debugging Quadrant** (how to diagnose wrong answers):

```
User gets wrong answer
        │
        ▼
Check debug_info.retrieval_sources[].scores
        │
        ├── Score < 0.3 → RETRIEVAL FAILURE
        │   Fix: better chunking, add docs, rewrite query
        │
        └── Score > 0.5 → GENERATION FAILURE (hallucination)
            Fix: enforce citations in prompt, lower temp, stronger model
```

### 4. Human-in-the-Loop with SQLite Checkpointing
`SqliteSaver` persists conversation state to disk. When an escalation is triggered, `interrupt_before=["escalation"]` pauses the graph. The FastAPI server can restart and the state is still available.

```
graph.invoke(inputs, config)   # Graph pauses before escalation
→ needs_approval = True        # Returned in API response
→ Supervisor calls POST /api/approve
→ graph.invoke(None, config)   # Graph resumes from checkpoint
```

### 5. Full Audit Trail in Every API Response
Every `ChatResponse` includes a `debug_info` object:

```json
{
  "debug_info": {
    "agent_trajectory": ["[Triage] intent='technical'", "[technical_specialist] Completed"],
    "tool_calls": [
      {
        "tool_name": "search_knowledge_base",
        "inputs": {"query": "router disconnecting"},
        "output": "--- Technical Troubleshooting --- ...",
        "timestamp": "2026-07-19T08:00:00Z"
      }
    ],
    "retrieval_sources": [
      {
        "query": "router disconnecting",
        "top_chunks": ["For internet or router issues: ..."],
        "scores": [0.823, 0.641]
      }
    ],
    "active_agent": "technical_specialist"
  }
}
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-username/Customer-Support-System.git
cd Customer-Support-System
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env — add your GEMINI_API_KEY or GROQ_API_KEY
```

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | One of these | Google Gemini (recommended) |
| `GROQ_API_KEY` | One of these | Groq Llama-3 (free tier available) |
| `DATABASE_URL` | No | PostgreSQL URL (defaults to SQLite) |
| `LANGCHAIN_API_KEY` | No | LangSmith tracing |

### 3. Run the API

```bash
uvicorn app.main:app --reload
```

Open Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Run the Demo CLI

```bash
python run_demo.py
```

### 5. Run Tests

```bash
pytest tests/ -v
```

---

## API Reference

### `POST /api/chat`
Start or continue a conversation.

```json
{
  "thread_id": "uuid-per-session",
  "message": "I need a refund for order ORD12345",
  "user_email": "alice@example.com"
}
```

Response includes `needs_approval: true` when a HITL escalation is triggered.

### `POST /api/approve`
Supervisor approval for escalation.

```json
{
  "thread_id": "uuid-per-session",
  "approve": true,
  "feedback": ""
}
```

### `GET /api/history/{email}`
Fetch ticket and order history (demonstrates long-term memory).

---

## Project Structure

```
Customer-Support-System/
├── app/
│   ├── agents/
│   │   ├── graph.py         # LangGraph workflow (SqliteSaver + routing)
│   │   ├── state.py         # AgentState + debug field types
│   │   ├── triage.py        # Structured output intent classifier
│   │   ├── specialist.py    # Three domain-scoped ReAct agents
│   │   ├── escalation.py    # HITL escalation + ticket creation
│   │   ├── llm.py           # LLM factory (Gemini / Groq / mock)
│   │   └── tools/
│   │       ├── rag.py       # FAISS vector search with scores
│   │       ├── order.py     # Order status DB lookup
│   │       ├── ticket.py    # Ticket CRUD + history
│   │       └── email.py     # Mock email sender
│   ├── db/
│   │   ├── connection.py    # SQLAlchemy engine + session
│   │   └── models.py        # Ticket + Order ORM models
│   ├── config.py            # Settings from .env
│   ├── schemas.py           # Pydantic request/response schemas
│   └── main.py              # FastAPI app + endpoints
├── tests/
│   └── test_agents.py       # Integration tests
├── run_demo.py              # Rich CLI demo
├── requirements.txt
├── .env.example
└── README.md
```

---

## Interview Talking Points

**"Tell me about the architecture."**
> "It's a LangGraph StateGraph with four nodes: a Triage classifier, three domain-scoped Specialist agents, and an Escalation node. Routing is deterministic — the LLM only makes the classification decision in Triage, and a Python router function reads that and picks the next node. This makes the graph predictable and testable."

**"How did you handle the human-in-the-loop?"**
> "Using LangGraph's `interrupt_before` with a SQLite-backed checkpointer. When the graph hits the escalation node, it pauses and serializes state to disk. The FastAPI server can restart and the state is still there. A supervisor calls POST /api/approve which resumes execution from the exact checkpoint."

**"My RAG system gives wrong answers sometimes. How do you debug that?"**
> "First, I check the similarity scores in the debug_info.retrieval_sources field. Score below 0.3 means retrieval failed — the right document wasn't found. Score above 0.5 but wrong answer means the LLM received the right context but hallucinated anyway — that's a generation bug. You fix them completely differently: retrieval bugs need better chunking or query rewriting; generation bugs need prompt changes like enforcing citations or using a larger model."

**"Why three separate agents instead of one?"**
> "A single agent with 10 tools gets confused — the LLM sometimes calls billing tools for a technical question. With scoped agents, you also get much better debuggability: when something goes wrong, you know exactly which agent and which tool was responsible without digging through full traces."
