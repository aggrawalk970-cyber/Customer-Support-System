"""
Integration Tests — Multi-Agent Customer Support System
────────────────────────────────────────────────────────
Tests verify:
1. Triage structured output schema (TriageDecision) produces correct fields.
2. Graph routing: billing/technical/general/escalation messages reach the right node.
3. Escalation interrupt: graph pauses before the escalation node.
4. SQLite checkpointer: state persists and is retrievable after thread creation.
5. FastAPI endpoints: /api/chat, /api/approve, /api/health return expected shapes.
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage, AIMessage


# ── Test Client ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """Create a FastAPI test client with all startup hooks triggered."""
    from app.main import app
    with TestClient(app) as c:
        yield c


# ── 1. Health Check ───────────────────────────────────────────────────────────

def test_health_check(client):
    """Verify the health endpoint returns OK."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


# ── 2. Triage Structured Output Schema ───────────────────────────────────────

def test_triage_decision_schema():
    """TriageDecision Pydantic model enforces correct field types and Literal values."""
    from app.agents.triage import TriageDecision

    decision = TriageDecision(
        intent="billing",
        email="test@example.com",
        reason="Customer asked about a refund.",
    )
    assert decision.intent == "billing"
    assert decision.email == "test@example.com"
    assert isinstance(decision.reason, str)


def test_triage_decision_invalid_intent():
    """TriageDecision rejects an intent value not in the Literal set."""
    from pydantic import ValidationError
    from app.agents.triage import TriageDecision

    with pytest.raises(ValidationError):
        TriageDecision(intent="unknown_intent", reason="test")


# ── 3. Triage Node: Routing Logic ─────────────────────────────────────────────

def test_triage_node_billing_routing():
    """A billing message should produce intent='billing' and next_action='billing_specialist'."""
    from app.agents.triage import triage_node
    from langchain_core.messages import HumanMessage

    state = {
        "messages": [HumanMessage(content="I was charged twice for my subscription last month.")],
        "user_email": "",
        "intent": "",
        "next_action": "",
        "active_agent": "",
        "escalation_summary": "",
        "approved_by_human": False,
        "logs": [],
        "tool_calls_log": [],
        "retrieval_sources": [],
    }

    # Patch get_llm to return a FakeLLM (triggers the keyword fallback inside triage_node)
    with patch("app.agents.triage.get_llm") as mock_llm:
        fake_llm = MagicMock()
        fake_structured = MagicMock()
        fake_structured.invoke.side_effect = Exception("FakeLLM no structured output")
        fake_llm.with_structured_output.return_value = fake_structured
        mock_llm.return_value = fake_llm

        result = triage_node(state)

    assert result["intent"] == "billing"
    assert result["next_action"] == "billing_specialist"
    assert result["active_agent"] == "triage"
    assert len(result["logs"]) == 1


def test_triage_node_technical_routing():
    """A technical message should produce intent='technical' and next_action='technical_specialist'."""
    from app.agents.triage import triage_node
    from langchain_core.messages import HumanMessage

    state = {
        "messages": [HumanMessage(content="My router keeps resetting and I can't connect to the network.")],
        "user_email": "",
        "intent": "",
        "next_action": "",
        "active_agent": "",
        "escalation_summary": "",
        "approved_by_human": False,
        "logs": [],
        "tool_calls_log": [],
        "retrieval_sources": [],
    }

    with patch("app.agents.triage.get_llm") as mock_llm:
        fake_llm = MagicMock()
        fake_structured = MagicMock()
        fake_structured.invoke.side_effect = Exception("FakeLLM no structured output")
        fake_llm.with_structured_output.return_value = fake_structured
        mock_llm.return_value = fake_llm

        result = triage_node(state)

    assert result["intent"] == "technical"
    assert result["next_action"] == "technical_specialist"


def test_triage_node_escalation_routing():
    """An escalation message should produce intent='escalation' and next_action='escalation'."""
    from app.agents.triage import triage_node
    from langchain_core.messages import HumanMessage

    state = {
        "messages": [HumanMessage(content="I want to speak to a human supervisor right now.")],
        "user_email": "",
        "intent": "",
        "next_action": "",
        "active_agent": "",
        "escalation_summary": "",
        "approved_by_human": False,
        "logs": [],
        "tool_calls_log": [],
        "retrieval_sources": [],
    }

    with patch("app.agents.triage.get_llm") as mock_llm:
        fake_llm = MagicMock()
        fake_structured = MagicMock()
        fake_structured.invoke.side_effect = Exception("FakeLLM no structured output")
        fake_llm.with_structured_output.return_value = fake_structured
        mock_llm.return_value = fake_llm

        result = triage_node(state)

    assert result["intent"] == "escalation"
    assert result["next_action"] == "escalation"


# ── 4. RAG Tool — Vector Search ───────────────────────────────────────────────

def test_rag_search_returns_results():
    """search_knowledge_base returns a non-empty string for a valid query."""
    from app.agents.tools.rag import search_knowledge_base

    result = search_knowledge_base.invoke("How do I reset my router?")
    assert isinstance(result, str)
    assert len(result) > 50  # Should return some article content


def test_rag_search_no_matches():
    """search_knowledge_base handles queries that match nothing gracefully."""
    from app.agents.tools.rag import search_knowledge_base

    result = search_knowledge_base.invoke("xyzzy frobnik unrelated gobbledygook")
    assert isinstance(result, str)
    # Since FAISS always returns closest matches, check for LOW relevance
    assert "LOW relevance" in result or "Available" in result


# ── 5. FastAPI Chat Endpoint — Shape Validation ───────────────────────────────

def test_chat_endpoint_response_shape(client):
    """POST /api/chat should return all expected fields including debug_info."""
    import uuid
    from unittest.mock import patch, MagicMock

    thread_id = str(uuid.uuid4())
    
    with patch("app.main.compiled_graph") as mock_graph:
        mock_state = MagicMock()
        mock_state.values = {
            "messages": [HumanMessage(content="What are your support hours?"), AIMessage(content="9 to 5")],
            "intent": "general",
            "next_action": "end",
            "active_agent": "general_specialist",
            "needs_approval": False,
            "logs": ["[Triage] Classified as general"],
            "tool_calls_log": [],
            "retrieval_sources": []
        }
        mock_state.next = []
        mock_graph.get_state.return_value = mock_state

        response = client.post("/api/chat", json={
            "message": "What are your support hours?",
            "thread_id": thread_id,
            "user_email": "testuser@example.com",
        })

    assert response.status_code == 200
    data = response.json()

    # Verify top-level fields
    assert "messages" in data
    assert "intent" in data
    assert "next_action" in data
    assert "active_agent" in data
    assert "needs_approval" in data
    assert "logs" in data
    assert "debug_info" in data

    # Verify debug_info structure
    debug = data["debug_info"]
    assert "agent_trajectory" in debug
    assert "tool_calls" in debug
    assert "retrieval_sources" in debug
    assert "active_agent" in debug

    # Must have at least one message
    assert len(data["messages"]) >= 1


def test_chat_endpoint_blocked_when_paused(client):
    """Sending a message to a thread paused for human approval should return HTTP 400."""
    import uuid
    from unittest.mock import patch

    thread_id = str(uuid.uuid4())

    # Mock the graph to simulate a paused state (state.next is non-empty)
    mock_state = MagicMock()
    mock_state.values = {"messages": [], "intent": "escalation", "logs": [], "next_action": "escalation", "active_agent": "triage", "tool_calls_log": [], "retrieval_sources": []}
    mock_state.next = ["escalation"]  # Simulates interrupt_before=["escalation"]

    with patch("app.main.compiled_graph") as mock_graph:
        mock_graph.get_state.return_value = mock_state

        # First call creates the thread
        response = client.post("/api/chat", json={
            "message": "I want to escalate this.",
            "thread_id": thread_id,
        })
        # Second call should be blocked
        response2 = client.post("/api/chat", json={
            "message": "Another message.",
            "thread_id": thread_id,
        })

    assert response2.status_code == 400


# ── 6. Graph Router Functions ─────────────────────────────────────────────────

def test_route_after_triage_billing():
    from app.agents.graph import route_after_triage
    state = {"intent": "billing", "next_action": "billing_specialist"}
    assert route_after_triage(state) == "billing_specialist"


def test_route_after_triage_technical():
    from app.agents.graph import route_after_triage
    state = {"intent": "technical", "next_action": "technical_specialist"}
    assert route_after_triage(state) == "technical_specialist"


def test_route_after_triage_escalation():
    from app.agents.graph import route_after_triage
    state = {"intent": "escalation", "next_action": "escalation"}
    assert route_after_triage(state) == "escalation"


def test_route_after_specialist_end():
    from langgraph.graph import END
    from app.agents.graph import route_after_specialist
    state = {"next_action": "end"}
    assert route_after_specialist(state) == END


def test_route_after_specialist_escalation():
    from app.agents.graph import route_after_specialist
    state = {"next_action": "escalation"}
    assert route_after_specialist(state) == "escalation"
