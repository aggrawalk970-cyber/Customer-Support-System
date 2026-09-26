from app.agents.graph import route_after_triage
from app.agents.state import AgentState

def test_triage_routing_billing():
    """Test that billing intent routes to billing_specialist."""
    state = {"intent": "billing", "messages": []}
    result = route_after_triage(state)
    assert result == "billing_specialist"

def test_triage_routing_technical():
    """Test that technical intent routes to technical_specialist."""
    state = {"intent": "technical", "messages": []}
    result = route_after_triage(state)
    assert result == "technical_specialist"

def test_triage_routing_escalate():
    """Test that escalate intent routes to escalation node."""
    state = {"intent": "escalation", "messages": []}
    result = route_after_triage(state)
    assert result == "escalation"
    
def test_triage_routing_unknown_to_general():
    """Test that fallback or unknown intent routes to general_specialist."""
    state = {"intent": "gibberish", "messages": []}
    result = route_after_triage(state)
    assert result == "general_specialist"
