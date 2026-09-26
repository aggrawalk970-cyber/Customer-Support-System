import pytest
from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage
from app.agents.memory import count_tokens, trim_memory_node
import app.core.config

def test_count_tokens():
    """Test that the tiktoken integration accurately counts message length."""
    messages = [HumanMessage(content="Hello world, this is a test of token counting.")]
    tokens = count_tokens(messages)
    assert tokens > 5

def test_trim_memory_node(monkeypatch):
    """Test that trim_memory_node yields RemoveMessage when MAX_TOKENS is exceeded."""
    # Temporarily force MAX_TOKENS to be very small for the test
    monkeypatch.setattr(app.core.config.settings, "MAX_TOKENS", 10)
    
    msg1 = HumanMessage(content="This is a very very long message that definitely exceeds ten tokens easily and should be truncated out of memory.", id="msg1")
    msg2 = AIMessage(content="Yes, understood.", id="msg2")
    msg3 = HumanMessage(content="Short.", id="msg3")
    
    messages = [msg1, msg2, msg3]
    state = {"messages": messages}
    
    result = trim_memory_node(state)
    
    # It should return a dict containing 'messages' with RemoveMessage objects
    assert "messages" in result
    removed_ids = [m.id for m in result["messages"] if isinstance(m, RemoveMessage)]
    
    # The oldest and longest message (msg1) should have been targeted for removal
    assert "msg1" in removed_ids
    # The newest message should definitely NOT be targeted for removal
    assert "msg3" not in removed_ids
