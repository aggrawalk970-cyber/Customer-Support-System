import pytest
import uuid
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings
from unittest.mock import patch

client = TestClient(app)

def get_headers():
    if settings.API_KEYS:
        valid_key = settings.API_KEYS.split(",")[0].strip()
        return {"X-API-Key": valid_key}
    return {}

def test_rate_limit_enforcement():
    """
    Test that making more than 20 requests to /api/v1/chat within a minute triggers a 429.
    """
    headers = get_headers()
    
    # We will use a VALID payload so it passes guardrails
    payload = {
        "thread_id": str(uuid.uuid4()),
        "message": "hello this is a valid message" 
    }
    
    responses = []
    
    with patch("app.api.v1.chat.compiled_graph.invoke"), patch("app.api.v1.chat.compiled_graph.get_state") as mock_state:
        # Mock state so it seems like a valid empty or running thread
        class MockState:
            values = {}
            next = []
        mock_state.return_value = MockState()
        
        # The limit is 20/minute. We make 22 requests.
        for _ in range(22):
            resp = client.post("/api/v1/chat", headers=headers, json=payload)
            responses.append(resp.status_code)
        
    # At least the last one should be 429 Too Many Requests
    assert 429 in responses, "Rate limit of 20/min was not enforced!"
