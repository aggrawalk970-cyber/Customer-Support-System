import pytest
import uuid
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings

client = TestClient(app)

def get_headers():
    if settings.API_KEYS:
        valid_key = settings.API_KEYS.split(",")[0].strip()
        return {"X-API-Key": valid_key}
    return {}

def test_approve_flow_missing_thread():
    """
    Test that trying to approve/reject a non-existent thread 
    gracefully fails with a 404 or 400 error rather than crashing.
    """
    fake_thread = str(uuid.uuid4())
    
    response = client.post(
        f"/api/v1/approvals/{fake_thread}/resolve",
        headers=get_headers(),
        json={
            "approve": True,
            "feedback": "Approved by testing suite."
        }
    )
    
    # The API should reject this because the LangGraph state for this UUID doesn't exist
    assert response.status_code in [404, 400]
    
def test_approve_invalid_action():
    """Test that sending an invalid payload fails validation."""
    fake_thread = str(uuid.uuid4())
    
    response = client.post(
        f"/api/v1/approvals/{fake_thread}/resolve",
        headers=get_headers(),
        json={
            "approve": "not_a_boolean",
        }
    )
    
    # Pydantic should catch the invalid boolean validation
    assert response.status_code == 422
