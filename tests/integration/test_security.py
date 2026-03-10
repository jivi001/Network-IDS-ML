import pytest
import os
from fastapi.testclient import TestClient

# Must set the environment var before app load happens in test
os.environ["NIDS_API_KEY"] = "super_secret_test_key"
from deployment.inference_api import app

client = TestClient(app)

def test_missing_api_key_rejected():
    """Predict without matching X-API-Key should 401."""
    # Just sending payload with no headers
    response = client.post("/predict", json={
        "features": [0.0] * 20
    })
    assert response.status_code == 401
    
def test_valid_api_key_but_malformed_payload():
    """Valid keys should pass auth, but be blocked by strict Pydantic rules."""
    response = client.post("/predict", 
        json={"features": []},  # Under minimum length 1
        headers={"X-API-Key": "super_secret_test_key"}
    )
    assert response.status_code == 422
    
def test_rate_limit_exceeded():
    """Ensure sending 601 requests trips the rate limiter."""
    # We'll mock the rate limit state temporarily to avoid slow tests
    import deployment.inference_api as api
    import time
    
    # Fill the rate limit bucket for test client ip "testclient"
    api.RATE_LIMIT_DICT["testclient"] = [time.time()] * 600
    
    response = client.post("/predict", 
        json={"features": [0.0]*20},
        headers={"X-API-Key": "super_secret_test_key"}
    )
    assert response.status_code == 429
    assert response.json()["error"] == "Rate limit exceeded"
