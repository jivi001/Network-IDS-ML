import pytest
from fastapi.testclient import TestClient
from deployment.inference_api import app

client = TestClient(app)

def test_health_endpoint_smoke():
    """
    Basic smoke test to ensure the API starts up and the health endpoint is reachable.
    Used primarily by the CI/CD pipeline to verify the Docker image containerizes cleanly.
    """
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert "uptime_seconds" in data
