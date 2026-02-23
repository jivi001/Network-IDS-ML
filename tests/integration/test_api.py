"""
Integration tests for the FastAPI inference API (/health, /predict, /predict/batch).
Uses FastAPI's TestClient — no live server needed.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


# ─── Mock the InferencePipeline so tests run without a real model on disk ─────

def _make_mock_pipeline():
    mock = MagicMock()
    mock.predict_single.return_value = {
        "prediction": "Normal",
        "confidence": 0.97,
        "tier_used": 1,
        "anomaly_score": -0.15,
    }
    mock.predict_batch.return_value = {
        "predictions": ["Normal", "DoS"],
        "tier_used": [1, 1],
        "anomaly_scores": [-0.15, -0.05],
    }
    return mock


@pytest.fixture(scope="module")
def client():
    """Create a TestClient with the InferencePipeline mocked out."""
    import sys, importlib

    mock_pipeline = _make_mock_pipeline()

    # Patch InferencePipeline before importing the app module
    with patch("deployment.inference_api.InferencePipeline", return_value=mock_pipeline):
        with patch("deployment.inference_api.Path.exists", return_value=True):
            # Need to force re-import of the app module with patched deps
            if "deployment.inference_api" in sys.modules:
                del sys.modules["deployment.inference_api"]
            import deployment.inference_api as api_module
            api_module.pipeline = mock_pipeline
            yield TestClient(api_module.app)


SAMPLE_FEATURES = [0.1] * 20


# ─── Health check ─────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_response_schema(self, client):
        resp = client.get("/health")
        body = resp.json()
        assert "status" in body
        assert "model_loaded" in body
        assert body["status"] == "healthy"

    def test_health_uptime_positive(self, client):
        resp = client.get("/health")
        assert resp.json()["uptime_seconds"] >= 0


# ─── /predict ─────────────────────────────────────────────────────────────────

class TestPredictEndpoint:
    def test_predict_returns_200(self, client):
        resp = client.post("/predict", json={"features": SAMPLE_FEATURES})
        assert resp.status_code == 200

    def test_predict_response_schema(self, client):
        resp = client.post("/predict", json={"features": SAMPLE_FEATURES})
        body = resp.json()
        assert "prediction" in body
        assert "confidence" in body
        assert "tier_used" in body
        assert "anomaly_score" in body

    def test_predict_returns_string_label(self, client):
        resp = client.post("/predict", json={"features": SAMPLE_FEATURES})
        assert isinstance(resp.json()["prediction"], str)

    def test_predict_missing_features_returns_422(self, client):
        resp = client.post("/predict", json={})
        assert resp.status_code == 422

    def test_predict_empty_features_returns_422(self, client):
        resp = client.post("/predict", json={"features": []})
        assert resp.status_code == 422

    def test_predict_503_when_no_model(self, client):
        import deployment.inference_api as api
        original = api.pipeline
        api.pipeline = None
        try:
            resp = client.post("/predict", json={"features": SAMPLE_FEATURES})
            assert resp.status_code == 503
        finally:
            api.pipeline = original


# ─── /predict/batch ───────────────────────────────────────────────────────────

class TestBatchPredictEndpoint:
    def test_batch_predict_returns_200(self, client):
        resp = client.post("/predict/batch", json={"features": [SAMPLE_FEATURES, SAMPLE_FEATURES]})
        assert resp.status_code == 200

    def test_batch_response_schema(self, client):
        resp = client.post("/predict/batch", json={"features": [SAMPLE_FEATURES, SAMPLE_FEATURES]})
        body = resp.json()
        assert "predictions" in body
        assert "tier_used" in body
        assert "anomaly_scores" in body
        assert "count" in body

    def test_batch_count_matches_input(self, client):
        features = [SAMPLE_FEATURES] * 3
        resp = client.post("/predict/batch", json={"features": features})
        assert resp.json()["count"] == 3

    def test_batch_empty_features_returns_422(self, client):
        resp = client.post("/predict/batch", json={"features": []})
        assert resp.status_code == 422
