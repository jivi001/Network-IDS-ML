import time
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends, Security, Request
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from prometheus_client import make_asgi_app, Counter, Histogram

from ai_nids.models.hybrid_detector import HybridNIDS
from ai_nids.explainability.explainer import SHAPExplainer

# ─── Configuration & Telemetry ───────────────────────────────────────────────

app = FastAPI(
    title="AI-NIDS Production Platform",
    description="Tier 1 (Stacking) + Tier 2 (VAE) Inference & Streaming API",
    version="2.0.0"
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus Metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
INF_COUNT = Counter('inference_requests_total', 'Total inference requests')
INF_LATENCY = Histogram('inference_latency_seconds', 'Latency of /predict')
ANOMALY_ALERTS = Counter('anomaly_alerts_total', 'Zero-day anomalies detected')

import os
import logging

logger = logging.getLogger("nids")
logger.setLevel(logging.INFO)

# Security
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

API_KEY = os.getenv("NIDS_API_KEY", "default-dev-key")

def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid credentials")
    return api_key_header

# ─── Models ─────────────────────────────────────────────────────────────────

import mlflow.pyfunc

model = HybridNIDS() # Fallback skeleton
explainer = SHAPExplainer()

@app.on_event("startup")
async def startup_event():
    global model
    try:
        model = mlflow.pyfunc.load_model("models:/NIDS-Production/Production")
    except Exception as e:
        logger.warning(f"Could not load MLflow model: {e}")

EXPECTED_FEATURES = 49

class FeatureVector(BaseModel):
    features: List[float]

class BatchFeatureVector(BaseModel):
    features_batch: List[List[float]]

class InferenceResponse(BaseModel):
    attack_probability: float
    attack_type: str
    anomaly_score: float
    confidence: float
    tier_used: int

# ─── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=InferenceResponse)
@limiter.limit("100/second")
async def predict_single(request: Request, data: FeatureVector, api_key: str = Depends(get_api_key)):
    logger.info("Prediction request received")
    if len(data.features) != EXPECTED_FEATURES:
        raise HTTPException(status_code=400, detail="Invalid feature vector size")
        
    start_time = time.time()
    INF_COUNT.inc()
    
    try:
        import numpy as np
        X = np.array([data.features])
        result = model.predict(X)
        
        if result['attack_type'][0] == "Zero_Day_Anomaly":
            ANOMALY_ALERTS.inc()
            
        INF_LATENCY.observe(time.time() - start_time)
        return InferenceResponse(
            attack_probability=float(result['tier1_proba'][0].max()),
            attack_type=str(result['attack_type'][0]),
            anomaly_score=float(result['anomaly_score'][0]),
            confidence=float(result['tier1_proba'][0].max()),
            tier_used=int(result['tier_used'][0])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=List[InferenceResponse])
@limiter.limit("50/second")
async def predict_batch(request: Request, data: BatchFeatureVector, api_key: str = Depends(get_api_key)):
    logger.info(f"Batch prediction request received for {len(data.features_batch)} samples")
    for f in data.features_batch:
        if len(f) != EXPECTED_FEATURES:
            raise HTTPException(status_code=400, detail="Invalid feature vector size in batch")

    start_time = time.time()
    INF_COUNT.inc(len(data.features_batch))
    
    try:
        import numpy as np
        X = np.array(data.features_batch)
        result = model.predict(X)
        
        responses = []
        for i in range(len(data.features_batch)):
            if result['attack_type'][i] == "Zero_Day_Anomaly":
                ANOMALY_ALERTS.inc()
            responses.append(InferenceResponse(
                attack_probability=float(result['tier1_proba'][i].max()),
                attack_type=str(result['attack_type'][i]),
                anomaly_score=float(result['anomaly_score'][i]),
                confidence=float(result['tier1_proba'][i].max()),
                tier_used=int(result['tier_used'][i])
            ))
        INF_LATENCY.observe((time.time() - start_time) / len(data.features_batch))
        return responses
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain")
async def explain_prediction(data: FeatureVector, api_key: str = Depends(get_api_key)):
    """Returns local SHAP explanations for SOC analysts."""
    import numpy as np
    X = np.array([data.features])
    base_prediction = model.predict(X)
    
    # Generate SHAP explanation using the Tier 1 model
    explanation = explainer.explain(model.tier1_ensemble.model, X)
    
    return {
        "prediction": base_prediction['attack_type'][0],
        "shap_values": explanation
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
