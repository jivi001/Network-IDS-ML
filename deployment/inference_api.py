"""
REST API for NIDS inference service.
Built with FastAPI for automatic Swagger docs, Pydantic validation, and async support.
Access docs at: http://localhost:8000/docs
"""

import logging
import os
import sys
import time
import uuid
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from nids.utils.logging import setup_logger
from nids.governance.metrics_exporter import (
    INFERENCE_REQUEST_COUNT,
    INFERENCE_LATENCY,
    ANOMALY_SCORE_GAUGE,
    MODEL_VERSION_INFO
)
from prometheus_client import make_asgi_app

sys.path.insert(0, str(Path(__file__).parent.parent))
from nids.pipelines import InferencePipeline

# SOC JSON Logger
logger = setup_logger("nids.api")

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="NIDS Inference API",
    description="Hybrid Network Intrusion Detection System — REST inference endpoint.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Expose Prometheus Metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

@app.middleware("http")
async def soc_resilience_middleware(request: Request, call_next):
    if request.url.path == "/metrics":
        return await call_next(request)
        
    req_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        response = await call_next(request)
        status = response.status_code
    except ValueError as ve:
        status = 422
        logger.warning(f"Validation Failure: {str(ve)}", extra={"request_id": req_id})
        return JSONResponse(status_code=422, content={"error": str(ve), "request_id": req_id})
    except Exception as e:
        status = 500
        logger.error(f"Inference Crash: {str(e)}", extra={"request_id": req_id}, exc_info=True)
        return JSONResponse(status_code=500, content={"error": "Internal Model Error", "request_id": req_id})
    finally:
        latency = (time.time() - start_time) * 1000
        logger.info(
            "inference_audit", 
            extra={
                "request_id": req_id,
                "path": request.url.path,
                "method": request.method,
                "status_code": status,
                "latency_ms": round(latency, 2)
            }
        )
    return response

# ─── Startup: load model ───────────────────────────────────────────────────────
MODEL_DIR = os.getenv("MODEL_DIR", "models/production/v1.0.0")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0.0")
pipeline: InferencePipeline | None = None

def emit_telemetry(features: np.ndarray, anomaly_scores: np.ndarray, req_id: str):
    """Emits distribution metrics to SIEM for offline KS-test drift detection."""
    mean_score = float(np.mean(anomaly_scores))
    ANOMALY_SCORE_GAUGE.set(mean_score)
    
    logger.info(
        "model_telemetry",
        extra={
            "request_id": req_id,
            "iforest_score_mean": mean_score,
            "iforest_score_min": float(np.min(anomaly_scores)),
            "feature_norm": float(np.linalg.norm(features))
        }
    )

@app.on_event("startup")
async def startup_event():
    global pipeline
    MODEL_VERSION_INFO.labels(version=MODEL_VERSION).set(1)
    
    model_path = Path(MODEL_DIR)
    if not model_path.exists():
        logger.warning(
            f"Model directory '{MODEL_DIR}' not found. "
            "API will start but /predict will return 503 until a model is loaded."
        )
        return
    try:
        pipeline = InferencePipeline(model_dir=str(model_path))
        logger.info(f"Model loaded from: {model_path}")
    except Exception as exc:
        logger.error(f"Failed to load model: {exc}")


# ─── Schemas ──────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    features: List[float] = Field(
        ...,
        description="Network traffic feature vector (must match training feature count).",
        example=[0.0, 1.0, 0.5, 2.3, 0.1, 0.0, 1.0, 0.5, 2.3, 0.1,
                 0.0, 1.0, 0.5, 2.3, 0.1, 0.0, 1.0, 0.5, 2.3, 0.1],
    )

    @validator("features")
    def features_must_not_be_empty(cls, v):
        if len(v) == 0:
            raise ValueError("features list must not be empty")
        return v


class PredictResponse(BaseModel):
    prediction: str = Field(..., description="Predicted traffic class (e.g. 'Normal', 'DoS', 'Zero_Day_Anomaly')")
    confidence: float = Field(..., description="Max class probability from Tier-1 Random Forest (0–1)")
    tier_used: int = Field(..., description="Detection tier that produced the final result (1=RF, 2=IForest)")
    anomaly_score: float = Field(..., description="Isolation Forest anomaly score (more negative = more anomalous)")


class BatchPredictRequest(BaseModel):
    features: List[List[float]] = Field(
        ...,
        description="List of feature vectors for batch inference.",
    )

    @validator("features")
    def batch_must_not_be_empty(cls, v):
        if len(v) == 0:
            raise ValueError("features list must not be empty")
        return v


class BatchPredictResponse(BaseModel):
    predictions: List[str]
    tier_used: List[int]
    anomaly_scores: List[float]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_dir: str
    uptime_seconds: float


# ─── Middleware ────────────────────────────────────────────────────────────────
_start_time = time.time()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({duration:.3f}s)")
    return response


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Liveness / readiness probe. Returns 200 when the service is up."""
    return HealthResponse(
        status="healthy",
        model_loaded=pipeline is not None,
        model_dir=MODEL_DIR,
        uptime_seconds=round(time.time() - _start_time, 2),
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(body: PredictRequest):
    """
    Classify a single network traffic sample.

    - **Tier 1** (Random Forest) handles known attack categories.
    - **Tier 2** (Isolation Forest) flags zero-day anomalies.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check MODEL_DIR env variable.")

    try:
        result = pipeline.predict_single(body.features)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    return PredictResponse(**result)


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
async def predict_batch(body: BatchPredictRequest):
    """
    Classify a batch of network traffic samples in a single request.
    More efficient than calling `/predict` in a loop.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check MODEL_DIR env variable.")

    try:
        features = np.array(body.features, dtype=float)
        result = pipeline.predict_batch(features)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {exc}")

    return BatchPredictResponse(
        predictions=result["predictions"],
        tier_used=result["tier_used"],
        anomaly_scores=result["anomaly_scores"],
        count=len(result["predictions"]),
    )


# ─── SHAP Explainability ──────────────────────────────────────────────────────

class ExplainResponse(BaseModel):
    prediction: str
    confidence: float
    feature_contributions: dict = Field(
        ..., description="Per-feature SHAP values explaining this prediction"
    )


@app.post("/explain", response_model=ExplainResponse, tags=["Explainability"])
async def explain(body: PredictRequest):
    """
    Explain a single prediction using SHAP values.

    Returns the prediction alongside per-feature importance values
    showing how much each feature pushed the prediction toward or away
    from the predicted class.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        from nids.explainability import SHAPExplainer

        # Get prediction first
        result = pipeline.predict_single(body.features)

        # Preprocess the features the same way as predict_single
        import pandas as pd
        features = pd.DataFrame(
            [body.features],
            columns=pipeline.preprocessor.get_feature_names()
        )
        X_proc = pipeline.preprocessor.transform(features)
        X_sel = pipeline.selector.transform(X_proc)

        # Compute SHAP values
        explainer = SHAPExplainer()
        feature_names = pipeline.selector.get_selected_names()
        shap_values = explainer.explain_prediction(
            pipeline.model.tier1_model.model, X_sel, feature_names
        )

        # Sort by absolute SHAP value
        sorted_shap = dict(
            sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
        )

        return ExplainResponse(
            prediction=result["prediction"],
            confidence=result["confidence"],
            feature_contributions=sorted_shap,
        )

    except ImportError:
        raise HTTPException(
            status_code=501, detail="SHAP not installed. pip install shap"
        )
    except Exception as exc:
        logger.exception("Explain failed")
        raise HTTPException(status_code=500, detail=f"Explanation error: {exc}")


# ─── Dev entry-point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("inference_api:app", host="0.0.0.0", port=8000, reload=False)
