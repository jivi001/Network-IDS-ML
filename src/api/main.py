from __future__ import annotations

import os
import time

from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from src.api.schemas import ExplainRequest, PredictRequest, PredictResponse
from src.explainability.shap_explainer import ShapExplainer
from src.inference.inference import InferenceService
from src.utils.logging_config import configure_logging

load_dotenv()
logger = configure_logging("api")
request_latency = Histogram(
    "nids_request_latency_seconds", "NIDS API request latency", ["endpoint"]
)
prediction_counter = Counter(
    "nids_predictions_total", "Prediction class distribution", ["predicted_class"]
)

app = FastAPI(title="NIDS SOC API", version="2.0.0")


@app.on_event("startup")
async def startup_event() -> None:
    artifact_dir = os.getenv("MODEL_ARTIFACT_DIR", "experiments/latest")
    app.state.service = InferenceService(artifact_dir)
    app.state.explainer = ShapExplainer(app.state.service.model)


@app.middleware("http")
async def capture_latency(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    request_latency.labels(endpoint=request.url.path).observe(
        time.perf_counter() - start
    )
    return response


@app.exception_handler(ValueError)
async def handle_value_error(_: Request, exc: ValueError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "model_version": app.state.service.metadata["model_version"],
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    try:
        results = app.state.service.predict(request.records, request.threshold_override)
        for result in results:
            prediction_counter.labels(
                predicted_class=str(result["predicted_class"])
            ).inc()
        return PredictResponse(results=results)
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err)) from err


@app.post("/predict/batch", response_model=PredictResponse)
async def predict_batch(request: PredictRequest) -> PredictResponse:
    return await predict(request)


@app.post("/explain")
async def explain(request: ExplainRequest) -> dict[str, object]:
    import pandas as pd

    frame = pd.DataFrame([request.record])
    top_features = app.state.explainer.explain_top_features(frame, request.top_k)
    return {"top_features": top_features}


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    return PlainTextResponse(
        generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST
    )
