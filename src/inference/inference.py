from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.inference.input_validator import validate_payload


class InferenceService:
    """Production inference service with schema enforcement and structured response."""

    def __init__(self, artifact_dir: str | Path):
        self.artifact_dir = Path(artifact_dir)
        self.model = self.load_model()
        self.metadata = self._load_metadata()
        self.expected_features: list[str] = self.metadata["features"]

    def load_model(self) -> Any:
        model_path = self.artifact_dir / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        return joblib.load(model_path)

    def _load_metadata(self) -> dict[str, Any]:
        metadata_path = self.artifact_dir / "metadata.json"
        with metadata_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def predict(
        self, payload: list[dict[str, Any]], threshold_override: float | None = None
    ) -> list[dict[str, Any]]:
        frame = pd.DataFrame(payload)
        validation = validate_payload(frame, self.expected_features)
        if not validation.is_valid:
            raise ValueError(
                {"error_code": "INPUT_VALIDATION_FAILED", "details": validation.errors}
            )

        threshold = (
            float(threshold_override)
            if threshold_override is not None
            else float(self.metadata["threshold"])
        )
        ordered = frame[self.expected_features]
        probabilities = self.model.predict_proba(ordered)[:, 1]
        predictions = (probabilities >= threshold).astype(int)

        return [
            {
                "predicted_class": int(predictions[idx]),
                "probability": float(probabilities[idx]),
                "anomaly_score": float(probabilities[idx] - threshold),
                "threshold_used": threshold,
                "model_version": self.metadata["model_version"],
            }
            for idx in np.arange(len(payload))
        ]
