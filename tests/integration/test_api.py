import pytest

pytest.importorskip("fastapi")

import json
import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier


def test_health_endpoint(tmp_path: Path) -> None:
    model = DummyClassifier(strategy="prior")
    train = pd.DataFrame([{"f1": 0, "f2": 1}, {"f1": 1, "f2": 0}])
    model.fit(train, [0, 1])
    artifact_dir = tmp_path / "model"
    artifact_dir.mkdir()
    joblib.dump(model, artifact_dir / "model.joblib")
    (artifact_dir / "metadata.json").write_text(
        json.dumps(
            {"features": ["f1", "f2"], "threshold": 0.5, "model_version": "test"}
        ),
        encoding="utf-8",
    )

    os.environ["MODEL_ARTIFACT_DIR"] = str(artifact_dir)

    from src.api.main import app

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_version"] == "test"
