import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier

from src.inference.inference import InferenceService


def test_inference_service_predict(tmp_path: Path) -> None:
    model = DummyClassifier(strategy="prior")
    train = pd.DataFrame([{"f1": 0, "f2": 1}, {"f1": 1, "f2": 0}])
    model.fit(train, [0, 1])

    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    joblib.dump(model, artifact_dir / "model.joblib")
    (artifact_dir / "metadata.json").write_text(
        json.dumps({"features": ["f1", "f2"], "threshold": 0.5, "model_version": "v1"}),
        encoding="utf-8",
    )

    service = InferenceService(artifact_dir)
    output = service.predict([{"f1": 1, "f2": 0}], threshold_override=0.4)
    assert output[0]["model_version"] == "v1"
    assert "probability" in output[0]
