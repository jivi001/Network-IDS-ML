from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


def persist_training_bundle(
    output_dir: Path,
    pipeline: Any,
    metadata: dict[str, Any],
) -> None:
    """Persist model, metadata and metrics for immutable deployment artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_dir / "model.joblib")

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata.get("metrics", {}), handle, indent=2)
    with (output_dir / "feature_list.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata.get("features", []), handle, indent=2)
    with (output_dir / "config_snapshot.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata.get("config", {}), handle, indent=2)


def log_to_mlflow(
    tracking_uri: str,
    experiment_name: str,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    artifacts_dir: Path,
    model: Any,
) -> str:
    """Log run, metrics and serialized model to MLflow."""
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("mlflow is required for experiment tracking") from exc

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)
        mlflow.log_artifacts(str(artifacts_dir), artifact_path="artifacts")
        mlflow.sklearn.log_model(model, artifact_path="model")
        return run.info.run_id
